"""
new_v1 / new_v2 데이터셋 "사람이 직접 열어볼 수 있는" 샘플 추출 스크립트
=========================================================================

목적
----
baseline `_Open(Sample)/B1.2_raw/` 폴더(=`export_raw_data.py` 가 만든 것)와
**같은 디렉토리 레이아웃**으로 신규 데이터셋(new_v1, new_v2) 의 일부(샘플) 를
CSV / NPY / NPZ / TIFF 로 dump 한다.

  ORNL_Data/.../[new_v1] ..._Open(Sample)/N1_raw/
  ORNL_Data/.../[new_v2] ..._Open(Sample)/N2_raw/
    EXPORT_SUMMARY.txt
    metadata.json
    tensile_results_*.csv          (new_v2 만)
    process_parameters.csv         (new_v2 만)
    temporal_data/<sensor>.csv     (모든 센서 개별 CSV)
    temporal_data_combined.csv     (동일 길이 센서 합본)
    camera_images/                 (5 layer × 2 camera, NPY+TIFF + image_info.csv)
    segmentation/                  (5 layer; new_v2: 12 클래스 NPZ; new_v1: 단일 categorical NPZ + counts.csv)
    id_maps/                       (5 layer; new_v1: part_ids 만; new_v2: part+sample)
    scan_paths/                    (new_v2 만; 5 layer NPY+CSV)

baseline 패턴 재사용 / 어댑테이션
----------------------------------
- 재사용: 폴더 레이아웃, 파일명 컨벤션 (`layer_{:05d}_*.npy/.npz`),
          metadata.json 형식, EXPORT_SUMMARY.txt 의 DSCNN 12-class 표,
          temporal CSV 의 `index=layer` 포맷, scan path 5-column CSV header.
- 어댑테이션 (new_v1):
    * 카메라: visible 부재 → NIR/{0,1} 사용. suffix 가 `NIR_0`/`NIR_1`.
    * segmentation: 12 bool 그룹 부재 → 단일 (L,1900,1900) uint8 categorical.
      → 레이어당 단일 NPZ (key='seg') + class_pixel_counts.csv (관측된 ID 만).
    * scans/samples/parts 그룹 자체 부재 → 해당 폴더/CSV 생성하지 않음.
    * id_maps: sample_ids 부재 → part_ids 만.
    * temporal: 18채널 EB-PBF 전용 (beam_current, *vacuum_gauge_fb 등).
- 어댑테이션 (new_v2):
    * 카메라: visible/{0,1} 정상 → baseline 과 동일 `post_melt`/`post_powder`.
    * segmentation: baseline 과 동일 12 bool.
    * scans 정상.
    * samples/test_results: `pycnometry_density` 만.
      → `samples_test_results.csv` (이름 misleading 회피).
    * parts/test_results: YS/UTS/TE 만 (UE 부재). `_valid` 컬럼 추가.
    * process_parameters: baseline 키와 다름. 실제 키 그대로 dump.
      (mapping 표는 EXPORT_SUMMARY 에 적음.)
    * id_maps: part+sample 둘 다.
    * temporal: 19채널.

CLI
---
  --dataset {new_v1, new_v2, both}    (기본 both)
  --sample-layers N                   (기본 5)
  --output-base PATH                  (기본은 각 데이터셋의 _Open(Sample)/N{1,2}_raw/)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_BASE = PROJECT_ROOT / "ORNL_Data" / "Co-Registered In-Situ and Ex-Situ Dataset"

NEW_V1_DIR = DATA_BASE / "[new_v1] (Peregrine v2023-09)"
NEW_V2_DIR = DATA_BASE / "[new_v2] (Peregrine v2023-10)"

NEW_V1_HDF5 = NEW_V1_DIR / "2018-12-31 037-R1256_BLADE ITERATION 13_5.hdf5"
NEW_V2_HDF5 = NEW_V2_DIR / "2023-03-15 AMMTO Spatial Variation Baseline.hdf5"

NEW_V1_OUT_DEFAULT = DATA_BASE / "[new_v1] (Peregrine v2023-09)_Open(Sample)" / "N1_raw"
NEW_V2_OUT_DEFAULT = DATA_BASE / "[new_v2] (Peregrine v2023-10)_Open(Sample)" / "N2_raw"

# baseline 의 DSCNN 12 클래스 (참고용 - new_v2 에 그대로, new_v1 categorical 은 매핑 가정 X)
DSCNN_CLASSES = {
    0: "Powder", 1: "Printed", 2: "Recoater Hopping", 3: "Recoater Streaking",
    4: "Incomplete Spreading", 5: "Swelling", 6: "Debris", 7: "Super-Elevation",
    8: "Spatter", 9: "Misprint", 10: "Over Melting", 11: "Under Melting",
}


# ---------------------------------------------------------------------------
# 공용 IO 유틸 (baseline export_raw_data.py 와 호환되는 파일 컨벤션)
# ---------------------------------------------------------------------------

def save_image_npy_tiff(img: np.ndarray, npy_path: Path, tiff_path: Path) -> None:
    """이미지를 NPY (raw) + TIFF (display 용) 둘 다 저장.

    Raw 값은 NPY 에 그대로 보존된다. TIFF 는 사람이 일반 뷰어에서 열어보는 용도이므로
    32-bit float (mode='F') 대신 baseline 과 동일하게 16-bit 정수(I;16) 로 저장한다.
    float32 입력은 percentile [0.5, 99.5] 클립 후 0-65535 로 linear scale 한다.
    """
    np.save(npy_path, img)

    if img.dtype == np.uint8 or img.dtype == np.uint16:
        Image.fromarray(img).save(tiff_path, format="TIFF")
        return

    # float / 기타: 디스플레이용 16-bit TIFF — percentile clip + linear scale
    finite = np.isfinite(img)
    if finite.any():
        lo, hi = np.percentile(img[finite], [0.5, 99.5])
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    scaled[~finite] = 0.0
    u16 = (scaled * 65535.0 + 0.5).astype(np.uint16)
    Image.fromarray(u16, mode="I;16").save(tiff_path, format="TIFF")


def save_attrs_to_metadata(f: h5py.File, build_id: str, num_layers: int,
                           extra: dict | None = None) -> dict:
    """root attrs 를 JSON 직렬화 가능한 dict 로."""
    md: dict = {}
    for k in f.attrs:
        v = f.attrs[k]
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            v = v.item()
        elif isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8", errors="ignore")
        md[k] = v
    md["_export_info"] = {
        "build_id": build_id,
        "num_layers": int(num_layers),
        "export_date": datetime.now().isoformat(),
        "export_type": "raw_sample",
    }
    if extra:
        md["_export_info"].update(extra)
    return md


def even_layer_indices(num_layers: int, n: int) -> list[int]:
    """np.linspace 로 n 개 균등 (반올림 정수)."""
    if num_layers <= 0 or n <= 0:
        return []
    if num_layers <= n:
        return list(range(num_layers))
    idx = np.linspace(0, num_layers - 1, n)
    return [int(round(x)) for x in idx]


def folder_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Common: temporal export (양쪽 데이터셋 동일 패턴)
#  - baseline 과 동일: temporal_data/<key>.csv 개별 + temporal_data_combined.csv
# ---------------------------------------------------------------------------

def export_temporal(f: h5py.File, out_dir: Path) -> tuple[int, int]:
    """temporal/* 모두 추출. (saved_individual, combined_cols) 반환."""
    if "temporal" not in f:
        return 0, 0
    tdir = out_dir / "temporal_data"
    tdir.mkdir(parents=True, exist_ok=True)

    keys = sorted(f["temporal"].keys())
    arrays: dict[str, np.ndarray] = {}
    saved = 0
    for k in keys:
        try:
            arr = f[f"temporal/{k}"][...]
        except Exception:
            continue
        # 길이 0 이상 1D 만 처리
        if arr.ndim != 1:
            continue
        arrays[k] = arr
        # 개별 CSV
        df = pd.DataFrame({k: arr})
        df.index.name = "layer"
        # 키 이름에 "[", "]" 같은 특수문자 (예: turbopump_[0]_current) 가 있을 수 있어
        # 파일시스템 안전 파일명으로 sanitize (단, 키 자체는 보존 — 컬럼명에 그대로 둠)
        safe = k.replace("[", "_").replace("]", "_").replace("/", "_")
        df.to_csv(tdir / f"{safe}.csv")
        saved += 1

    # combined: 동일 길이만 합치기
    if arrays:
        max_len = max(len(v) for v in arrays.values())
        combined = {k: v for k, v in arrays.items() if len(v) == max_len}
        if combined:
            df = pd.DataFrame(combined)
            df.index.name = "layer"
            df.to_csv(out_dir / "temporal_data_combined.csv")
            return saved, len(combined)
    return saved, 0


# ---------------------------------------------------------------------------
# Camera images
# ---------------------------------------------------------------------------

def export_cameras(f: h5py.File, out_dir: Path, layers: list[int],
                   camera_paths: list[tuple[str, str]]) -> int:
    """
    camera_paths: list of (h5_path, suffix). 예:
      new_v1: [("slices/camera_data/NIR/0", "NIR_0"), ("slices/camera_data/NIR/1", "NIR_1")]
      new_v2: [("slices/camera_data/visible/0", "post_melt"), ("slices/camera_data/visible/1", "post_powder")]
    """
    img_dir = out_dir / "camera_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 존재하는 path 만
    valid = [(hp, sf) for hp, sf in camera_paths if hp in f]
    if not valid:
        return 0

    info = []
    saved = 0
    for li, layer in enumerate(layers):
        for hp, suffix in valid:
            ds = f[hp]
            if layer >= ds.shape[0]:
                continue
            img = ds[layer, ...]
            base = f"layer_{layer:05d}_{suffix}"
            npy_path = img_dir / f"{base}.npy"
            tiff_path = img_dir / f"{base}.tiff"
            save_image_npy_tiff(img, npy_path, tiff_path)
            info.append({
                "layer": layer,
                "h5_path": hp,
                "suffix": suffix,
                "dtype": str(img.dtype),
                "shape": tuple(img.shape),
                "min": float(np.nanmin(img)) if np.isfinite(img).any() else float("nan"),
                "max": float(np.nanmax(img)) if np.isfinite(img).any() else float("nan"),
                "mean": float(np.nanmean(img)) if np.isfinite(img).any() else float("nan"),
            })
            saved += 1
        print(f"  [camera] layer {layer} done ({li+1}/{len(layers)})")

    if info:
        pd.DataFrame(info).to_csv(img_dir / "image_info.csv", index=False)
    return saved


# ---------------------------------------------------------------------------
# ID maps
# ---------------------------------------------------------------------------

def export_id_maps(f: h5py.File, out_dir: Path, layers: list[int],
                   include_sample_ids: bool) -> int:
    id_dir = out_dir / "id_maps"
    id_dir.mkdir(parents=True, exist_ok=True)

    if "slices/part_ids" not in f:
        return 0
    pids_ds = f["slices/part_ids"]
    sids_ds = f["slices/sample_ids"] if (include_sample_ids and "slices/sample_ids" in f) else None

    saved = 0
    for layer in layers:
        if layer >= pids_ds.shape[0]:
            continue
        d = {"part_ids": pids_ds[layer, ...]}
        if sids_ds is not None and layer < sids_ds.shape[0]:
            d["sample_ids"] = sids_ds[layer, ...]
        np.savez_compressed(id_dir / f"layer_{layer:05d}_ids.npz", **d)
        saved += 1
    return saved


# ---------------------------------------------------------------------------
# Segmentation: new_v2 (12 bool group) — baseline 과 동일
# ---------------------------------------------------------------------------

def export_segmentation_v2(f: h5py.File, out_dir: Path, layers: list[int]) -> int:
    seg_dir = out_dir / "segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # class_names.csv (baseline 와 동일)
    pd.DataFrame([
        {"class_id": k, "class_name": v} for k, v in DSCNN_CLASSES.items()
    ]).to_csv(seg_dir / "class_names.csv", index=False)

    saved = 0
    for layer in layers:
        seg_data = {}
        for c in range(12):
            key = f"slices/segmentation_results/{c}"
            if key in f:
                ds = f[key]
                if layer < ds.shape[0]:
                    seg_data[f"class_{c}"] = ds[layer, ...]
        if seg_data:
            np.savez_compressed(seg_dir / f"layer_{layer:05d}_segmentation.npz", **seg_data)
            saved += 1
    return saved


# ---------------------------------------------------------------------------
# Segmentation: new_v1 (single categorical map) — baseline 과 다름
# ---------------------------------------------------------------------------

def export_segmentation_v1(f: h5py.File, out_dir: Path, layers: list[int]) -> int:
    """new_v1: slices/segmentation_results 가 단일 (L,H,W) uint8 categorical map.
    레이어당 NPZ (key='seg') + class_pixel_counts.csv (관측된 unique IDs 만)."""
    seg_dir = out_dir / "segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)

    if "slices/segmentation_results" not in f:
        return 0
    ds = f["slices/segmentation_results"]
    if not isinstance(ds, h5py.Dataset):
        return 0

    # 클래스 매핑은 알 수 없음 — caveat 로 적기. 단 baseline DSCNN 표는 참조용으로 같이.
    pd.DataFrame([
        {"class_id": k, "class_name_baseline_DSCNN": v} for k, v in DSCNN_CLASSES.items()
    ]).to_csv(seg_dir / "baseline_dscnn_classes_for_reference.csv", index=False)

    counts_rows = []
    saved = 0
    for layer in layers:
        if layer >= ds.shape[0]:
            continue
        seg = ds[layer, ...]  # (H, W) uint8
        np.savez_compressed(seg_dir / f"layer_{layer:05d}_segmentation.npz", seg=seg)
        ids, cnts = np.unique(seg, return_counts=True)
        for i, c in zip(ids.tolist(), cnts.tolist()):
            counts_rows.append({
                "layer": int(layer),
                "class_id": int(i),
                "pixel_count": int(c),
            })
        saved += 1
    if counts_rows:
        pd.DataFrame(counts_rows).to_csv(seg_dir / "class_pixel_counts.csv", index=False)
    return saved


# ---------------------------------------------------------------------------
# Scans (new_v2 only; new_v1 has none)
# ---------------------------------------------------------------------------

def export_scans(f: h5py.File, out_dir: Path, layers: list[int]) -> int:
    if "scans" not in f:
        return 0
    scan_dir = out_dir / "scan_paths"
    scan_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for layer in layers:
        key = f"scans/{layer}"
        if key not in f:
            continue
        scan_data = f[key][...]  # (N, 5)
        np.save(scan_dir / f"layer_{layer:05d}_scanpath.npy", scan_data)
        if scan_data.ndim == 2 and scan_data.shape[1] == 5:
            df = pd.DataFrame(scan_data, columns=[
                "x_start", "x_end", "y_start", "y_end", "relative_time",
            ])
            df.to_csv(scan_dir / f"layer_{layer:05d}_scanpath.csv", index=False)
        saved += 1
    return saved


# ---------------------------------------------------------------------------
# Tensile / process params (new_v2 only)
# ---------------------------------------------------------------------------

def export_tensile_v2(f: h5py.File, out_dir: Path) -> dict:
    """new_v2 의 samples/parts test results CSV 로 저장. 결과 통계 dict 반환."""
    stats: dict = {}

    # samples/test_results — pycnometry_density 만
    if "samples/test_results" in f:
        keys = sorted(f["samples/test_results"].keys())
        if keys:
            d = {}
            for k in keys:
                d[k] = f[f"samples/test_results/{k}"][...]
            df = pd.DataFrame(d)
            df.index.name = "sample_id"
            # 이름이 misleading 안되게 'samples_test_results.csv'
            df.to_csv(out_dir / "samples_test_results.csv")
            stats["samples_test_results"] = {"keys": keys, "n": len(df)}

    # parts/test_results — YS/UTS/TE (UE 부재) + single_track_*. valid flag.
    if "parts/test_results" in f:
        keys_all = sorted(f["parts/test_results"].keys())
        # tensile (YS/UTS/UE/TE) 컬럼만 골라 tensile_results_parts.csv
        tensile_keys = [k for k in keys_all if k in {
            "yield_strength", "ultimate_tensile_strength",
            "uniform_elongation", "total_elongation",
        }]
        if tensile_keys:
            d = {}
            for k in tensile_keys:
                d[k] = f[f"parts/test_results/{k}"][...]
            df = pd.DataFrame(d)
            df.index.name = "part_id"
            # _valid: 모든 tensile 컬럼이 finite & != 0
            valid_mask = pd.Series(True, index=df.index)
            for c in tensile_keys:
                valid_mask &= np.isfinite(df[c]) & (df[c] != 0.0)
            df["_valid"] = valid_mask
            df.to_csv(out_dir / "tensile_results_parts.csv")
            stats["tensile_results_parts"] = {
                "keys": tensile_keys, "n": len(df),
                "n_valid": int(valid_mask.sum()),
                "absent_keys": sorted({"yield_strength", "ultimate_tensile_strength",
                                       "uniform_elongation", "total_elongation"} - set(tensile_keys)),
            }

        # nontensile parts/test_results (single_track_*) 도 같이 dump 하고 싶으면 별도 파일
        nontensile = [k for k in keys_all if k not in {
            "yield_strength", "ultimate_tensile_strength",
            "uniform_elongation", "total_elongation",
        }]
        if nontensile:
            d = {}
            for k in nontensile:
                d[k] = f[f"parts/test_results/{k}"][...]
            df = pd.DataFrame(d)
            df.index.name = "part_id"
            df.to_csv(out_dir / "parts_test_results_nontensile.csv")
            stats["parts_test_results_nontensile"] = {"keys": nontensile, "n": len(df)}

    return stats


def export_process_params_v2(f: h5py.File, out_dir: Path) -> dict:
    """new_v2 process_parameters: 실제 키 그대로 dump (baseline 과 키 이름 다름)."""
    stats: dict = {}
    if "parts/process_parameters" not in f:
        return stats

    keys = sorted(f["parts/process_parameters"].keys())
    d = {}
    for k in keys:
        arr = f[f"parts/process_parameters/{k}"][...]
        # bytes → utf-8
        if arr.dtype.kind in ("S", "O"):
            arr = np.array([
                v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else str(v)
                for v in arr
            ])
        d[k] = arr
    df = pd.DataFrame(d)
    df.index.name = "part_id"
    df.to_csv(out_dir / "process_parameters.csv")

    # parameter_set unique 통계
    n_param_sets = None
    if "parameter_set" in df.columns:
        n_param_sets = int(df["parameter_set"].nunique())
    stats["process_parameters"] = {"keys": keys, "n_parts": len(df), "n_param_sets": n_param_sets}
    return stats


# ---------------------------------------------------------------------------
# EXPORT_SUMMARY
# ---------------------------------------------------------------------------

DSCNN_TABLE_TEXT = "\n".join(f"  {cid}: {name}" for cid, name in DSCNN_CLASSES.items())


def write_summary_v1(out_dir: Path, build_id: str, build_name: str,
                     num_layers: int, sampled_layers: list[int],
                     temporal_keys: list[str], camera_layers_count: int,
                     seg_layers_count: int, id_layers_count: int) -> None:
    total_mb = folder_size_mb(out_dir)
    text = f"""
{'='*60}
ORNL Dataset Raw Data Export Summary (Sample)
{'='*60}

Build Information:
  Build ID: {build_id}
  Build Name: {build_name}
  Source: new_v1 (Peregrine v2023-09)
  Printer: Arcam Q10/Q10+ (Electron Beam Powder Bed Fusion)
  Material: Inconel 738
  Total Layers: {num_layers}
  Sampled Layers: {sampled_layers}

Export Information:
  Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Output Directory: {out_dir}
  Total Size: {total_mb:.2f} MB

File Formats:
  - metadata.json:                        Build root attrs (JSON)
  - temporal_data/<sensor>.csv:           Per-sensor 1D series, 18 EB-PBF channels
  - temporal_data_combined.csv:           Same-length sensors merged
  - camera_images/layer_*_NIR_0.{{npy,tiff}}, _NIR_1.{{npy,tiff}}:
      NIR camera frames (no visible-light cameras on this printer).
      NPY = raw float32 (HDF5 그대로). TIFF = 16-bit (I;16) display copy
      (percentile 0.5-99.5 clip + linear scale 0..65535).
  - segmentation/layer_*_segmentation.npz (key='seg' uint8):
      Single categorical map per layer (NOT 12 bool channels).
  - segmentation/class_pixel_counts.csv:  Per-(layer, class_id) pixel count
                                          (only IDs actually observed).
  - segmentation/baseline_dscnn_classes_for_reference.csv:
      Baseline DSCNN 12-class table for reference (DO NOT assume the
      categorical IDs in this dataset map to it — see Caveats).
  - id_maps/layer_*_ids.npz:              part_ids only (sample_ids absent).

Adaptations from baseline format:
  - Cameras: slices/camera_data/visible/* absent — used slices/camera_data/NIR/{{0,1}}
    (suffix 'NIR_0'/'NIR_1' instead of 'post_melt'/'post_powder').
  - Segmentation: 12 bool group (slices/segmentation_results/{{0..11}}) absent —
    instead a single uint8 categorical map (L, 1900, 1900). Stored as one NPZ
    per layer (key='seg') plus a per-(layer,class_id) pixel-count CSV.
  - 'scans' group not present — scan_paths/ folder NOT created.
  - 'samples' and 'parts' groups not present — no tensile_results_*.csv,
    no process_parameters.csv (no tensile labels, no per-part process params).
  - id_maps: slices/sample_ids absent — only part_ids exported.
  - Temporal: 18 EB-PBF-specific channels (beam_current, chamber_vacuum_gauge_fb,
    column_vacuum_gauge_fb, turbopump_[0]_current, turbopump_[1]_current,
    backing_vacuum_gauge_fb, smoke_detector_counts, table_current,
    layer_times, build_time, etc.) — does NOT match baseline 7-channel set.

Caveats for downstream use:
  - No tensile labels (YS/UTS/UE/TE) anywhere in this build — cannot train
    or evaluate tensile-property models on this dataset.
  - No per-part process parameters — process-parameter ablations not feasible.
  - No scan vector data — laser/beam scan-path features (G4 in baseline) cannot
    be extracted.
  - The categorical segmentation IDs are NOT guaranteed to match the baseline
    DSCNN 12-class scheme. Treat the IDs as opaque until verified against the
    Peregrine v2023-09 release notes / readme.pdf.
  - Camera modality is NIR (electron-beam-induced thermal/visible-NIR) — not
    visible-light photography. Pixel statistics and dynamic range differ from
    baseline visible/0 and visible/1.

Temporal channels exported: {len(temporal_keys)}
Camera frames exported: {camera_layers_count}
Segmentation layers exported: {seg_layers_count}
ID-map layers exported: {id_layers_count}

Baseline DSCNN classes (FOR REFERENCE ONLY — see Caveats):
{DSCNN_TABLE_TEXT}

{'='*60}
"""
    (out_dir / "EXPORT_SUMMARY.txt").write_text(text, encoding="utf-8")


def write_summary_v2(out_dir: Path, build_id: str, build_name: str,
                     num_layers: int, sampled_layers: list[int],
                     temporal_keys: list[str], camera_layers_count: int,
                     seg_layers_count: int, id_layers_count: int,
                     scan_layers_count: int, tensile_stats: dict,
                     proc_stats: dict) -> None:
    total_mb = folder_size_mb(out_dir)

    # parts tensile detail
    parts_info = tensile_stats.get("tensile_results_parts", {})
    samples_info = tensile_stats.get("samples_test_results", {})
    proc_info = proc_stats.get("process_parameters", {})

    # baseline → new_v2 process-param 키 매핑 표
    mapping_table = """
  baseline key (B1.*)        →  new_v2 key
  ---------------------------    ----------------------
  laser_beam_power           →  power
  laser_beam_speed           →  velocity
  hatch_spacing              →  hatch_space
  laser_spot_size            →  spot_size
  (none)                     →  parameter_set        (uint8, sweep id)
  (none)                     →  contours, supports, melting_strategy,
                                offset_to_cad_edge, stripe_angle_rotation,
                                stripe_overlap, stripe_width
"""

    text = f"""
{'='*60}
ORNL Dataset Raw Data Export Summary (Sample)
{'='*60}

Build Information:
  Build ID: {build_id}
  Build Name: {build_name}
  Source: new_v2 (Peregrine v2023-10)
  Printer: Concept Laser M2 (Laser Powder Bed Fusion)
  Material: SS 316L
  Total Layers: {num_layers}
  Sampled Layers: {sampled_layers}

Export Information:
  Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Output Directory: {out_dir}
  Total Size: {total_mb:.2f} MB

File Formats:
  - metadata.json:                          Build root attrs (JSON)
  - process_parameters.csv:                 Per-part process params (271 parts).
                                            Keys differ from baseline — see mapping below.
  - tensile_results_parts.csv:              Per-part YS / UTS / TE (uniform_elongation absent).
                                            Includes a `_valid` boolean column
                                            (finite & != 0 across all tensile cols).
  - parts_test_results_nontensile.csv:      single_track_{{depth,height,width}} per part.
  - samples_test_results.csv:               Per-sample pycnometry_density only
                                            (named to avoid 'tensile_results_samples'
                                            misnomer — no YS/UTS/UE/TE at sample level).
  - temporal_data/<sensor>.csv, temporal_data_combined.csv:
                                            19 sensor channels (baseline 7 + 12 extra).
  - camera_images/layer_*_post_melt.{{npy,tiff}}, _post_powder.{{npy,tiff}}:
                                            visible/0 = post_melt, visible/1 = post_powder.
                                            NPY = raw float32. TIFF = 16-bit (I;16) display copy
                                            (percentile 0.5-99.5 clip + linear scale).
  - segmentation/layer_*_segmentation.npz:  12 bool channels (class_0 .. class_11),
                                            same DSCNN scheme as baseline.
  - segmentation/class_names.csv:           DSCNN 12-class id ↔ name table.
  - id_maps/layer_*_ids.npz:                part_ids + sample_ids per layer.
  - scan_paths/layer_*_scanpath.{{npy,csv}}: per-layer (N, 5) scan vectors
                                            [x_start, x_end, y_start, y_end, relative_time].

Adaptations from baseline format:
  - Process-parameter keys renamed in this build. Mapping:{mapping_table}
    The CSV uses the **actual key names** present in the HDF5 (no rename).
    `parameter_set` (uint8, {proc_info.get('n_param_sets', 'N/A')} unique values) is exposed as a column.
  - Tensile labels at the *sample* level: only `pycnometry_density` is available
    (NO yield_strength / UTS / uniform_elongation / total_elongation under
    samples/test_results). The corresponding file is named
    `samples_test_results.csv` rather than `tensile_results_samples.csv` to
    avoid a misleading filename.
  - Tensile labels at the *part* level: YS / UTS / TE present
    ({parts_info.get('n_valid', 'N/A')} valid out of {parts_info.get('n', 'N/A')}); `uniform_elongation` key is
    absent in parts/test_results — column intentionally omitted (not NaN-filled).
  - In addition to tensile, parts/test_results also contains
    `single_track_{{depth,height,width}}` — exported separately as
    `parts_test_results_nontensile.csv`.

Caveats for downstream use:
  - This is a **single build (271 parts, 27 samples)**, not a multi-build series
    like B1.1–B1.5. Build-level cross-validation is not directly comparable.
  - `parameter_set` is a process-sweep label ({proc_info.get('n_param_sets', 'N/A')} unique values across
    271 parts). Many parts share the same parameter_set but only some have
    tensile labels — verify before stratifying.
  - `uniform_elongation` is absent at both sample and part levels. Models that
    predict the full {{YS, UTS, UE, TE}} 4-target tensor cannot be trained on
    this build without dropping UE.
  - Process-parameter keys do NOT match baseline by name. Any code that
    hardcodes 'laser_beam_power'/'laser_beam_speed'/'hatch_spacing'/'laser_spot_size'
    will silently miss values — adapt to {{power, velocity, hatch_space, spot_size}}.
  - sample_ids has many near-empty layers (sample volume is small relative to
    build height); scarce sample_ids ≠ data corruption.

Temporal channels exported: {len(temporal_keys)}
Camera frames exported: {camera_layers_count}
Segmentation layers exported: {seg_layers_count}
ID-map layers exported: {id_layers_count}
Scan-path layers exported: {scan_layers_count}

DSCNN classes (segmentation_results/{{0..11}}):
{DSCNN_TABLE_TEXT}

{'='*60}
"""
    (out_dir / "EXPORT_SUMMARY.txt").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-dataset orchestration
# ---------------------------------------------------------------------------

def export_new_v1(hdf5_path: Path, out_dir: Path, sample_layers: int) -> None:
    print("=" * 70)
    print(f"new_v1: exporting from {hdf5_path.name}")
    print(f"        → {out_dir}")
    print("=" * 70)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with h5py.File(hdf5_path, "r") as f:
        # 레이어 수 (파일에서 attrs 또는 dataset shape 로 확인)
        if "slices/part_ids" in f:
            num_layers = int(f["slices/part_ids"].shape[0])
        else:
            num_layers = int(f.attrs.get("core/number_of_layers", 0))
        layers = even_layer_indices(num_layers, sample_layers)
        print(f"  total_layers={num_layers}, sample_layers={layers}")

        # 1) metadata.json
        md = save_attrs_to_metadata(f, build_id="N1", num_layers=num_layers,
                                    extra={"source": "new_v1 (Peregrine v2023-09)",
                                           "sampled_layers": layers})
        (out_dir / "metadata.json").write_text(
            json.dumps(md, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        print(f"  [1/5] metadata.json")

        # 2) temporal
        n_indiv, n_comb = export_temporal(f, out_dir)
        print(f"  [2/5] temporal: {n_indiv} per-sensor CSV, {n_comb} cols in combined")
        temporal_keys = sorted(f["temporal"].keys()) if "temporal" in f else []

        # 3) cameras (NIR/0, NIR/1)
        n_cam = export_cameras(f, out_dir, layers, [
            ("slices/camera_data/NIR/0", "NIR_0"),
            ("slices/camera_data/NIR/1", "NIR_1"),
        ])
        print(f"  [3/5] cameras: {n_cam} frames")

        # 4) segmentation (categorical)
        n_seg = export_segmentation_v1(f, out_dir, layers)
        print(f"  [4/5] segmentation: {n_seg} layers")

        # 5) id_maps (part only)
        n_ids = export_id_maps(f, out_dir, layers, include_sample_ids=False)
        print(f"  [5/5] id_maps: {n_ids} layers")

        # SUMMARY
        write_summary_v1(
            out_dir,
            build_id="N1",
            build_name=hdf5_path.name,
            num_layers=num_layers,
            sampled_layers=layers,
            temporal_keys=temporal_keys,
            camera_layers_count=n_cam,
            seg_layers_count=n_seg,
            id_layers_count=n_ids,
        )

    print(f"new_v1 done in {time.time()-t0:.1f}s, total size = {folder_size_mb(out_dir):.2f} MB")


def export_new_v2(hdf5_path: Path, out_dir: Path, sample_layers: int) -> None:
    print("=" * 70)
    print(f"new_v2: exporting from {hdf5_path.name}")
    print(f"        → {out_dir}")
    print("=" * 70)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with h5py.File(hdf5_path, "r") as f:
        if "slices/part_ids" in f:
            num_layers = int(f["slices/part_ids"].shape[0])
        else:
            num_layers = int(f.attrs.get("core/number_of_layers", 0))
        layers = even_layer_indices(num_layers, sample_layers)
        print(f"  total_layers={num_layers}, sample_layers={layers}")

        # 1) metadata
        md = save_attrs_to_metadata(f, build_id="N2", num_layers=num_layers,
                                    extra={"source": "new_v2 (Peregrine v2023-10)",
                                           "sampled_layers": layers})
        (out_dir / "metadata.json").write_text(
            json.dumps(md, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        print(f"  [1/8] metadata.json")

        # 2) tensile (samples / parts)
        tstats = export_tensile_v2(f, out_dir)
        print(f"  [2/8] tensile/test results: {list(tstats.keys())}")

        # 3) process_parameters
        pstats = export_process_params_v2(f, out_dir)
        print(f"  [3/8] process_parameters: {pstats.get('process_parameters', {}).get('n_parts', 0)} parts")

        # 4) temporal
        n_indiv, n_comb = export_temporal(f, out_dir)
        print(f"  [4/8] temporal: {n_indiv} per-sensor CSV, {n_comb} cols in combined")
        temporal_keys = sorted(f["temporal"].keys()) if "temporal" in f else []

        # 5) cameras (visible/0, visible/1)
        n_cam = export_cameras(f, out_dir, layers, [
            ("slices/camera_data/visible/0", "post_melt"),
            ("slices/camera_data/visible/1", "post_powder"),
        ])
        print(f"  [5/8] cameras: {n_cam} frames")

        # 6) segmentation (12 bool)
        n_seg = export_segmentation_v2(f, out_dir, layers)
        print(f"  [6/8] segmentation: {n_seg} layers")

        # 7) id_maps (part + sample)
        n_ids = export_id_maps(f, out_dir, layers, include_sample_ids=True)
        print(f"  [7/8] id_maps: {n_ids} layers")

        # 8) scans
        n_scans = export_scans(f, out_dir, layers)
        print(f"  [8/8] scans: {n_scans} layers")

        write_summary_v2(
            out_dir,
            build_id="N2",
            build_name=hdf5_path.name,
            num_layers=num_layers,
            sampled_layers=layers,
            temporal_keys=temporal_keys,
            camera_layers_count=n_cam,
            seg_layers_count=n_seg,
            id_layers_count=n_ids,
            scan_layers_count=n_scans,
            tensile_stats=tstats,
            proc_stats=pstats,
        )

    print(f"new_v2 done in {time.time()-t0:.1f}s, total size = {folder_size_mb(out_dir):.2f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["new_v1", "new_v2", "both"], default="both")
    p.add_argument("--sample-layers", type=int, default=5,
                   help="Number of evenly-spaced layers to sample (default 5)")
    p.add_argument("--output-base", type=str, default=None,
                   help="Optional override base dir. If set, outputs go to "
                        "<base>/N1_raw and <base>/N2_raw.")
    args = p.parse_args(argv)

    if args.output_base:
        base = Path(args.output_base)
        n1_out = base / "N1_raw"
        n2_out = base / "N2_raw"
    else:
        n1_out = NEW_V1_OUT_DEFAULT
        n2_out = NEW_V2_OUT_DEFAULT

    if args.dataset in ("new_v1", "both"):
        if not NEW_V1_HDF5.exists():
            print(f"ERROR: new_v1 HDF5 not found: {NEW_V1_HDF5}", file=sys.stderr)
            return 2
        export_new_v1(NEW_V1_HDF5, n1_out, args.sample_layers)

    if args.dataset in ("new_v2", "both"):
        if not NEW_V2_HDF5.exists():
            print(f"ERROR: new_v2 HDF5 not found: {NEW_V2_HDF5}", file=sys.stderr)
            return 2
        export_new_v2(NEW_V2_HDF5, n2_out, args.sample_layers)

    print("\n=== ALL DONE ===")
    if args.dataset in ("new_v1", "both"):
        print(f"  new_v1 → {n1_out} ({folder_size_mb(n1_out):.2f} MB)")
    if args.dataset in ("new_v2", "both"):
        print(f"  new_v2 → {n2_out} ({folder_size_mb(n2_out):.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
