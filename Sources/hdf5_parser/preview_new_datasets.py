"""
new_v1 / new_v2 데이터셋 프리뷰 시각화 생성 스크립트
====================================================

목적: 학습용 추출이 아니라 "이 데이터셋이 어떻게 생겼는지" 빠르게 감 잡기.

출력:
  Sources/pipeline_outputs/figures/dataset_preview/
    - new_v1_overview.png
    - new_v2_overview.png
    - new_v2_tensile_hist.png
    - new_v2_parameter_set_distribution.png

특징:
  - new_v1 (Arcam EB-PBF, IN738, NIR): visible 카메라 없음, segmentation 은 단일 categorical 3D
  - new_v2 (Concept Laser M2, SS 316L): baseline 과 동일 in-situ 포맷 (visible/0,1 + 12 bool seg)

메모리 안전:
  - 카메라/세그 이미지는 한 레이어만 read
  - 시계열은 1D (L,) 만 read
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")  # GUI 없이 저장만
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.collections import LineCollection


# -----------------------------------------------------------------------------
# 경로 설정
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

NEW_V1_PATH = (
    PROJECT_ROOT
    / "ORNL_Data"
    / "Co-Registered In-Situ and Ex-Situ Dataset"
    / "[new_v1] (Peregrine v2023-09)"
    / "2018-12-31 037-R1256_BLADE ITERATION 13_5.hdf5"
)

NEW_V2_PATH = (
    PROJECT_ROOT
    / "ORNL_Data"
    / "Co-Registered In-Situ and Ex-Situ Dataset"
    / "[new_v2] (Peregrine v2023-10)"
    / "2023-03-15 AMMTO Spatial Variation Baseline.hdf5"
)

OUTPUT_DIR = PROJECT_ROOT / "Sources" / "pipeline_outputs" / "figures" / "dataset_preview"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 120


# -----------------------------------------------------------------------------
# 공용 유틸
# -----------------------------------------------------------------------------

def percentile_clip(img: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> Tuple[float, float]:
    """이미지의 percentile clip 범위를 반환. 실패 시 raw min/max."""
    finite = np.isfinite(img)
    if not finite.any():
        return 0.0, 1.0
    try:
        v = img[finite]
        vmin = float(np.percentile(v, lo))
        vmax = float(np.percentile(v, hi))
        if vmax <= vmin:
            return float(v.min()), float(v.max())
        return vmin, vmax
    except Exception:
        v = img[finite]
        return float(v.min()), float(v.max())


def imshow_with_title(ax, arr, title, cmap="gray", vmin=None, vmax=None, colorbar=True):
    if vmin is None or vmax is None:
        vmin, vmax = percentile_clip(arr, 1.0, 99.0)
    im = ax.imshow(arr, cmap=cmap, interpolation="none", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


# -----------------------------------------------------------------------------
# new_v1 overview
# -----------------------------------------------------------------------------

def make_new_v1_overview() -> Path:
    """new_v1 (NIR-only EB-PBF) 멀티패널 프리뷰."""
    out_path = OUTPUT_DIR / "new_v1_overview.png"
    print(f"[new_v1] reading {NEW_V1_PATH.name}")

    with h5py.File(NEW_V1_PATH, "r") as f:
        nir0 = f["slices/camera_data/NIR/0"]
        nir1 = f["slices/camera_data/NIR/1"]
        seg = f["slices/segmentation_results"]  # single categorical Dataset
        part_ids = f["slices/part_ids"]

        L = nir0.shape[0]
        layer = L // 2  # 중간 레이어
        H, W = nir0.shape[1], nir0.shape[2]
        print(f"[new_v1]   total layers={L}, picking layer={layer}, grid={H}x{W}")

        # 레이어 한 장만 read
        img0 = nir0[layer, ...]
        img1 = nir1[layer, ...]
        seg_slice = seg[layer, ...]
        pids = part_ids[layer, ...]

        # 센서 시계열 (전체 build) - 1D 만
        beam_current = f["temporal/beam_current"][...]
        chamber_vac = f["temporal/chamber_vacuum_gauge_fb"][...]

        nir0_shape = nir0.shape
        nir1_shape = nir1.shape
        seg_shape = seg.shape
        part_shape = part_ids.shape

    # ---- figure: 2x3 grid ----
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    # (0,0) NIR/0
    vmin, vmax = percentile_clip(img0, 1, 99)
    imshow_with_title(
        axes[0, 0], img0,
        f"slices/camera_data/NIR/0\nshape={nir0_shape}, layer={layer} (clip 1-99%: {vmin:.1f}-{vmax:.1f})",
        cmap="inferno", vmin=vmin, vmax=vmax,
    )

    # (0,1) NIR/1
    vmin, vmax = percentile_clip(img1, 1, 99)
    imshow_with_title(
        axes[0, 1], img1,
        f"slices/camera_data/NIR/1\nshape={nir1_shape}, layer={layer} (clip 1-99%: {vmin:.1f}-{vmax:.1f})",
        cmap="inferno", vmin=vmin, vmax=vmax,
    )

    # (0,2) segmentation - categorical with discovered class IDs
    unique_ids = sorted(int(c) for c in np.unique(seg_slice).tolist())
    n_classes = len(unique_ids)
    # tab20 컬러맵에서 클래스 ID 별 색 추출 (categorical)
    base_cmap = plt.get_cmap("tab20")
    colors = [base_cmap(i % 20) for i in range(max(unique_ids) + 1)]
    seg_cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, max(unique_ids) + 1.5, 1.0)
    seg_norm = mcolors.BoundaryNorm(bounds, seg_cmap.N)

    ax = axes[0, 2]
    im = ax.imshow(seg_slice, cmap=seg_cmap, norm=seg_norm, interpolation="none")
    ax.set_title(
        f"slices/segmentation_results\nshape={seg_shape}, layer={layer}, observed IDs={unique_ids}",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(
        im, ax=ax, fraction=0.046, pad=0.04,
        ticks=unique_ids,
    )
    cbar.ax.set_yticklabels([f"ID {c}" for c in unique_ids])

    # (1,0) part_ids
    pids_unique = np.unique(pids)
    n_parts = len(pids_unique)
    # 0=배경 → 흰색, 나머지 categorical
    ax = axes[1, 0]
    # tab20 으로 categorical
    pid_max = int(pids_unique.max())
    pid_colors = ["white"] + [plt.get_cmap("tab20")(i % 20) for i in range(pid_max)]
    pid_cmap = mcolors.ListedColormap(pid_colors)
    pid_bounds = np.arange(-0.5, pid_max + 1.5, 1.0)
    pid_norm = mcolors.BoundaryNorm(pid_bounds, pid_cmap.N)
    im = ax.imshow(pids, cmap=pid_cmap, norm=pid_norm, interpolation="none")
    ax.set_title(
        f"slices/part_ids\nshape={part_shape}, layer={layer}, unique IDs={n_parts} (incl. 0=bg)",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (1,1) beam_current 시계열
    ax = axes[1, 1]
    ax.plot(np.arange(len(beam_current)), beam_current, lw=0.8, color="tab:red")
    ax.set_title(
        f"temporal/beam_current\nshape={beam_current.shape}, dtype=float32",
        fontsize=10,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("beam_current")
    ax.grid(alpha=0.3)

    # (1,2) chamber_vacuum_gauge_fb 시계열
    ax = axes[1, 2]
    ax.plot(np.arange(len(chamber_vac)), chamber_vac, lw=0.8, color="tab:blue")
    ax.set_title(
        f"temporal/chamber_vacuum_gauge_fb\nshape={chamber_vac.shape}, dtype=float32",
        fontsize=10,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("chamber_vacuum_gauge_fb")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"new_v1 (Arcam EB-PBF, IN738, NIR)  |  {NEW_V1_PATH.name}",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[new_v1] saved {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# new_v2 overview
# -----------------------------------------------------------------------------

DSCNN_CLASSES = {
    0: "Powder", 1: "Printed", 2: "Recoater Hopping", 3: "Recoater Streaking",
    4: "Incomplete Spreading", 5: "Swelling", 6: "Debris", 7: "Super-Elevation",
    8: "Spatter", 9: "Misprint", 10: "Over Melting", 11: "Under Melting",
}

BASELINE_SENSOR_KEYS = [
    "layer_times",
    "top_flow_rate",
    "bottom_flow_rate",
    "module_oxygen",
    "build_plate_temperature",
    "bottom_flow_temperature",
    "actual_ventilator_flow_rate",
]


def make_new_v2_overview() -> Path:
    """new_v2 (Concept Laser M2 baseline-format) 멀티패널 프리뷰."""
    out_path = OUTPUT_DIR / "new_v2_overview.png"
    print(f"[new_v2] reading {NEW_V2_PATH.name}")

    with h5py.File(NEW_V2_PATH, "r") as f:
        vis0 = f["slices/camera_data/visible/0"]
        vis1 = f["slices/camera_data/visible/1"]
        part_ids = f["slices/part_ids"]
        sample_ids = f["slices/sample_ids"]

        L = vis0.shape[0]
        layer = 900 if L > 900 else L // 2
        H, W = vis0.shape[1], vis0.shape[2]
        print(f"[new_v2]   total layers={L}, picking layer={layer}, grid={H}x{W}")

        img0 = vis0[layer, ...]
        img1 = vis1[layer, ...]
        pids = part_ids[layer, ...]
        sids = sample_ids[layer, ...]

        # 12 클래스 segmentation: 한 레이어씩
        seg_layer = {}
        for c in range(12):
            key = f"slices/segmentation_results/{c}"
            if key in f:
                seg_layer[c] = f[key][layer, ...]

        # scan path (이 레이어)
        scan_key = f"scans/{layer}"
        if scan_key in f:
            scan_data = f[scan_key][...]
        else:
            scan_data = None

        # 7 baseline sensor 시계열
        sensors = {}
        for k in BASELINE_SENSOR_KEYS:
            ds_key = f"temporal/{k}"
            if ds_key in f:
                sensors[k] = f[ds_key][...]

        vis0_shape = vis0.shape
        vis1_shape = vis1.shape

    # ---- figure: 큰 grid ----
    # row 1: vis0, vis1, scan_path, part_ids, sample_ids  (5 cols, but use 4 cols and put scan + sample in row 2)
    # 단순히 row 1: vis0/vis1/part_ids/sample_ids (4) + scan + 12 seg grid (4x3) + 7 sensors
    # 깔끔하게: figure 를 두 영역으로 나눔 - subfigures 사용
    fig = plt.figure(figsize=(24, 22))

    gs_root = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.55], hspace=0.32)

    # === Row 0: vis0, vis1, part_ids, sample_ids, scan_path ===
    gs0 = gs_root[0].subgridspec(1, 5, wspace=0.25)

    ax = fig.add_subplot(gs0[0, 0])
    vmin, vmax = percentile_clip(img0, 1, 99)
    imshow_with_title(
        ax, img0,
        f"slices/camera_data/visible/0\nshape={vis0_shape}, layer={layer}\nclip 1-99%: {vmin:.2f}-{vmax:.2f}",
        cmap="gray", vmin=vmin, vmax=vmax,
    )

    ax = fig.add_subplot(gs0[0, 1])
    vmin, vmax = percentile_clip(img1, 1, 99)
    imshow_with_title(
        ax, img1,
        f"slices/camera_data/visible/1\nshape={vis1_shape}, layer={layer}\nclip 1-99%: {vmin:.2f}-{vmax:.2f}",
        cmap="gray", vmin=vmin, vmax=vmax,
    )

    # part_ids
    ax = fig.add_subplot(gs0[0, 2])
    pids_unique = np.unique(pids)
    pid_max = int(pids_unique.max())
    pid_colors = ["white"] + [plt.get_cmap("tab20")(i % 20) for i in range(max(1, pid_max))]
    pid_cmap = mcolors.ListedColormap(pid_colors)
    pid_bounds = np.arange(-0.5, pid_max + 1.5, 1.0)
    pid_norm = mcolors.BoundaryNorm(pid_bounds, pid_cmap.N)
    im = ax.imshow(pids, cmap=pid_cmap, norm=pid_norm, interpolation="none")
    ax.set_title(
        f"slices/part_ids\nshape={part_ids.shape if False else pids.shape}, layer={layer}\nunique IDs={len(pids_unique)} (incl. 0=bg)",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # sample_ids
    ax = fig.add_subplot(gs0[0, 3])
    sids_unique = np.unique(sids)
    sid_max = int(sids_unique.max())
    sid_colors = ["white"] + [plt.get_cmap("tab20b")(i % 20) for i in range(max(1, sid_max))]
    sid_cmap = mcolors.ListedColormap(sid_colors)
    sid_bounds = np.arange(-0.5, sid_max + 1.5, 1.0)
    sid_norm = mcolors.BoundaryNorm(sid_bounds, sid_cmap.N)
    im = ax.imshow(sids, cmap=sid_cmap, norm=sid_norm, interpolation="none")
    ax.set_title(
        f"slices/sample_ids\nshape={sids.shape}, layer={layer}\nunique IDs={len(sids_unique)} (incl. 0=bg)",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # scan path
    ax = fig.add_subplot(gs0[0, 4])
    if scan_data is not None and scan_data.shape[0] > 0:
        x = scan_data[:, 0:2]
        y = scan_data[:, 2:4]
        t = scan_data[:, 4]
        segments = np.stack([x, y], axis=-1)  # (N, 2, 2): each (start,end) point with (x,y)
        # 위 axis 정렬: x col 2개 + y col 2개 → reshape to (N, 2, 2)
        # 사실 Line은 [(x0,y0),(x1,y1)] 페어가 필요
        seg_pts = np.stack(
            [
                np.stack([scan_data[:, 0], scan_data[:, 2]], axis=-1),
                np.stack([scan_data[:, 1], scan_data[:, 3]], axis=-1),
            ],
            axis=1,
        )  # (N, 2, 2)
        norm = mcolors.Normalize(vmin=t.min(), vmax=t.max())
        lc = LineCollection(seg_pts, cmap="viridis", norm=norm, linewidths=0.6)
        lc.set_array(t)
        ax.add_collection(lc)
        ax.set_xlim(seg_pts[..., 0].min(), seg_pts[..., 0].max())
        ax.set_ylim(seg_pts[..., 1].min(), seg_pts[..., 1].max())
        ax.set_aspect("equal")
        ax.set_title(
            f"scans/{layer}\nshape={scan_data.shape}, dtype=float32\ncolor=relative_time",
            fontsize=10,
        )
        plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04, label="rel. time")
    else:
        ax.text(0.5, 0.5, f"No scan data for layer {layer}", ha="center", va="center")
        ax.set_title("scans/N/A", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # === Row 1: 12 DSCNN classes (4x3) ===
    gs1 = gs_root[1].subgridspec(3, 4, wspace=0.15, hspace=0.35)
    seg_shape_str = f"(L, {H}, {W}) bool x12"
    for idx in range(12):
        r, c = divmod(idx, 4)
        ax = fig.add_subplot(gs1[r, c])
        if idx in seg_layer:
            seg_img = seg_layer[idx]
            n_pos = int(seg_img.sum())
            ax.imshow(seg_img, cmap="binary_r", interpolation="none", vmin=0, vmax=1)
            ax.set_title(
                f"seg/{idx} ({DSCNN_CLASSES[idx]})  n_pos={n_pos}",
                fontsize=9,
            )
        else:
            ax.text(0.5, 0.5, f"missing seg/{idx}", ha="center", va="center")
            ax.set_title(f"seg/{idx}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.text(
        0.5, 0.66,
        f"slices/segmentation_results/[0..11]   shape per class={seg_shape_str},   layer={layer}",
        ha="center", fontsize=11, style="italic",
    )

    # === Row 2: 7 baseline sensors (2x4 grid, last cell empty) ===
    gs2 = gs_root[2].subgridspec(2, 4, wspace=0.3, hspace=0.55)
    for i, key in enumerate(BASELINE_SENSOR_KEYS):
        r, c = divmod(i, 4)
        ax = fig.add_subplot(gs2[r, c])
        if key in sensors:
            arr = sensors[key]
            ax.plot(np.arange(len(arr)), arr, lw=0.7, color=f"C{i}")
            ax.set_title(f"temporal/{key}\nshape={arr.shape}", fontsize=9)
            ax.set_xlabel("Layer")
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"missing temporal/{key}", ha="center", va="center")
            ax.set_title(f"temporal/{key}", fontsize=9)
    # 빈 칸
    ax = fig.add_subplot(gs2[1, 3])
    ax.axis("off")
    ax.text(
        0.05, 0.5,
        "Sensor channels matched to baseline (B1.*) format:\n"
        "all 7 keys present in new_v2.\n\n"
        "Plus 12 extra temporal channels (build_time,\n"
        "ventilator_speed, gas_loop_oxygen, etc.).",
        fontsize=9, va="center",
    )

    fig.suptitle(
        f"new_v2 (Concept Laser M2, SS 316L; baseline-format in-situ)  |  {NEW_V2_PATH.name}",
        fontsize=14, y=0.995,
    )
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[new_v2] saved {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# new_v2 bonus: tensile histograms
# -----------------------------------------------------------------------------

def make_new_v2_tensile_hist() -> Path:
    out_path = OUTPUT_DIR / "new_v2_tensile_hist.png"
    print("[new_v2 bonus] tensile histograms")

    with h5py.File(NEW_V2_PATH, "r") as f:
        keys = ["yield_strength", "ultimate_tensile_strength", "total_elongation"]
        data = {}
        for k in keys:
            full = f"parts/test_results/{k}"
            if full in f:
                data[k] = f[full][...]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    units = {
        "yield_strength": "MPa",
        "ultimate_tensile_strength": "MPa",
        "total_elongation": "%",
    }
    for ax, key in zip(axes, keys):
        if key not in data:
            ax.text(0.5, 0.5, f"missing {key}", ha="center", va="center")
            ax.set_title(key, fontsize=11)
            continue
        arr = data[key]
        valid_mask = np.isfinite(arr) & (arr != 0.0)
        valid = arr[valid_mask]
        n_total = arr.size
        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            ax.text(0.5, 0.5, "no valid values", ha="center", va="center")
            ax.set_title(f"parts/test_results/{key}", fontsize=11)
            continue
        ax.hist(valid, bins=20, edgecolor="black", alpha=0.75, color="tab:blue")
        mean_v = float(np.mean(valid))
        ax.axvline(mean_v, color="red", linestyle="--", lw=1.5,
                   label=f"mean={mean_v:.1f}")
        ax.set_title(
            f"parts/test_results/{key}\nshape={arr.shape}, "
            f"N(valid)={n_valid}/{n_total}",
            fontsize=10,
        )
        ax.set_xlabel(f"{key} [{units.get(key, 'a.u.')}]")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(
        f"new_v2 tensile properties (parts-level)  |  {NEW_V2_PATH.name}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[new_v2 bonus] saved {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# new_v2 bonus: parameter_set distribution
# -----------------------------------------------------------------------------

def make_new_v2_parameter_set_distribution() -> Path:
    out_path = OUTPUT_DIR / "new_v2_parameter_set_distribution.png"
    print("[new_v2 bonus] parameter_set distribution")

    with h5py.File(NEW_V2_PATH, "r") as f:
        key = "parts/process_parameters/parameter_set"
        if key not in f:
            raise KeyError(key)
        arr = f[key][...]

    # arr 가 string 인지 int 인지 확인
    print(f"   parameter_set dtype={arr.dtype}, shape={arr.shape}")
    if arr.dtype.kind in ("S", "O", "U"):
        # bytes → str
        labels = [
            (v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v))
            for v in arr
        ]
    else:
        labels = [str(int(v)) for v in arr]

    # 빈도
    from collections import Counter
    cnt = Counter(labels)
    items = sorted(cnt.items(), key=lambda x: -x[1])
    keys_sorted = [k for k, _ in items]
    counts = [v for _, v in items]
    n_unique = len(keys_sorted)

    # 너무 많으면 (55+) bar 가로
    fig, ax = plt.subplots(figsize=(14, max(6, 0.18 * n_unique + 2)))
    y_pos = np.arange(n_unique)
    ax.barh(y_pos, counts, color="tab:purple", alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(keys_sorted, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("part count")
    ax.set_title(
        f"parts/process_parameters/parameter_set\n"
        f"shape={arr.shape}, dtype={arr.dtype}, unique values = {n_unique}",
        fontsize=11,
    )
    ax.grid(alpha=0.3, axis="x")
    for i, c in enumerate(counts):
        ax.text(c + 0.1, i, f" {c}", va="center", fontsize=7)

    fig.suptitle(
        f"new_v2 process parameter sweep distribution  |  {NEW_V2_PATH.name}",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[new_v2 bonus] saved {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> List[Path]:
    outputs = []
    outputs.append(make_new_v1_overview())
    outputs.append(make_new_v2_overview())
    outputs.append(make_new_v2_tensile_hist())
    outputs.append(make_new_v2_parameter_set_distribution())

    print("\n=== Generated previews ===")
    for p in outputs:
        print(f"  {p}")
    return outputs


if __name__ == "__main__":
    main()
