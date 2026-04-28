"""Phase L1 — HDF5 → 샘플별 시퀀스 캐시.

각 샘플 s 에 대해:
  - sample_ids[layer] == s 인 픽셀의 bbox 추출
  - raw camera (channel 0) bbox crop → 64×64 resize
  - 등장 모든 layer 에 대해 같은 작업 → (T_s, 64, 64) float16 시퀀스
  - per-build min/max 로 [0, 1] 정규화

산출물:
  Sources/pipeline_outputs/sample_stacks/{build_id}.h5
  Sources/pipeline_outputs/sample_stacks/normalization.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..common import config


# ── HDF5 helpers ──────────────────────────────────────────


def find_sample_layer_ranges(sample_ids: h5py.Dataset, sample_id: int) -> tuple[int, int]:
    """sample_id 가 등장하는 layer 범위 (l0, l1) 를 찾는다 (l1 exclusive).

    HDF5 sample_ids 는 너무 커서 in-memory 로딩 어려움. 레이어 단위 로딩으로 검색.
    """
    n_layers = sample_ids.shape[0]
    l0, l1 = -1, -1
    for L in range(n_layers):
        layer = sample_ids[L]
        if (layer == sample_id).any():
            if l0 == -1:
                l0 = L
            l1 = L + 1
    return l0, l1


def find_all_sample_layer_ranges(hdf5_path: Path) -> dict[int, tuple[int, int]]:
    """전체 레이어를 한 번 훑어 모든 sample id 의 (l0, l1) 를 집계.

    return: {sample_id: (l0, l1)}
    """
    ranges: dict[int, list[int]] = {}
    with h5py.File(hdf5_path, "r") as f:
        sids = f["slices/sample_ids"]
        n_layers = sids.shape[0]
        for L in tqdm(range(n_layers), desc="scan layers", leave=False):
            layer = sids[L]
            ids = np.unique(layer)
            for sid in ids:
                if sid == 0:
                    continue
                sid = int(sid)
                if sid not in ranges:
                    ranges[sid] = [L, L + 1]
                else:
                    ranges[sid][1] = L + 1
    return {sid: (lo, hi) for sid, [lo, hi] in ranges.items()}


def crop_resize(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """mask 의 bbox 로 image 를 crop 후 size×size 로 resize.

    Args:
        image: (H, W) float32
        mask:  (H, W) bool
        size:  출력 정사각 크기

    Returns:
        (resized: (size, size) float32, (r0, r1, c0, c1)) — mask 가 비어 있으면 None
    """
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    if not rows.any() or not cols.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r1 += 1; c1 += 1
    crop = image[r0:r1, c0:c1]
    # 마스크 밖 픽셀은 0 으로 (해당 sample 영역 외 정보 차단)
    sub_mask = mask[r0:r1, c0:c1]
    crop = np.where(sub_mask, crop, 0.0).astype(np.float32)
    pil = Image.fromarray(crop, mode="F").resize((size, size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32), (int(r0), int(r1), int(c0), int(c1))


# ── 캐시 빌드 ──────────────────────────────────────────────


def build_cache_for_build(
    build_id: str,
    out_dir: Path,
    crop_size: int = config.LSTM_CROP_SIZE,
    raw_channel: int = config.LSTM_RAW_CHANNEL,
) -> dict:
    """한 빌드의 모든 샘플 시퀀스 캐시 생성.

    Returns:
        {n_samples, total_frames, raw_min, raw_max}
    """
    hdf5_path = config.hdf5_path(build_id)
    out_path = out_dir / f"{build_id}.h5"
    print(f"[L1] {build_id}: scanning sample layer ranges...")
    t0 = time.time()
    ranges = find_all_sample_layer_ranges(hdf5_path)
    print(f"[L1] {build_id}: {len(ranges)} samples, scan took {time.time()-t0:.1f}s")

    sids_sorted = sorted(ranges.keys())
    n_samples = len(sids_sorted)

    # 1pass: 전체 frame 수 계산 (l1 - l0 의 합 — 실제 등장 layer 수는 같거나 작음)
    total_max = sum(ranges[s][1] - ranges[s][0] for s in sids_sorted)

    print(f"[L1] {build_id}: encoding sequences (max {total_max} frames)...")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_min, raw_max = +np.inf, -np.inf

    with h5py.File(hdf5_path, "r") as fin, h5py.File(out_path, "w") as fout:
        cam = fin[f"slices/camera_data/visible/{raw_channel}"]
        sids = fin["slices/sample_ids"]

        # 동적 크기로 채우기 위해 list 누적 → 마지막에 dataset 작성
        seq_lengths = np.zeros(n_samples, dtype=np.int32)
        seq_chunks: list[np.ndarray] = []
        bbox_chunks: list[np.ndarray] = []
        layer_id_chunks: list[np.ndarray] = []

        for i, sid in enumerate(tqdm(sids_sorted, desc=f"{build_id} samples")):
            l0, l1 = ranges[sid]
            seq_frames: list[np.ndarray] = []
            seq_bboxes: list[tuple[int, int, int, int]] = []
            seq_layer_ids: list[int] = []
            for L in range(l0, l1):
                mask = (sids[L] == sid)
                if not mask.any():
                    continue
                img = cam[L]
                got = crop_resize(img, mask, crop_size)
                if got is None:
                    continue
                resized, bbox = got
                seq_frames.append(resized)
                seq_bboxes.append(bbox)
                seq_layer_ids.append(L)
                # min/max 갱신 (마스크 영역 안 픽셀만)
                m = (resized > 0)
                if m.any():
                    raw_min = min(raw_min, float(resized[m].min()))
                    raw_max = max(raw_max, float(resized[m].max()))

            T_s = len(seq_frames)
            if T_s == 0:
                continue
            seq_arr = np.stack(seq_frames, axis=0)            # (T_s, H, W) float32
            bb_arr = np.asarray(seq_bboxes, dtype=np.int32)
            li_arr = np.asarray(seq_layer_ids, dtype=np.int32)
            seq_lengths[i] = T_s
            seq_chunks.append(seq_arr)
            bbox_chunks.append(bb_arr)
            layer_id_chunks.append(li_arr)

        # 평탄화 + dataset 저장 (정규화는 train 에서 별도 수행)
        sequences_flat = np.concatenate(seq_chunks, axis=0)   # (sum T_s, H, W) float32
        bboxes_flat = np.concatenate(bbox_chunks, axis=0)
        layer_ids_flat = np.concatenate(layer_id_chunks, axis=0)
        offsets = np.zeros(n_samples + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(seq_lengths.astype(np.int64))

        # float16 저장으로 디스크 절약
        fout.create_dataset("sample_ids", data=np.asarray(sids_sorted, dtype=np.uint32))
        fout.create_dataset("seq_lengths", data=seq_lengths)
        fout.create_dataset("layer_offsets", data=offsets)
        fout.create_dataset("sequences", data=sequences_flat.astype(np.float16),
                            compression="gzip", compression_opts=4, chunks=True)
        fout.create_dataset("bboxes", data=bboxes_flat)
        fout.create_dataset("layer_ids", data=layer_ids_flat)
        fout.attrs["build_id"] = build_id
        fout.attrs["crop_size"] = crop_size
        fout.attrs["raw_channel"] = raw_channel
        fout.attrs["raw_min_local"] = float(raw_min)
        fout.attrs["raw_max_local"] = float(raw_max)

    print(f"[L1] {build_id}: ✓ {n_samples} samples, {sequences_flat.shape[0]} frames "
          f"→ {out_path}, raw_range=[{raw_min:.3f}, {raw_max:.3f}]")
    return {
        "build_id": build_id,
        "n_samples": int(n_samples),
        "total_frames": int(sequences_flat.shape[0]),
        "raw_min": float(raw_min),
        "raw_max": float(raw_max),
    }


# ── CLI ──────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase L1 — Sample sequence cache builder")
    parser.add_argument("--builds", nargs="+", default=list(config.BUILDS.keys()),
                        help="Build IDs (default: all 5)")
    parser.add_argument("--out-dir", default=config.LSTM_CACHE_DIR)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for bid in args.builds:
        s = build_cache_for_build(bid, out_dir)
        summaries.append(s)

    # 글로벌 정규화 통계 (모든 빌드 공통)
    g_min = min(s["raw_min"] for s in summaries)
    g_max = max(s["raw_max"] for s in summaries)
    norm = {
        "method": "minmax_global",
        "raw_min": g_min,
        "raw_max": g_max,
        "per_build": summaries,
    }
    with open(out_dir / "normalization.json", "w") as f:
        json.dump(norm, f, indent=2)
    print(f"[L1] ✓ {sum(s['n_samples'] for s in summaries)} samples cached "
          f"→ {out_dir}, global raw=[{g_min:.3f}, {g_max:.3f}]")


if __name__ == "__main__":
    main()
