"""
Phase L1 — per-supervoxel 이미지 스택 추출 및 캐싱.

기존 `features.npz` 의 슈퍼복셀 순서(빌드별 concat)와 완전히 동일한
행 순서로 `{cache_dir}/stacks_all.h5` 를 생성한다.

디스크 절약 원칙: 기본 cache_dir 는 /tmp/image_stacks (tmpfs 가 권장).
호스트 디스크에 영속화하지 않음.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from ..common import config
from ..common.supervoxel import SuperVoxelGrid


# ------------------------------------------------------------
# Channel resolver
# ------------------------------------------------------------
def resolve_channels(name: str) -> List[str]:
    """채널 이름 → HDF5 키 리스트"""
    name = name.lower()
    cam0 = "slices/camera_data/visible/0"
    cam1 = "slices/camera_data/visible/1"
    dscnn = [f"slices/segmentation_results/{cid}"
             for cid, _ in config.DSCNN_FEATURE_MAP.values()]
    if name == "raw":
        return [cam0]
    if name == "raw_both":
        return [cam0, cam1]
    if name == "dscnn":
        return dscnn
    if name == "raw+dscnn":
        return [cam0] + dscnn
    raise ValueError(f"Unknown LSTM_INPUT_CHANNELS: {name}")


# ------------------------------------------------------------
# Per-build extraction
# ------------------------------------------------------------
def _read_zblock(f: h5py.File, key: str, l0: int, l1: int,
                 image_h: int, image_w: int) -> np.ndarray:
    """HDF5 의 z-block 을 float32 로 로드. 없으면 0."""
    if key not in f:
        return np.zeros((l1 - l0, image_h, image_w), dtype=np.float32)
    arr = f[key][l0:l1]
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype != np.float32:
        return arr.astype(np.float32)
    return arr


def extract_stacks_for_build(
    hdf5_path: str,
    voxel_indices: np.ndarray,
    grid: SuperVoxelGrid,
    channel_keys: Sequence[str],
    patch_px: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """단일 빌드의 슈퍼복셀 이미지 스택 추출.

    Args:
        voxel_indices: (N, 3)  (ix, iy, iz) — features.npz 와 동일 순서
        channel_keys:  len=C  HDF5 키 리스트
        patch_px:      Hs = Ws = patch_px

    Returns:
        stacks: (N, T, C, patch_px, patch_px) float16
        masks:  (N, T) bool
    """
    N = len(voxel_indices)
    T = config.SV_Z_LAYERS
    C = len(channel_keys)
    Hs = Ws = patch_px

    stacks = np.zeros((N, T, C, Hs, Ws), dtype=np.float16)
    masks = np.zeros((N, T), dtype=bool)

    # iz 별로 그룹핑 → z-block 을 딱 한 번 읽고 해당 z 의 모든 슈퍼복셀 crop
    by_iz: dict[int, list[tuple[int, int, int]]] = {}
    for row, (ix, iy, iz) in enumerate(voxel_indices):
        by_iz.setdefault(int(iz), []).append((row, int(ix), int(iy)))

    with h5py.File(hdf5_path, "r") as f:
        for iz in tqdm(sorted(by_iz.keys()), desc=Path(hdf5_path).stem[:20]):
            items = by_iz[iz]
            l0, l1 = grid.get_layer_range(iz)
            actual_t = l1 - l0

            # 채널 × z-block 을 로드 (C 개의 (actual_t, H, W))
            block = [
                _read_zblock(f, key, l0, l1, grid.image_h, grid.image_w)
                for key in channel_keys
            ]

            for row, ix, iy in items:
                r0, r1, c0, c1 = grid.get_pixel_range(ix, iy)
                ph = r1 - r0
                pw = c1 - c0
                # 센터 패딩: 실제 크기가 patch_px 보다 작거나 크면 맞춤
                copy_h = min(ph, Hs)
                copy_w = min(pw, Ws)
                for c_idx, arr in enumerate(block):
                    patch = arr[:, r0:r0 + copy_h, c0:c0 + copy_w]  # (actual_t, h, w)
                    stacks[row, :actual_t, c_idx, :copy_h, :copy_w] = patch.astype(np.float16)
                masks[row, :actual_t] = True

    return stacks, masks


# ------------------------------------------------------------
# Top-level cache builder
# ------------------------------------------------------------
def build_stacks_cache(
    builds: Sequence[str] | None = None,
    channels_name: str | None = None,
    patch_px: int | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """features.npz 와 정렬된 단일 stacks_all.h5 생성."""
    channels_name = channels_name or config.LSTM_INPUT_CHANNELS
    patch_px = patch_px or config.LSTM_PATCH_PX
    cache_dir = Path(cache_dir or config.LSTM_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    builds = list(builds) if builds else list(config.BUILDS.keys())
    channel_keys = resolve_channels(channels_name)
    C = len(channel_keys)
    T = config.SV_Z_LAYERS

    # per-build npz 로드 (voxel_indices 필수)
    per_build_voxels = []
    total_N = 0
    for bid in builds:
        npz_path = config.FEATURES_DIR / f"{bid}_features.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"{npz_path} 가 없습니다. Phase 1~2 (features 추출) 을 먼저 수행해야 합니다."
            )
        npz = np.load(npz_path)
        if "voxel_indices" not in npz.files:
            raise KeyError(
                f"{npz_path} 에 voxel_indices 키가 없습니다. features 재추출 필요."
            )
        voxels = npz["voxel_indices"]
        per_build_voxels.append((bid, voxels))
        total_N += len(voxels)

    out_path = cache_dir / "stacks_all.h5"
    est_gb = total_N * T * C * patch_px * patch_px * 2 / 1e9
    print(f"[L1] {out_path}")
    print(f"     N_total={total_N} T={T} C={C} HxW={patch_px}x{patch_px}")
    print(f"     estimated size: {est_gb:.2f} GB (float16)")

    with h5py.File(out_path, "w") as h5:
        stacks_ds = h5.create_dataset(
            "stacks",
            shape=(total_N, T, C, patch_px, patch_px),
            dtype="float16",
            chunks=(1, T, C, patch_px, patch_px),
        )
        masks_ds = h5.create_dataset("masks", shape=(total_N, T), dtype=bool)
        build_ids_ds = h5.create_dataset("build_ids", shape=(total_N,), dtype="int32")

        cursor = 0
        boundaries = [0]
        all_build_names = list(config.BUILDS.keys())
        for bid, voxels in per_build_voxels:
            print(f"\n[L1] {bid}: {len(voxels)} supervoxels")
            hdf5_path = str(config.hdf5_path(bid))
            grid = SuperVoxelGrid.from_hdf5(hdf5_path)
            stacks, masks = extract_stacks_for_build(
                hdf5_path, voxels, grid, channel_keys, patch_px
            )
            n = len(voxels)
            stacks_ds[cursor:cursor + n] = stacks
            masks_ds[cursor:cursor + n] = masks
            build_ids_ds[cursor:cursor + n] = all_build_names.index(bid)
            cursor += n
            boundaries.append(cursor)

            del stacks, masks  # 메모리 해제

        h5.create_dataset("build_boundaries", data=np.array(boundaries, dtype=np.int32))
        h5.attrs["channels"] = channels_name
        h5.attrs["patch_px"] = int(patch_px)
        h5.attrs["builds"] = ",".join(builds)
        h5.attrs["n_channels"] = int(C)
        h5.attrs["T"] = int(T)

    size_gb = out_path.stat().st_size / 1e9
    print(f"\n[L1] done: {out_path} ({size_gb:.2f} GB)")
    return out_path


def verify_alignment(cache_h5: Path | str | None = None) -> None:
    """stacks_all.h5 행 수가 all_features.npz 와 일치하는지 확인."""
    cache_h5 = Path(cache_h5 or Path(config.LSTM_CACHE_DIR) / "stacks_all.h5")
    npz = np.load(config.FEATURES_DIR / "all_features.npz")
    n_feat = len(npz["features"])
    with h5py.File(cache_h5, "r") as h5:
        n_stack = h5["stacks"].shape[0]
        bb = h5["build_boundaries"][...]
    assert n_stack == n_feat, f"행 수 불일치: stacks={n_stack} vs features={n_feat}"
    print(f"[L1] alignment OK: N={n_feat}, build_boundaries={bb.tolist()}")
