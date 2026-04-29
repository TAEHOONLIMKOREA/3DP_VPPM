"""
Phase L1 — SV별 가변 길이 크롭 시퀀스 캐시 빌드

각 유효 슈퍼복셀에 대해 다음을 저장:
  - stacks   (N_sv, 70, 8, 8) float16  — zero-padded 크롭 (uint8 → /255)
  - lengths  (N_sv,) int16             — 실제 시퀀스 길이 T_sv (1 ≤ T_sv ≤ 70)
  - sv_indices (N_sv, 3) int32         — (ix, iy, iz) — features.npz 와 1:1 매칭
  - sample_ids (N_sv,) int32

유효 레이어 정의:
  SV 의 z-범위 [l0, l1) 안에서 part_ids[L][SV xy] > 0 가 한 픽셀이라도 있는 레이어만 시퀀스에 포함.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from ..common import config
from ..common.supervoxel import SuperVoxelGrid, find_valid_supervoxels


def _build_one_build(build_id: str, out_dir: Path) -> Path:
    hdf5 = str(config.hdf5_path(build_id))
    out_path = out_dir / f"crop_stacks_{build_id}.h5"

    grid = SuperVoxelGrid.from_hdf5(hdf5)
    valid = find_valid_supervoxels(grid, hdf5)
    N = len(valid["voxel_indices"])
    if N == 0:
        print(f"  {build_id}: no valid supervoxels, skip")
        return out_path

    T_max = config.LSTM_T_MAX
    H = config.LSTM_CROP_H
    W = config.LSTM_CROP_W
    cam_key = f"slices/camera_data/visible/{config.LSTM_CAMERA_CHANNEL}"

    indices = valid["voxel_indices"]   # (N, 3) — (ix, iy, iz)

    # iz → 해당 z-block 의 SV idx 목록
    iz_to_svs: dict[int, list[int]] = {}
    for i, (_, _, iz) in enumerate(indices):
        iz_to_svs.setdefault(int(iz), []).append(i)

    print(f"  {build_id}: N={N} SVs, {len(iz_to_svs)} z-blocks")

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as out, h5py.File(hdf5, "r") as f:
        cam = f[cam_key]            # (num_layers, H_img, W_img) uint8
        part_ds = f["slices/part_ids"]

        stacks = out.create_dataset(
            "stacks", shape=(N, T_max, H, W), dtype="float16",
            compression="gzip", compression_opts=4,
            chunks=(1, T_max, H, W),
        )
        lengths = out.create_dataset("lengths", shape=(N,), dtype="int16")
        sv_indices = out.create_dataset(
            "sv_indices", data=indices.astype(np.int32),
        )
        sample_ids = out.create_dataset(
            "sample_ids", data=valid["sample_ids"].astype(np.int32),
        )

        for iz in tqdm(sorted(iz_to_svs.keys()), desc=f"{build_id} z-blocks"):
            l0, l1 = grid.get_layer_range(iz)
            block_cam = cam[l0:l1].astype(np.float16) / np.float16(255.0)   # (Tb, Himg, Wimg)
            block_part = part_ds[l0:l1]                                      # (Tb, Himg, Wimg)
            Tb = block_cam.shape[0]

            for sv_i in iz_to_svs[iz]:
                ix, iy, _ = indices[sv_i]
                r0, r1, c0, c1 = grid.get_pixel_range(int(ix), int(iy))
                cam_crop = block_cam[:, r0:r1, c0:c1]      # (Tb, h, w)
                part_crop = block_part[:, r0:r1, c0:c1]    # (Tb, h, w)

                # 유효 레이어: 이 SV xy 안에 part 가 한 픽셀이라도 있는 레이어
                valid_mask = (part_crop > 0).reshape(Tb, -1).any(axis=1)
                if not valid_mask.any():
                    # 안전장치 — 이론상 valid SV 면 발생 X
                    valid_mask[Tb // 2] = True

                seq = cam_crop[valid_mask]                  # (T_sv, h, w)
                T_sv = seq.shape[0]

                # 크롭이 8×8 보다 작은 경우 zero-pad (1842 % 8 = 2 → 모서리에서 발생 X 이지만 방어)
                if seq.shape[1] < H or seq.shape[2] < W:
                    padded = np.zeros((T_sv, H, W), dtype=np.float16)
                    padded[:, : seq.shape[1], : seq.shape[2]] = seq
                    seq = padded
                elif seq.shape[1] > H or seq.shape[2] > W:
                    seq = seq[:, :H, :W]   # 이론상 발생 X

                # T_sv 가 T_max 를 초과할 수 없지만 (Tb<=T_max), 안전 클립
                T_sv_clip = min(T_sv, T_max)
                stacks[sv_i, :T_sv_clip] = seq[:T_sv_clip]
                lengths[sv_i] = T_sv_clip

        out.attrs["T_max"] = T_max
        out.attrs["H"] = H
        out.attrs["W"] = W
        out.attrs["camera_channel"] = config.LSTM_CAMERA_CHANNEL
        out.attrs["build_id"] = build_id
        out.attrs["n_sv"] = N
        out.attrs["valid_layer_rule"] = "part_ids>0 in SV xy region"

    print(f"  saved {out_path}  ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path


def build_cache(build_ids: list[str] | None = None,
                out_dir: Path = config.LSTM_CACHE_DIR) -> list[Path]:
    """모든 빌드에 대해 크롭 시퀀스 캐시 빌드.

    이미 존재하는 파일은 건너뜀 (덮어쓰려면 삭제 후 재실행).
    """
    if build_ids is None:
        build_ids = list(config.BUILDS.keys())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for bid in build_ids:
        out_path = out_dir / f"crop_stacks_{bid}.h5"
        if out_path.exists():
            print(f"[cache] {out_path.name} already exists, skip")
            paths.append(out_path)
            continue
        print(f"\n[cache] Building {bid} → {out_path}")
        paths.append(_build_one_build(bid, out_dir))

    return paths


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--builds", nargs="+", default=list(config.BUILDS.keys()))
    p.add_argument("--out-dir", type=Path, default=config.LSTM_CACHE_DIR)
    args = p.parse_args()
    build_cache(args.builds, args.out_dir)
