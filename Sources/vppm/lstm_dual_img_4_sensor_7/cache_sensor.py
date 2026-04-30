"""Phase S1 — Sensor 시퀀스 캐시 빌드 (lstm_dual_img_4_sensor_7).

각 valid SV 에 대해 7-channel sensor 시계열을 추출:
  - 시퀀스 길이 = 카메라 v0 캐시의 lengths 와 동일 (valid_mask = part_ids > 0 in SV xy)
  - 값은 raw 보존 — 정규화는 dataset.py 의 build_normalized_dataset 에서 수행

산출물: experiments/vppm_lstm_dual_img_4_sensor_7/cache/sensor_stacks_{B1.x}.h5
  - sensors    (N, T_max=70, 7) float32, zero-padded
  - lengths    (N,) int16
  - sv_indices (N, 3) int32
  - sample_ids (N,) int32

이미 존재하는 파일은 건너뜀 (덮어쓰려면 삭제 후 재실행).
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
    out_path = out_dir / f"sensor_stacks_{build_id}.h5"

    grid = SuperVoxelGrid.from_hdf5(hdf5)
    valid = find_valid_supervoxels(grid, hdf5)
    indices = valid["voxel_indices"]              # (N, 3)
    N = len(indices)
    if N == 0:
        print(f"  {build_id}: no valid SVs, skip")
        return out_path

    T_max = config.LSTM_T_MAX
    n_ch = config.LSTM_DUAL_IMG_4_SENSOR_7_N_CHANNELS

    iz_to_svs: dict[int, list[int]] = {}
    for i, (_, _, iz) in enumerate(indices):
        iz_to_svs.setdefault(int(iz), []).append(i)

    print(f"  {build_id}: N={N} SVs, {len(iz_to_svs)} z-blocks")

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as out, h5py.File(hdf5, "r") as f:
        # temporal/* 일괄 로드 (각 1D 시계열, 무시할 만한 메모리)
        temporal_arrays = []
        for key in config.TEMPORAL_FEATURES:
            full_key = f"temporal/{key}"
            if full_key not in f:
                raise KeyError(f"{build_id}: HDF5 에 {full_key} 없음")
            temporal_arrays.append(f[full_key][...].astype(np.float32))
        # (n_ch, num_layers) — 각 채널 길이가 같다고 가정
        num_layers_per_ch = [len(t) for t in temporal_arrays]
        if len(set(num_layers_per_ch)) != 1:
            raise RuntimeError(
                f"{build_id}: temporal 채널별 길이 불일치 {dict(zip(config.TEMPORAL_FEATURES, num_layers_per_ch))}"
            )
        temporals = np.stack(temporal_arrays, axis=0)        # (7, num_layers)

        part_ds = f["slices/part_ids"]

        sensors = out.create_dataset(
            "sensors", shape=(N, T_max, n_ch), dtype="float32",
            compression="gzip", compression_opts=4,
            chunks=(1, T_max, n_ch),
        )
        lengths = out.create_dataset("lengths", shape=(N,), dtype="int16")
        out.create_dataset("sv_indices", data=indices.astype(np.int32))
        out.create_dataset("sample_ids", data=valid["sample_ids"].astype(np.int32))

        for iz in tqdm(sorted(iz_to_svs.keys()), desc=f"{build_id} z-blocks"):
            l0, l1 = grid.get_layer_range(iz)
            block_part = part_ds[l0:l1]                       # (Tb, H_img, W_img)
            block_temp = temporals[:, l0:l1].T                # (Tb, 7)
            Tb = block_part.shape[0]

            for sv_i in iz_to_svs[iz]:
                ix, iy, _ = indices[sv_i]
                r0, r1, c0, c1 = grid.get_pixel_range(int(ix), int(iy))
                part_crop = block_part[:, r0:r1, c0:c1]       # (Tb, h, w)

                # lstm/crop_stacks.py 와 동일한 valid_mask 정의
                valid_mask = (part_crop > 0).reshape(Tb, -1).any(axis=1)
                if not valid_mask.any():
                    valid_mask[Tb // 2] = True               # 안전장치

                seq = block_temp[valid_mask]                 # (T_sv, 7)
                T_sv = seq.shape[0]
                T_sv_clip = min(T_sv, T_max)

                sensors[sv_i, :T_sv_clip] = seq[:T_sv_clip]
                lengths[sv_i] = T_sv_clip

        out.attrs["T_max"] = T_max
        out.attrs["n_channels"] = n_ch
        out.attrs["channel_names"] = np.array(config.TEMPORAL_FEATURES, dtype="S")
        out.attrs["build_id"] = build_id
        out.attrs["n_sv"] = N
        out.attrs["valid_layer_rule"] = "part_ids>0 in SV xy region (matches lstm/crop_stacks.py)"

    print(f"  saved {out_path}  ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path


def build_cache(build_ids: list[str] | None = None,
                out_dir: Path = config.LSTM_DUAL_IMG_4_SENSOR_7_CACHE_DIR) -> list[Path]:
    """모든 빌드에 대해 sensor 시퀀스 캐시 빌드.

    이미 존재하는 파일은 건너뜀 (덮어쓰려면 삭제 후 재실행).
    """
    if build_ids is None:
        build_ids = list(config.BUILDS.keys())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for bid in build_ids:
        out_path = out_dir / f"sensor_stacks_{bid}.h5"
        if out_path.exists():
            print(f"[cache] {out_path.name} already exists, skip")
            paths.append(out_path)
            continue
        print(f"\n[cache] Building {bid} → {out_path}")
        paths.append(_build_one_build(bid, out_dir))

    return paths


def verify_v0_consistency(build_ids: list[str] | None = None,
                          cache_v0_dir: Path = config.LSTM_CACHE_DIR,
                          cache_sensor_dir: Path = config.LSTM_DUAL_IMG_4_SENSOR_7_CACHE_DIR) -> None:
    """v0 카메라 캐시와 sensor 캐시의 lengths/sv_indices/sample_ids 일치 검증.

    valid_mask 가 동일 규칙(part_ids > 0 in SV xy) 으로 계산되므로 비트 단위 일치해야 함.
    """
    if build_ids is None:
        build_ids = list(config.BUILDS.keys())

    cache_v0_dir = Path(cache_v0_dir)
    cache_sensor_dir = Path(cache_sensor_dir)

    for bid in build_ids:
        v0 = cache_v0_dir / f"crop_stacks_{bid}.h5"
        sn = cache_sensor_dir / f"sensor_stacks_{bid}.h5"
        if not v0.exists():
            raise FileNotFoundError(f"v0 캐시 누락: {v0} — `python -m Sources.vppm.lstm.run --phase cache` 먼저 실행")
        if not sn.exists():
            raise FileNotFoundError(f"sensor 캐시 누락: {sn} — `--phase cache_sensor` 먼저 실행")

        with h5py.File(v0, "r") as f0, h5py.File(sn, "r") as fs:
            len0 = f0["lengths"][...]
            len_s = fs["lengths"][...]
            sv0 = f0["sv_indices"][...]
            sv_s = fs["sv_indices"][...]
            sid0 = f0["sample_ids"][...]
            sid_s = fs["sample_ids"][...]

        if len0.shape != len_s.shape:
            raise RuntimeError(f"{bid}: lengths shape 불일치 v0={len0.shape} sensor={len_s.shape}")
        if not np.array_equal(len0, len_s):
            n_diff = int((len0 != len_s).sum())
            raise RuntimeError(f"{bid}: lengths {n_diff}/{len0.size} 개가 다름 — v0/sensor 캐시가 다른 valid_mask 로 빌드됨")
        if not np.array_equal(sv0, sv_s):
            raise RuntimeError(f"{bid}: sv_indices 불일치")
        if not np.array_equal(sid0, sid_s):
            raise RuntimeError(f"{bid}: sample_ids 불일치")
        print(f"  [verify] {bid}: OK (N={len0.size}, T_sv median={int(np.median(len0))})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--builds", nargs="+", default=list(config.BUILDS.keys()))
    p.add_argument("--out-dir", type=Path, default=config.LSTM_DUAL_IMG_4_SENSOR_7_CACHE_DIR)
    p.add_argument("--verify", action="store_true",
                   help="기존 v0 캐시와 일치 여부만 검증 (빌드 안 함)")
    args = p.parse_args()

    if args.verify:
        verify_v0_consistency(args.builds, cache_sensor_dir=args.out_dir)
    else:
        build_cache(args.builds, args.out_dir)
        verify_v0_consistency(args.builds, cache_sensor_dir=args.out_dir)
