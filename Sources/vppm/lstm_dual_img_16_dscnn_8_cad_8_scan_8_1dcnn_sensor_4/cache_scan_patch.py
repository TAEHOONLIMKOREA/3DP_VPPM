"""Phase S1b — Scan 8×8 패치 시퀀스 캐시 빌드 (lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4).

각 valid SV 에 대해 (T_sv, 2, h, w) 패치 시퀀스 저장:
  - 채널 0: return_delay      raw 픽셀 맵 (s, saturation 0.75, **NaN→0 변환**)
  - 채널 1: stripe_boundaries raw 픽셀 맵 (a.u., Sobel RMS, baseline 이 미용융=0 부여)
  - inversion 미적용 — raw 의미 자체가 "0=no melt(nominal), large=signal" 컨벤션 일치
  - 마스킹 미적용 — baseline `scan_features.py` 가 이미 미용융 픽셀에 0 부여 (#21) /
    NaN→0 변환으로 동일 처리 (#20)

valid_mask = part_ids > 0 in SV xy 영역 — v0 카메라 / dscnn / cad 캐시와 동일 규칙.
활성 layer 인데 melt 픽셀 0 인 경우 → 패치 전체 0 (= nominal) 으로 채움.

산출물: experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache/scan_patch_stacks_{B1.x}.h5
  - scan_patch (N, T_max=70, 2, H, W) float16, zero-padded — H=W=8
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

from ..baseline.scan_features import (
    build_melt_time_map,
    compute_return_delay_map,
    compute_stripe_boundaries_map,
)
from ..common import config
from ..common.supervoxel import SuperVoxelGrid, find_valid_supervoxels


def _build_one_build(build_id: str, out_dir: Path) -> Path:
    hdf5 = str(config.hdf5_path(build_id))
    out_path = out_dir / f"scan_patch_stacks_{build_id}.h5"

    grid = SuperVoxelGrid.from_hdf5(hdf5)
    valid = find_valid_supervoxels(grid, hdf5)
    indices = valid["voxel_indices"]
    N = len(indices)
    if N == 0:
        print(f"  {build_id}: no valid SVs, skip")
        return out_path

    T_max = config.LSTM_T_MAX
    H_patch = config.LSTM_FULL86_SCAN_PATCH_H      # = 8
    W_patch = config.LSTM_FULL86_SCAN_PATCH_W      # = 8
    n_ch = config.LSTM_FULL86_N_SCAN_CH            # = 2
    sat_s = 0.75                                    # baseline #20 saturation
    pixel_size_mm = config.PIXEL_SIZE_MM
    kernel_px = max(1, int(round(1.0 / pixel_size_mm)))    # 1 mm 이웃 ≈ 8 px

    iz_to_svs: dict[int, list[int]] = {}
    for i, (_, _, iz) in enumerate(indices):
        iz_to_svs.setdefault(int(iz), []).append(i)

    print(f"  {build_id}: N={N} SVs, {len(iz_to_svs)} z-blocks, "
          f"patch=({H_patch}×{W_patch}), n_ch={n_ch}, "
          f"sat_s={sat_s}, kernel_px={kernel_px}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as out, h5py.File(hdf5, "r") as f:
        part_ds = f["slices/part_ids"]
        H_img = grid.image_h
        W_img = grid.image_w
        img_shape = (H_img, W_img)

        scan_patch = out.create_dataset(
            "scan_patch", shape=(N, T_max, n_ch, H_patch, W_patch), dtype="float16",
            compression="gzip", compression_opts=4,
            chunks=(1, T_max, n_ch, H_patch, W_patch),
        )
        lengths = out.create_dataset("lengths", shape=(N,), dtype="int16")
        out.create_dataset("sv_indices", data=indices.astype(np.int32))
        out.create_dataset("sample_ids", data=valid["sample_ids"].astype(np.int32))

        sorted_izs = sorted(iz_to_svs.keys())
        for iz in tqdm(sorted_izs, desc=f"{build_id} z-blocks"):
            l0, l1 = grid.get_layer_range(iz)
            block_part = part_ds[l0:l1]
            Tb = block_part.shape[0]

            sv_list = iz_to_svs[iz]
            sv_meta: list[dict] = []
            for sv_i in sv_list:
                ix, iy, _ = indices[sv_i]
                r0, r1, c0, c1 = grid.get_pixel_range(int(ix), int(iy))
                part_crop = block_part[:, r0:r1, c0:c1]
                valid_mask = (part_crop > 0).reshape(Tb, -1).any(axis=1)
                if not valid_mask.any():
                    valid_mask[Tb // 2] = True

                T_sv = int(valid_mask.sum())
                T_sv_clip = min(T_sv, T_max)
                active_offsets = np.flatnonzero(valid_mask)
                if len(active_offsets) > T_sv_clip:
                    active_offsets = active_offsets[:T_sv_clip]
                offset_to_ti = {int(off): ti for ti, off in enumerate(active_offsets)}
                seq = np.zeros((T_sv_clip, n_ch, H_patch, W_patch), dtype=np.float16)
                sv_meta.append({
                    "sv_i": sv_i,
                    "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                    "valid_mask": valid_mask,
                    "T_sv_clip": T_sv_clip,
                    "offset_to_ti": offset_to_ti,
                    "seq": seq,
                })

            # === layer-major 루프: melt-time map → rd / sb 픽셀 맵 → SV 패치 저장 ===
            for off in range(Tb):
                layer = l0 + off
                # 활성 SV 가 없으면 layer 처리 자체 skip (성능)
                active_svs = [m for m in sv_meta if m["valid_mask"][off]]
                if not active_svs:
                    continue

                scan_key = f"scans/{layer}"
                if scan_key not in f:
                    # 이 layer 는 scan 데이터 부재 → 패치 전체 0 (= nominal) 유지
                    continue

                scans = f[scan_key][...]
                if scans.size == 0:
                    continue

                mt_map = build_melt_time_map(scans, img_shape, pixel_size_mm)
                rd_map = compute_return_delay_map(mt_map, kernel_px=kernel_px, sat_s=sat_s)
                sb_map = compute_stripe_boundaries_map(mt_map)

                # NaN → 0 변환 (CNN 입력으로 NaN 흘리면 폭발). sb_map 은 이미 0 채움됨.
                rd_map = np.nan_to_num(rd_map, nan=0.0, copy=False).astype(np.float32)

                for m in active_svs:
                    ti = m["offset_to_ti"].get(int(off))
                    if ti is None:
                        continue
                    r0, r1, c0, c1 = m["r0"], m["r1"], m["c0"], m["c1"]
                    rd_patch = rd_map[r0:r1, c0:c1]
                    sb_patch = sb_map[r0:r1, c0:c1]

                    h_actual, w_actual = rd_patch.shape
                    h_use = min(h_actual, H_patch)
                    w_use = min(w_actual, W_patch)
                    m["seq"][ti, 0, :h_use, :w_use] = rd_patch[:h_use, :w_use]
                    m["seq"][ti, 1, :h_use, :w_use] = sb_patch[:h_use, :w_use]

            for m in sv_meta:
                T_sv_clip = m["T_sv_clip"]
                if T_sv_clip > 0:
                    scan_patch[m["sv_i"], :T_sv_clip] = m["seq"]
                lengths[m["sv_i"]] = T_sv_clip

        out.attrs["T_max"] = T_max
        out.attrs["n_channels"] = n_ch
        out.attrs["patch_h"] = H_patch
        out.attrs["patch_w"] = W_patch
        out.attrs["channel_names"] = np.array(["return_delay", "stripe_boundaries"], dtype="S")
        out.attrs["channel_units"] = np.array(["s", "a.u."], dtype="S")
        out.attrs["return_delay_saturation_s"] = sat_s
        out.attrs["kernel_px"] = kernel_px
        out.attrs["inversion_applied"] = False
        out.attrs["mask_applied"] = False
        out.attrs["convention"] = (
            "0 = no melt (nominal); large = signal. "
            "return_delay: raw + NaN→0 변환. stripe_boundaries: raw (baseline 이 미용융=0 부여)."
        )
        out.attrs["build_id"] = build_id
        out.attrs["n_sv"] = N
        out.attrs["valid_layer_rule"] = "part_ids>0 in SV xy region (matches lstm/crop_stacks.py)"

    print(f"  saved {out_path}  ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path


def build_cache(build_ids: list[str] | None = None,
                out_dir: Path = config.LSTM_FULL86_CACHE_SCAN_DIR) -> list[Path]:
    """모든 빌드에 대해 Scan 패치 시퀀스 캐시 빌드. 이미 존재하면 skip."""
    if build_ids is None:
        build_ids = list(config.BUILDS.keys())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for bid in build_ids:
        out_path = out_dir / f"scan_patch_stacks_{bid}.h5"
        if out_path.exists():
            print(f"[cache] {out_path.name} already exists, skip")
            paths.append(out_path)
            continue
        print(f"\n[cache] Building {bid} → {out_path}")
        paths.append(_build_one_build(bid, out_dir))

    return paths


def verify_scan_patch_v0_consistency(
    build_ids: list[str] | None = None,
    cache_v0_dir: Path = config.LSTM_CACHE_DIR,
    cache_scan_dir: Path = config.LSTM_FULL86_CACHE_SCAN_DIR,
) -> None:
    """v0 카메라 캐시와 Scan 패치 캐시의 lengths/sv_indices/sample_ids 비트 일치 + attrs 검증."""
    if build_ids is None:
        build_ids = list(config.BUILDS.keys())

    cache_v0_dir = Path(cache_v0_dir)
    cache_scan_dir = Path(cache_scan_dir)
    sat_s = 0.75

    for bid in build_ids:
        v0 = cache_v0_dir / f"crop_stacks_{bid}.h5"
        sn = cache_scan_dir / f"scan_patch_stacks_{bid}.h5"
        if not v0.exists():
            raise FileNotFoundError(f"v0 캐시 누락: {v0}")
        if not sn.exists():
            raise FileNotFoundError(f"Scan 패치 캐시 누락: {sn}")

        with h5py.File(v0, "r") as f0, h5py.File(sn, "r") as fs:
            len0 = f0["lengths"][...]
            len_s = fs["lengths"][...]
            sv0 = f0["sv_indices"][...]
            sv_s = fs["sv_indices"][...]
            sid0 = f0["sample_ids"][...]
            sid_s = fs["sample_ids"][...]
            inversion = bool(fs.attrs.get("inversion_applied", True))
            mask_app = bool(fs.attrs.get("mask_applied", True))
            n_ch = int(fs.attrs.get("n_channels", -1))

            if inversion or mask_app:
                raise RuntimeError(
                    f"{bid}: scan_patch attrs 검증 실패 — "
                    f"inversion={inversion} (False 기대), mask={mask_app} (False 기대)"
                )
            if n_ch != config.LSTM_FULL86_N_SCAN_CH:
                raise RuntimeError(f"{bid}: n_channels={n_ch} ≠ {config.LSTM_FULL86_N_SCAN_CH}")

            if len0.shape != len_s.shape or not np.array_equal(len0, len_s):
                n_diff = int((len0 != len_s).sum()) if len0.shape == len_s.shape else -1
                raise RuntimeError(
                    f"{bid}: lengths 불일치 v0 vs scan ({n_diff}/{len0.size} 다름)"
                )
            if not np.array_equal(sv0, sv_s):
                raise RuntimeError(f"{bid}: sv_indices 불일치 (v0 vs scan)")
            if not np.array_equal(sid0, sid_s):
                raise RuntimeError(f"{bid}: sample_ids 불일치 (v0 vs scan)")

            # 값 범위 sanity check (랜덤 샘플 100 SV, ch0 ≤ sat_s + 여유)
            N = len_s.size
            sample_n = min(100, N)
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(N, size=sample_n, replace=False)
            for sv_i in sample_idx:
                T_sv = int(len_s[sv_i])
                if T_sv == 0:
                    continue
                seq = fs["scan_patch"][sv_i, :T_sv]                 # (T, 2, H, W) float16
                ch0 = seq[:, 0]
                if ch0.min() < -1e-3 or ch0.max() > sat_s + 1e-2:
                    raise RuntimeError(
                        f"{bid} SV {sv_i}: ch0 (return_delay) 범위 "
                        f"[{ch0.min()}, {ch0.max()}] ⊄ [0, {sat_s}]"
                    )
                # ch1 (stripe_boundaries) 은 Sobel RMS 라 상한 없음 (raw 분포 검사 생략)

        print(f"  [verify] {bid}: OK (N={len_s.size}, T_sv median={int(np.median(len_s))}, "
              f"raw 컨벤션, return_delay ∈ [0, {sat_s}])")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--builds", nargs="+", default=list(config.BUILDS.keys()))
    p.add_argument("--out-dir", type=Path,
                   default=config.LSTM_FULL86_CACHE_SCAN_DIR)
    p.add_argument("--verify", action="store_true",
                   help="기존 v0 캐시와 일치 여부만 검증 (빌드 안 함)")
    args = p.parse_args()

    if args.verify:
        verify_scan_patch_v0_consistency(args.builds, cache_scan_dir=args.out_dir)
    else:
        build_cache(args.builds, args.out_dir)
        verify_scan_patch_v0_consistency(args.builds, cache_scan_dir=args.out_dir)
