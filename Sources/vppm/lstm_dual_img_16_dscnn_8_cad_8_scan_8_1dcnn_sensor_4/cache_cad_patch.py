"""Phase S1a — CAD 8×8 패치 시퀀스 캐시 빌드 (lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4).

각 valid SV 에 대해 (T_sv, 2, h, w) 패치 시퀀스 저장:
  - 채널 0: edge_proximity     = (3.0  − distance_from_edge      ) × cad_mask  [mm]
  - 채널 1: overhang_proximity = (71.0 − distance_from_overhang  ) × cad_mask  [layers]
  - inversion 의미: 0 = nominal (interior / no recent overhang), saturation = signal
  - cad_mask 픽셀곱: 분말 영역 픽셀 = 0 (= nominal, 다른 시퀀스 입력과 컨벤션 통일)
  - 가우시안 블러는 baseline 과 동일 (sigma_px=GAUSSIAN_STD_PIXELS≈3.76)

valid_mask = part_ids > 0 in SV xy 영역 — v0 카메라 / dscnn 캐시와 동일 규칙.
overhang 누적 상태 (`_last_overhang_layer`) 는 빌드 전체에 걸쳐 layer-major 로 carry-over.

산출물: experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache/cad_patch_stacks_{B1.x}.h5
  - cad_patch  (N, T_max=70, 2, H, W) float16, zero-padded — H=W=8 (`LSTM_CROP_{H,W}`)
  - lengths    (N,) int16
  - sv_indices (N, 3) int32
  - sample_ids (N,) int32

이미 존재하는 파일은 건너뜀 (덮어쓰려면 삭제 후 재실행).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from tqdm import tqdm

from ..common import config
from ..common.supervoxel import SuperVoxelGrid, find_valid_supervoxels


def _build_one_build(build_id: str, out_dir: Path) -> Path:
    hdf5 = str(config.hdf5_path(build_id))
    out_path = out_dir / f"cad_patch_stacks_{build_id}.h5"

    grid = SuperVoxelGrid.from_hdf5(hdf5)
    valid = find_valid_supervoxels(grid, hdf5)
    indices = valid["voxel_indices"]              # (N, 3)
    N = len(indices)
    if N == 0:
        print(f"  {build_id}: no valid SVs, skip")
        return out_path

    T_max = config.LSTM_T_MAX
    H_patch = config.LSTM_FULL86_CAD_PATCH_H      # = 8
    W_patch = config.LSTM_FULL86_CAD_PATCH_W      # = 8
    n_ch = config.LSTM_FULL86_N_CAD_CH            # = 2
    sigma_px = config.GAUSSIAN_STD_PIXELS
    EDGE_SAT = float(config.DIST_EDGE_SATURATION_MM)               # 3.0
    OH_SAT = float(config.DIST_OVERHANG_SATURATION_LAYERS)         # 71

    # SV 를 z-block 별로 그룹핑 → z-block 단위 처리 + 그 안에서 layer-major 루프
    iz_to_svs: dict[int, list[int]] = {}
    for i, (_, _, iz) in enumerate(indices):
        iz_to_svs.setdefault(int(iz), []).append(i)

    print(f"  {build_id}: N={N} SVs, {len(iz_to_svs)} z-blocks, "
          f"patch=({H_patch}×{W_patch}), n_ch={n_ch}, "
          f"sigma_px={sigma_px:.2f}, EDGE_SAT={EDGE_SAT}, OH_SAT={OH_SAT}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as out, h5py.File(hdf5, "r") as f:
        part_ds = f["slices/part_ids"]
        H_img = grid.image_h
        W_img = grid.image_w

        cad_patch = out.create_dataset(
            "cad_patch", shape=(N, T_max, n_ch, H_patch, W_patch), dtype="float16",
            compression="gzip", compression_opts=4,
            chunks=(1, T_max, n_ch, H_patch, W_patch),
        )
        lengths = out.create_dataset("lengths", shape=(N,), dtype="int16")
        out.create_dataset("sv_indices", data=indices.astype(np.int32))
        out.create_dataset("sample_ids", data=valid["sample_ids"].astype(np.int32))

        # === overhang carry-over 상태 (빌드 전체에 걸쳐 유지, layer-major) ===
        last_overhang = np.full((H_img, W_img), -np.inf, dtype=np.float32)
        prev_cad: np.ndarray | None = None

        # z-block 을 iz 오름차순으로 처리해야 layer 순서가 보장됨
        sorted_izs = sorted(iz_to_svs.keys())
        for iz in tqdm(sorted_izs, desc=f"{build_id} z-blocks"):
            l0, l1 = grid.get_layer_range(iz)
            block_part = part_ds[l0:l1]                    # (Tb, H_img, W_img)
            Tb = block_part.shape[0]

            sv_list = iz_to_svs[iz]                        # 이 z-block 의 SV 인덱스
            # 각 SV 의 valid_mask 와 출력 시퀀스 메타데이터 준비
            sv_meta: list[dict] = []
            for sv_i in sv_list:
                ix, iy, _ = indices[sv_i]
                r0, r1, c0, c1 = grid.get_pixel_range(int(ix), int(iy))
                part_crop = block_part[:, r0:r1, c0:c1]    # (Tb, h, w)
                valid_mask = (part_crop > 0).reshape(Tb, -1).any(axis=1)
                if not valid_mask.any():
                    valid_mask[Tb // 2] = True             # 안전장치 (lstm/crop_stacks.py:94)

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

            # === layer-major 루프: overhang carry-over 상태 유지 + SV 패치 저장 ===
            for off in range(Tb):
                layer = l0 + off
                cad_mask_layer = (block_part[off] > 0)     # bool (H_img, W_img)

                # 채널 0: distance_from_edge → edge_proximity
                if cad_mask_layer.any():
                    dist_edge = distance_transform_edt(cad_mask_layer) * config.PIXEL_SIZE_MM
                    dist_edge = np.minimum(dist_edge, EDGE_SAT).astype(np.float32)
                    dist_edge = gaussian_filter(dist_edge, sigma=sigma_px)
                else:
                    dist_edge = np.zeros((H_img, W_img), dtype=np.float32)

                # 채널 1: distance_from_overhang (vertical column carry-over)
                # 빌드 첫 layer (prev_cad = None) 는 빌드 플레이트 위 출력 → overhang 미검출
                if prev_cad is not None:
                    overhang = cad_mask_layer & (~prev_cad)
                    if overhang.any():
                        last_overhang[overhang] = float(layer)
                dist_oh = (float(layer) - last_overhang)
                dist_oh = np.minimum(dist_oh, OH_SAT).astype(np.float32)
                dist_oh = gaussian_filter(dist_oh, sigma=sigma_px)

                prev_cad = cad_mask_layer.copy()

                # === Inversion + cad_mask 픽셀곱 (layer 전체 맵 단위) ===
                cad_mask_f32 = cad_mask_layer.astype(np.float32)
                edge_prox = (EDGE_SAT - dist_edge) * cad_mask_f32      # (H_img, W_img)
                oh_prox   = (OH_SAT   - dist_oh)   * cad_mask_f32      # (H_img, W_img)

                # 이 layer 가 활성인 SV 만 패치 저장
                active_svs = [m for m in sv_meta if m["valid_mask"][off]]
                for m in active_svs:
                    ti = m["offset_to_ti"].get(int(off))
                    if ti is None:
                        continue
                    r0, r1, c0, c1 = m["r0"], m["r1"], m["c0"], m["c1"]
                    edge_patch = edge_prox[r0:r1, c0:c1]               # (h, w) ≈ (8, 8)
                    oh_patch   = oh_prox  [r0:r1, c0:c1]

                    # 가장자리 SV: 패치 < 8×8 → zero-pad (= nominal)
                    h_actual, w_actual = edge_patch.shape
                    h_use = min(h_actual, H_patch)
                    w_use = min(w_actual, W_patch)
                    m["seq"][ti, 0, :h_use, :w_use] = edge_patch[:h_use, :w_use]
                    m["seq"][ti, 1, :h_use, :w_use] = oh_patch  [:h_use, :w_use]

            # 결과를 cad_patch / lengths 에 기록
            for m in sv_meta:
                T_sv_clip = m["T_sv_clip"]
                if T_sv_clip > 0:
                    cad_patch[m["sv_i"], :T_sv_clip] = m["seq"]
                lengths[m["sv_i"]] = T_sv_clip

        # === 메타데이터 attrs ===
        out.attrs["T_max"] = T_max
        out.attrs["n_channels"] = n_ch
        out.attrs["patch_h"] = H_patch
        out.attrs["patch_w"] = W_patch
        out.attrs["channel_names"] = np.array(["edge_proximity", "overhang_proximity"], dtype="S")
        out.attrs["channel_units"] = np.array(["mm", "layers"], dtype="S")
        out.attrs["edge_saturation_mm"] = EDGE_SAT
        out.attrs["overhang_saturation_layers"] = OH_SAT
        out.attrs["inversion_applied"] = True
        out.attrs["mask_applied"] = True
        out.attrs["inversion_formula"] = (
            f"edge_proximity = ({EDGE_SAT} - distance_from_edge) * cad_mask; "
            f"overhang_proximity = ({OH_SAT} - distance_from_overhang) * cad_mask; "
            "convention: 0 = nominal (interior / no recent overhang / 분말), "
            "saturation = signal max"
        )
        out.attrs["sigma_px"] = float(sigma_px)
        out.attrs["build_id"] = build_id
        out.attrs["n_sv"] = N
        out.attrs["valid_layer_rule"] = "part_ids>0 in SV xy region (matches lstm/crop_stacks.py)"

    print(f"  saved {out_path}  ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path


def build_cache(build_ids: list[str] | None = None,
                out_dir: Path = config.LSTM_FULL86_CACHE_CAD_DIR) -> list[Path]:
    """모든 빌드에 대해 CAD 패치 시퀀스 캐시 빌드.

    이미 존재하는 파일은 건너뜀 (덮어쓰려면 삭제 후 재실행).
    """
    if build_ids is None:
        build_ids = list(config.BUILDS.keys())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for bid in build_ids:
        out_path = out_dir / f"cad_patch_stacks_{bid}.h5"
        if out_path.exists():
            print(f"[cache] {out_path.name} already exists, skip")
            paths.append(out_path)
            continue
        print(f"\n[cache] Building {bid} → {out_path}")
        paths.append(_build_one_build(bid, out_dir))

    return paths


def verify_cad_patch_v0_consistency(
    build_ids: list[str] | None = None,
    cache_v0_dir: Path = config.LSTM_CACHE_DIR,
    cache_cad_dir: Path = config.LSTM_FULL86_CACHE_CAD_DIR,
) -> None:
    """v0 카메라 캐시와 CAD 패치 캐시의 lengths/sv_indices/sample_ids 일치 + 채널 범위 sanity check.

    valid_mask 정의가 동일 (part_ids > 0 in SV xy) 하므로 비트 단위 일치해야 함.
    """
    if build_ids is None:
        build_ids = list(config.BUILDS.keys())

    cache_v0_dir = Path(cache_v0_dir)
    cache_cad_dir = Path(cache_cad_dir)
    EDGE_SAT = float(config.DIST_EDGE_SATURATION_MM)
    OH_SAT = float(config.DIST_OVERHANG_SATURATION_LAYERS)

    for bid in build_ids:
        v0 = cache_v0_dir / f"crop_stacks_{bid}.h5"
        cad = cache_cad_dir / f"cad_patch_stacks_{bid}.h5"
        if not v0.exists():
            raise FileNotFoundError(
                f"v0 캐시 누락: {v0} — `python -m Sources.vppm.lstm.run --phase cache` 먼저 실행"
            )
        if not cad.exists():
            raise FileNotFoundError(
                f"CAD 패치 캐시 누락: {cad} — `--phase cache_cad_patch` 먼저 실행"
            )

        with h5py.File(v0, "r") as f0, h5py.File(cad, "r") as fc:
            len0 = f0["lengths"][...]
            len_c = fc["lengths"][...]
            sv0 = f0["sv_indices"][...]
            sv_c = fc["sv_indices"][...]
            sid0 = f0["sample_ids"][...]
            sid_c = fc["sample_ids"][...]
            inversion = bool(fc.attrs.get("inversion_applied", False))
            mask_app = bool(fc.attrs.get("mask_applied", False))
            n_ch = int(fc.attrs.get("n_channels", -1))

            if not (inversion and mask_app):
                raise RuntimeError(
                    f"{bid}: cad_patch attrs 검증 실패 — inversion={inversion}, mask={mask_app}. "
                    "캐시 재빌드 필요."
                )
            if n_ch != config.LSTM_FULL86_N_CAD_CH:
                raise RuntimeError(f"{bid}: n_channels={n_ch} ≠ {config.LSTM_FULL86_N_CAD_CH}")

            # lengths/sv_indices/sample_ids 비트 일치
            if len0.shape != len_c.shape or not np.array_equal(len0, len_c):
                n_diff = int((len0 != len_c).sum()) if len0.shape == len_c.shape else -1
                raise RuntimeError(
                    f"{bid}: lengths 불일치 v0 vs cad ({n_diff}/{len0.size} 다름)"
                )
            if not np.array_equal(sv0, sv_c):
                raise RuntimeError(f"{bid}: sv_indices 불일치 (v0 vs cad)")
            if not np.array_equal(sid0, sid_c):
                raise RuntimeError(f"{bid}: sample_ids 불일치 (v0 vs cad)")

            # 값 범위 sanity check (랜덤 샘플 100 SV)
            N = len_c.size
            sample_n = min(100, N)
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(N, size=sample_n, replace=False)
            for sv_i in sample_idx:
                T_sv = int(len_c[sv_i])
                if T_sv == 0:
                    continue
                seq = fc["cad_patch"][sv_i, :T_sv]                  # (T, 2, H, W) float16
                ch0 = seq[:, 0]
                ch1 = seq[:, 1]
                if ch0.min() < -1e-2 or ch0.max() > EDGE_SAT + 1e-2:
                    raise RuntimeError(
                        f"{bid} SV {sv_i}: ch0 (edge_proximity) 범위 [{ch0.min()}, {ch0.max()}] "
                        f"⊄ [0, {EDGE_SAT}]"
                    )
                if ch1.min() < -1e-2 or ch1.max() > OH_SAT + 1e-2:
                    raise RuntimeError(
                        f"{bid} SV {sv_i}: ch1 (overhang_proximity) 범위 "
                        f"[{ch1.min()}, {ch1.max()}] ⊄ [0, {OH_SAT}]"
                    )

        print(f"  [verify] {bid}: OK (N={len_c.size}, T_sv median={int(np.median(len_c))}, "
              f"inversion+mask 적용, 값 범위 ∈ [0, sat])")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--builds", nargs="+", default=list(config.BUILDS.keys()))
    p.add_argument("--out-dir", type=Path,
                   default=config.LSTM_FULL86_CACHE_CAD_DIR)
    p.add_argument("--verify", action="store_true",
                   help="기존 v0 캐시와 일치 여부만 검증 (빌드 안 함)")
    args = p.parse_args()

    if args.verify:
        verify_cad_patch_v0_consistency(args.builds, cache_cad_dir=args.out_dir)
    else:
        build_cache(args.builds, args.out_dir)
        verify_cad_patch_v0_consistency(args.builds, cache_cad_dir=args.out_dir)
