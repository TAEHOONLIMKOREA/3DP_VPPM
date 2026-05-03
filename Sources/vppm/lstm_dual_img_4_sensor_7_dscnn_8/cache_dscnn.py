"""Phase S1 — DSCNN segmentation 시퀀스 캐시 빌드 (lstm_dual_img_4_sensor_7_dscnn_8).

각 valid SV 에 대해 8-channel DSCNN segmentation 시계열을 추출:
  - 시퀀스 길이 = 카메라 v0 캐시의 lengths 와 동일 (valid_mask = part_ids > 0 in SV xy)
  - 채널: DSCNN_FEATURE_MAP 의 8 paper class (hdf5 cls_ids = [0, 1, 3, 5, 6, 7, 8, 10])
  - 값은 각 layer 에서 (가우시안 블러 적용된 segmentation) ∩ (SV crop 내 CAD mask) 의 평균
    (baseline `_extract_dscnn_features_block` 의 layer-별 결과를 누적하지 않고 그대로 (T, 8) 시퀀스로 저장)
  - raw 보존 — 정규화는 dataset.py 의 build_normalized_dataset 에서 수행

산출물: experiments/vppm_lstm_dual_img_4_sensor_7_dscnn_8/cache/dscnn_stacks_{B1.x}.h5
  - dscnn      (N, T_max=70, 8) float32, zero-padded
  - lengths    (N,) int16
  - sv_indices (N, 3) int32
  - sample_ids (N,) int32

이미 존재하는 파일은 건너뜀 (덮어쓰려면 삭제 후 재실행).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from ..common import config
from ..common.supervoxel import SuperVoxelGrid, find_valid_supervoxels


def _build_one_build(build_id: str, out_dir: Path) -> Path:
    hdf5 = str(config.hdf5_path(build_id))
    out_path = out_dir / f"dscnn_stacks_{build_id}.h5"

    grid = SuperVoxelGrid.from_hdf5(hdf5)
    valid = find_valid_supervoxels(grid, hdf5)
    indices = valid["voxel_indices"]              # (N, 3)
    N = len(indices)
    if N == 0:
        print(f"  {build_id}: no valid SVs, skip")
        return out_path

    T_max = config.LSTM_T_MAX
    n_ch = config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_N_CHANNELS  # = 8
    sigma_px = config.GAUSSIAN_STD_PIXELS

    # 8 hdf5 class ids — DSCNN_FEATURE_MAP 의 hdf5_class_id 값들
    dscnn_class_ids = [v[0] for v in config.DSCNN_FEATURE_MAP.values()]
    dscnn_channel_names = [v[1] for v in config.DSCNN_FEATURE_MAP.values()]

    iz_to_svs: dict[int, list[int]] = {}
    for i, (_, _, iz) in enumerate(indices):
        iz_to_svs.setdefault(int(iz), []).append(i)

    print(f"  {build_id}: N={N} SVs, {len(iz_to_svs)} z-blocks, "
          f"{n_ch} DSCNN channels, sigma_px={sigma_px:.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as out, h5py.File(hdf5, "r") as f:
        part_ds = f["slices/part_ids"]
        # segmentation_results 데이터셋 핸들 — 클래스별로 1번만 열기
        seg_dsets = []
        for cls_id in dscnn_class_ids:
            seg_key = f"slices/segmentation_results/{cls_id}"
            if seg_key in f:
                seg_dsets.append(f[seg_key])
            else:
                seg_dsets.append(None)

        dscnn = out.create_dataset(
            "dscnn", shape=(N, T_max, n_ch), dtype="float32",
            compression="gzip", compression_opts=4,
            chunks=(1, T_max, n_ch),
        )
        lengths = out.create_dataset("lengths", shape=(N,), dtype="int16")
        out.create_dataset("sv_indices", data=indices.astype(np.int32))
        out.create_dataset("sample_ids", data=valid["sample_ids"].astype(np.int32))

        for iz in tqdm(sorted(iz_to_svs.keys()), desc=f"{build_id} z-blocks"):
            l0, l1 = grid.get_layer_range(iz)
            block_part = part_ds[l0:l1]                       # (Tb, H_img, W_img)
            Tb = block_part.shape[0]

            # 이 z-block 의 layer 별로 8 채널 segmentation + 가우시안 블러
            # (메모리: Tb × 8 × H × W × 4B — 70 × 8 × 1842² × 4B ≈ 7.6 GB → block 단위 cache 부담 큼.
            #  대안: layer 단위로 즉석 로드/블러 후 사용. 한 layer 의 SV 들이 같은 layer 데이터를 공유.)
            #
            # 전략: layer 단위 loop 안에서 (1) part_layer 로드 (2) 8 채널 seg 로드 + 블러
            #       (3) 그 layer 에 속한 모든 활성 SV 에 대해 patch CAD ∩ seg crop 평균 계산.
            #
            # 각 SV 는 z-block 내 valid_mask 가 True 인 layer 에만 기여 — 그래서 SV 별로 valid_mask
            # 를 먼저 구해두고, layer 단위 loop 에서 active 인 SV 만 처리.

            sv_list = iz_to_svs[iz]   # 이 z-block 의 SV 인덱스들
            # 각 SV 의 valid_mask, T_sv, output sequence 준비
            sv_meta: list[dict] = []
            for sv_i in sv_list:
                ix, iy, _ = indices[sv_i]
                r0, r1, c0, c1 = grid.get_pixel_range(int(ix), int(iy))
                part_crop = block_part[:, r0:r1, c0:c1]       # (Tb, h, w)

                # lstm/crop_stacks.py 와 동일한 valid_mask 정의
                valid_mask = (part_crop > 0).reshape(Tb, -1).any(axis=1)
                if not valid_mask.any():
                    valid_mask[Tb // 2] = True               # 안전장치 (lstm/crop_stacks.py:94)

                T_sv = int(valid_mask.sum())
                T_sv_clip = min(T_sv, T_max)
                # 이 SV 가 활성인 z-block 내 layer offset 들 (오름차순)
                active_offsets = np.flatnonzero(valid_mask)
                if len(active_offsets) > T_sv_clip:
                    active_offsets = active_offsets[:T_sv_clip]
                # ti (시퀀스 인덱스) ↔ z-block 내 layer offset 매핑
                offset_to_ti = {int(off): ti for ti, off in enumerate(active_offsets)}
                seq = np.zeros((T_sv_clip, n_ch), dtype=np.float32)
                sv_meta.append({
                    "sv_i": sv_i,
                    "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                    "valid_mask": valid_mask,
                    "T_sv_clip": T_sv_clip,
                    "offset_to_ti": offset_to_ti,
                    "seq": seq,
                })

            # layer 단위로 segmentation 로드 + 블러 → 활성 SV 에 분배
            for off in range(Tb):
                # 이 layer 가 활성인 SV 만 추리기
                active_svs = [m for m in sv_meta if m["valid_mask"][off]]
                if not active_svs:
                    continue
                layer = l0 + off
                part_layer = block_part[off]                  # (H_img, W_img)
                # 8 채널 seg 로드 + 블러
                seg_smoothed = []
                for ds in seg_dsets:
                    if ds is None:
                        seg_smoothed.append(None)
                        continue
                    seg = ds[layer].astype(np.float32)
                    seg = gaussian_filter(seg, sigma=sigma_px)
                    seg_smoothed.append(seg)

                for m in active_svs:
                    r0, r1, c0, c1 = m["r0"], m["r1"], m["c0"], m["c1"]
                    patch_cad = part_layer[r0:r1, c0:c1] > 0
                    n_cad = int(patch_cad.sum())
                    ti = m["offset_to_ti"].get(int(off))
                    if ti is None:
                        continue
                    if n_cad == 0:
                        # 드물게 SV xy 안에 part 가 있어도 layer 내 CAD 가 0 인 경우
                        # (일관성: baseline 도 n_cad==0 이면 가산 안 함). seq 는 0 으로 둠.
                        continue
                    for ci, seg in enumerate(seg_smoothed):
                        if seg is None:
                            continue
                        m["seq"][ti, ci] = seg[r0:r1, c0:c1][patch_cad].mean()

            # 결과를 dscnn / lengths 에 기록
            for m in sv_meta:
                T_sv_clip = m["T_sv_clip"]
                if T_sv_clip > 0:
                    dscnn[m["sv_i"], :T_sv_clip] = m["seq"]
                lengths[m["sv_i"]] = T_sv_clip

        out.attrs["T_max"] = T_max
        out.attrs["n_channels"] = n_ch
        out.attrs["channel_names"] = np.array(dscnn_channel_names, dtype="S")
        out.attrs["dscnn_class_ids"] = np.array(dscnn_class_ids, dtype=np.int32)
        out.attrs["build_id"] = build_id
        out.attrs["n_sv"] = N
        out.attrs["sigma_px"] = float(sigma_px)
        out.attrs["valid_layer_rule"] = "part_ids>0 in SV xy region (matches lstm/crop_stacks.py)"

    print(f"  saved {out_path}  ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path


def build_cache(build_ids: list[str] | None = None,
                out_dir: Path = config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_DIR) -> list[Path]:
    """모든 빌드에 대해 DSCNN 시퀀스 캐시 빌드.

    이미 존재하는 파일은 건너뜀 (덮어쓰려면 삭제 후 재실행).
    """
    if build_ids is None:
        build_ids = list(config.BUILDS.keys())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for bid in build_ids:
        out_path = out_dir / f"dscnn_stacks_{bid}.h5"
        if out_path.exists():
            print(f"[cache] {out_path.name} already exists, skip")
            paths.append(out_path)
            continue
        print(f"\n[cache] Building {bid} → {out_path}")
        paths.append(_build_one_build(bid, out_dir))

    return paths


def verify_dscnn_v0_consistency(
    build_ids: list[str] | None = None,
    cache_v0_dir: Path = config.LSTM_CACHE_DIR,
    cache_dscnn_dir: Path = config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_DIR,
) -> None:
    """v0 카메라 캐시와 DSCNN 캐시의 lengths/sv_indices/sample_ids 일치 검증.

    valid_mask 가 동일 규칙(part_ids > 0 in SV xy) 으로 계산되므로 비트 단위 일치해야 함.
    """
    if build_ids is None:
        build_ids = list(config.BUILDS.keys())

    cache_v0_dir = Path(cache_v0_dir)
    cache_dscnn_dir = Path(cache_dscnn_dir)

    for bid in build_ids:
        v0 = cache_v0_dir / f"crop_stacks_{bid}.h5"
        dn = cache_dscnn_dir / f"dscnn_stacks_{bid}.h5"
        if not v0.exists():
            raise FileNotFoundError(
                f"v0 캐시 누락: {v0} — `python -m Sources.vppm.lstm.run --phase cache` 먼저 실행"
            )
        if not dn.exists():
            raise FileNotFoundError(
                f"DSCNN 캐시 누락: {dn} — `--phase cache_dscnn` 먼저 실행"
            )

        with h5py.File(v0, "r") as f0, h5py.File(dn, "r") as fd:
            len0 = f0["lengths"][...]
            len_d = fd["lengths"][...]
            sv0 = f0["sv_indices"][...]
            sv_d = fd["sv_indices"][...]
            sid0 = f0["sample_ids"][...]
            sid_d = fd["sample_ids"][...]

        if len0.shape != len_d.shape:
            raise RuntimeError(
                f"{bid}: lengths shape 불일치 v0={len0.shape} dscnn={len_d.shape}"
            )
        if not np.array_equal(len0, len_d):
            n_diff = int((len0 != len_d).sum())
            raise RuntimeError(
                f"{bid}: lengths {n_diff}/{len0.size} 개가 다름 — "
                "v0/dscnn 캐시가 다른 valid_mask 로 빌드됨"
            )
        if not np.array_equal(sv0, sv_d):
            raise RuntimeError(f"{bid}: sv_indices 불일치 (v0 vs dscnn)")
        if not np.array_equal(sid0, sid_d):
            raise RuntimeError(f"{bid}: sample_ids 불일치 (v0 vs dscnn)")
        print(f"  [verify] {bid}: OK (N={len0.size}, T_sv median={int(np.median(len0))})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--builds", nargs="+", default=list(config.BUILDS.keys()))
    p.add_argument("--out-dir", type=Path,
                   default=config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_DIR)
    p.add_argument("--verify", action="store_true",
                   help="기존 v0 캐시와 일치 여부만 검증 (빌드 안 함)")
    args = p.parse_args()

    if args.verify:
        verify_dscnn_v0_consistency(args.builds, cache_dscnn_dir=args.out_dir)
    else:
        build_cache(args.builds, args.out_dir)
        verify_dscnn_v0_consistency(args.builds, cache_dscnn_dir=args.out_dir)
