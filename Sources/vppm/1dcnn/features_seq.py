"""Phase 1 — SV 당 (70, 21) layer-시퀀스 캐시 생성.

baseline ``FeatureExtractor.extract_features`` 와 동일한 4 그룹을 추출하지만,
**z-축 평균 직전 단계**에서 멈춰 layer 별 raw 값을 그대로 보존한다.
평균 작업은 학습 시 1D CNN + AdaptiveAvgPool 이 학습으로 대신한다.

채널 정렬은 baseline ``FEATURE_NAMES`` 순서 (#0..#20) 와 일치한다.
P4 (#3 build_height, #19 laser_module) 는 70 layer 모두에 동일 값을 broadcast 해
채널 21개 모두 ``(N_sv, 70)`` 로 통일된 입력 텐서를 만든다.

캐시는 ``Sources/pipeline_outputs/experiments/vppm_1dcnn/features/features_seq.npz`` 에
저장되며, ``--validate`` 옵션으로 z-평균이 baseline ``all_features.npz`` 와 ±1e-5
이내인지 채널별로 비교할 수 있다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Sources.vppm.baseline.scan_features import (
    build_melt_time_map,
    compute_return_delay_map,
    compute_stripe_boundaries_map,
)
from Sources.vppm.common import config as common_config
from Sources.vppm.common.supervoxel import SuperVoxelGrid, find_valid_supervoxels

from . import config as exp_config


class FeatureSequenceExtractor:
    """슈퍼복셀 단위로 (70, 21) layer-시퀀스를 추출.

    baseline ``FeatureExtractor`` 와 같은 4 그룹을 처리하지만, z-평균 단계를
    제거하고 layer 별 SV-patch 값을 그대로 누적한다.
    """

    def __init__(self, grid: SuperVoxelGrid, hdf5_path: str):
        self.grid = grid
        self.hdf5_path = hdf5_path
        self.sigma_px = common_config.GAUSSIAN_STD_PIXELS
        # vertical-column distance_from_overhang 누적 상태 (build 전체 carry-over)
        self._prev_cad_layer = None
        self._last_overhang_layer = None

    # ------------------------------------------------------------
    # public
    # ------------------------------------------------------------
    def extract_sequences(self, valid_voxels: dict) -> dict:
        """4 그룹 layer-시퀀스 추출.

        Returns:
            features_seq         : (N_sv, 70, 21) float32
            valid_layer_mask     : (N_sv, 70) bool   — z-블록 안에 있는 layer
            cad_count_per_layer  : (N_sv, 70) int32  — P1 가중 검증용
            melt_count_per_layer : (N_sv, 70) int32  — P2 검증용 (layer 가 melt 픽셀
                                                       1개 이상이면 1, 아니면 0)
        """
        indices = valid_voxels["voxel_indices"]
        n_voxels = len(indices)
        T = self.grid.sv_z_layers   # 70

        features_seq = np.zeros((n_voxels, T, common_config.N_FEATURES), dtype=np.float32)
        valid_layer_mask = np.zeros((n_voxels, T), dtype=bool)
        cad_count_per_layer = np.zeros((n_voxels, T), dtype=np.int32)
        melt_count_per_layer = np.zeros((n_voxels, T), dtype=np.int32)

        # vertical-column overhang 상태 초기화 (build 전체 carry-over)
        self._prev_cad_layer = None
        self._last_overhang_layer = np.full(
            (self.grid.image_h, self.grid.image_w), -np.inf, dtype=np.float32,
        )

        with h5py.File(self.hdf5_path, "r") as f:
            temporal_data = self._load_temporal(f)
            laser_modules = self._load_laser_modules(f)

            for iz in tqdm(range(self.grid.nz), desc="z-blocks"):
                l0, l1 = self.grid.get_layer_range(iz)
                voxel_mask = indices[:, 2] == iz
                if not voxel_mask.any():
                    # 이 z-block 의 overhang 상태도 일관되게 갱신해야 하므로
                    # block_voxels 가 비어 있어도 prev_cad_layer 는 forward 됨.
                    self._advance_overhang_state(f, l0, l1)
                    continue

                block_idx = np.where(voxel_mask)[0]
                block_voxels = indices[voxel_mask]
                n_block = len(block_idx)

                # --- P1: CAD (#0, #1) + DSCNN (#3..#10) — layer 별 패치 평균 ---
                cad_per_layer, dscnn_per_layer, cad_count = self._per_layer_cad_dscnn_block(
                    f, block_voxels, l0, l1,
                )
                # cad_per_layer : (n_block, 70, 2)
                # dscnn_per_layer : (n_block, 70, 8)
                features_seq[block_idx, :, 0:2] = cad_per_layer
                features_seq[block_idx, :, 3:11] = dscnn_per_layer
                cad_count_per_layer[block_idx, :] = cad_count

                # --- P4 #3: build_height (z-block 단일값을 70 layer 에 broadcast) ---
                bh = np.float32(self.grid.get_z_center_mm(iz))
                features_seq[block_idx, :, 2] = bh

                # --- P3: temporal #11..#17 (z-block 내 모든 SV 동일 시계열) ---
                temporal_block = self._per_layer_temporal_block(temporal_data, l0, l1)
                # temporal_block : (70, 7)
                features_seq[block_idx, :, 11:18] = temporal_block[None, :, :]   # broadcast over n_block

                # --- P4 #19: laser_module (part 단위 0/1 — 70 layer broadcast) ---
                part_ids_block = valid_voxels["part_ids"][voxel_mask]
                lm_vals = np.zeros((n_block,), dtype=np.float32)
                lm_valid = np.zeros((n_block,), dtype=bool)
                for i, pid in enumerate(part_ids_block):
                    pid = int(pid)
                    if pid in laser_modules:
                        lm_vals[i] = 0.0 if laser_modules[pid] == 1 else 1.0
                        lm_valid[i] = True
                # baseline 은 lm 미존재 시 NaN → SV 드롭. 시퀀스 캐시도 동일하게 NaN 채움.
                lm_vals[~lm_valid] = np.nan
                features_seq[block_idx, :, 18] = lm_vals[:, None]

                # --- P2: scan #19/#20 (#20 laser_return_delay, #21 laser_stripe_boundaries) ---
                scan_per_layer, melt_count = self._per_layer_scan_block(
                    f, block_voxels, l0, l1,
                )
                features_seq[block_idx, :, 19:21] = scan_per_layer
                melt_count_per_layer[block_idx, :] = melt_count

                # 본 캐시는 z-block 단위로 채워지므로 이 SV 들의 70 layer 는 모두 valid.
                # (이후 빈 시편 영역 처리 등으로 확장 시 사용)
                valid_layer_mask[block_idx, :] = True

        return {
            "features_seq": features_seq,
            "valid_layer_mask": valid_layer_mask,
            "cad_count_per_layer": cad_count_per_layer,
            "melt_count_per_layer": melt_count_per_layer,
        }

    # ------------------------------------------------------------
    # 그룹별 헬퍼 — z-block (70 layer) 단위로 layer-별 raw 누적
    # ------------------------------------------------------------
    def _per_layer_cad_dscnn_block(
        self, f, block_voxels, l0, l1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """P1 의 핵심 — CAD 2채널 + DSCNN 8채널 + cad_count_per_layer.

        baseline ``_extract_cad_features_block`` / ``_extract_dscnn_features_block``
        의 z-평균 직전 단계 (각 layer 의 SV-patch CAD-pixel 평균) 까지를 그대로 따른다.
        한 layer 에 CAD 픽셀이 0 이면 해당 layer 채널은 0 (가중치도 0 → z-평균 시 무시됨).
        """
        n = len(block_voxels)
        T = l1 - l0
        cad_per_layer = np.zeros((n, common_config.SV_Z_LAYERS, 2), dtype=np.float32)
        dscnn_per_layer = np.zeros((n, common_config.SV_Z_LAYERS, 8), dtype=np.float32)
        cad_count = np.zeros((n, common_config.SV_Z_LAYERS), dtype=np.int32)

        part_ids_ds = f["slices/part_ids"]
        sat_layers = float(common_config.DIST_OVERHANG_SATURATION_LAYERS)
        dscnn_class_ids = [v[0] for v in common_config.DSCNN_FEATURE_MAP.values()]

        # SV 별 픽셀 범위 미리 계산
        ranges = [self.grid.get_pixel_range(int(v[0]), int(v[1])) for v in block_voxels]

        for ti, layer in enumerate(range(l0, l1)):
            part_layer = part_ids_ds[layer]
            cad_mask = part_layer > 0

            # ---- CAD #0: distance_from_edge ----
            if cad_mask.any():
                dist = distance_transform_edt(cad_mask) * common_config.PIXEL_SIZE_MM
                dist = np.minimum(dist, common_config.DIST_EDGE_SATURATION_MM)
                dist_smooth = gaussian_filter(dist.astype(np.float32), sigma=self.sigma_px)
            else:
                dist_smooth = np.zeros_like(part_layer, dtype=np.float32)

            # ---- CAD #1: vertical-column distance_from_overhang ----
            if self._prev_cad_layer is not None:
                overhang = cad_mask & (~self._prev_cad_layer)
                if overhang.any():
                    self._last_overhang_layer[overhang] = float(layer)

            dist_oh_layers = float(layer) - self._last_overhang_layer
            dist_oh_layers = np.minimum(dist_oh_layers, sat_layers).astype(np.float32)
            dist_oh_smooth = gaussian_filter(dist_oh_layers, sigma=self.sigma_px)

            self._prev_cad_layer = cad_mask.copy()

            # ---- DSCNN 8 채널 (CAD 픽셀 있는 layer 만 의미 있음) ----
            seg_smoothed = None
            if cad_mask.any():
                seg_smoothed = []
                for cls_id in dscnn_class_ids:
                    seg_key = f"slices/segmentation_results/{cls_id}"
                    if seg_key in f:
                        seg = f[seg_key][layer].astype(np.float32)
                        seg = gaussian_filter(seg, sigma=self.sigma_px)
                    else:
                        seg = np.zeros((self.grid.image_h, self.grid.image_w), dtype=np.float32)
                    seg_smoothed.append(seg)

            # ---- SV-patch 평균 ----
            for vi in range(n):
                r0, r1, c0, c1 = ranges[vi]
                patch_cad = cad_mask[r0:r1, c0:c1]
                n_cad = int(patch_cad.sum())
                cad_count[vi, ti] = n_cad
                if n_cad <= 0:
                    continue   # 0 으로 두면 가중평균에 무게 0 으로 기여

                cad_per_layer[vi, ti, 0] = dist_smooth[r0:r1, c0:c1][patch_cad].mean()
                cad_per_layer[vi, ti, 1] = dist_oh_smooth[r0:r1, c0:c1][patch_cad].mean()

                if seg_smoothed is not None:
                    for ci, seg in enumerate(seg_smoothed):
                        dscnn_per_layer[vi, ti, ci] = seg[r0:r1, c0:c1][patch_cad].mean()

        return cad_per_layer, dscnn_per_layer, cad_count

    def _per_layer_temporal_block(self, temporal_data: dict, l0: int, l1: int) -> np.ndarray:
        """P3 — z-block 내 70 layer × 7 센서 시계열 (모든 SV 동일).

        키 부재 시 NaN 으로 채움 — baseline 과 동일한 SV-드롭 정책 유지.
        """
        T = common_config.SV_Z_LAYERS
        out = np.full((T, len(common_config.TEMPORAL_FEATURES)), np.nan, dtype=np.float32)
        actual_T = l1 - l0
        for ti, key in enumerate(common_config.TEMPORAL_FEATURES):
            if key in temporal_data:
                vals = temporal_data[key][l0:l1]
                out[:actual_T, ti] = vals
                if actual_T < T:
                    # 마지막 z-block 이 잘려 70 보다 작은 경우 — 유효 부분만 채우고 나머지는 NaN.
                    pass
        return out

    def _per_layer_scan_block(
        self, f, block_voxels, l0, l1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """P2 — return_delay / stripe_boundaries 의 layer 별 SV-patch 평균.

        baseline ``_extract_scan_features_block`` 의 z-평균 직전 단계와 같은 정책:
          - melt 픽셀 1개 이상인 layer 에서 patch 평균 누적
          - melt 픽셀 0 layer 는 0 (NaN 방지) — baseline 의 의도적 처리 유지
        """
        n = len(block_voxels)
        T = common_config.SV_Z_LAYERS
        scan_per_layer = np.zeros((n, T, 2), dtype=np.float32)
        melt_count = np.zeros((n, T), dtype=np.int32)

        kernel_px = max(1, int(round(1.0 / common_config.PIXEL_SIZE_MM)))
        img_shape = (self.grid.image_h, self.grid.image_w)
        ranges = [self.grid.get_pixel_range(int(v[0]), int(v[1])) for v in block_voxels]

        for ti, layer in enumerate(range(l0, l1)):
            key = f"scans/{layer}"
            if key not in f:
                continue
            scans = f[key][...]
            if len(scans) == 0:
                continue

            mt = build_melt_time_map(scans, img_shape, common_config.PIXEL_SIZE_MM)
            rd_map = compute_return_delay_map(mt, kernel_px=kernel_px, sat_s=0.75)
            sb_map = compute_stripe_boundaries_map(mt)

            for vi in range(n):
                r0, r1, c0, c1 = ranges[vi]
                rd_patch = rd_map[r0:r1, c0:c1]
                sb_patch = sb_map[r0:r1, c0:c1]
                rd_valid = ~np.isnan(rd_patch)
                n_valid = int(rd_valid.sum())
                if n_valid <= 0:
                    continue
                scan_per_layer[vi, ti, 0] = rd_patch[rd_valid].mean()
                scan_per_layer[vi, ti, 1] = sb_patch[rd_valid].mean()
                melt_count[vi, ti] = n_valid

        return scan_per_layer, melt_count

    def _advance_overhang_state(self, f, l0: int, l1: int) -> None:
        """voxel 이 없는 z-block 에서도 overhang carry-over 상태를 갱신.

        ``_per_layer_cad_dscnn_block`` 내부에서 prev_cad_layer 를 update 하는 부분만
        layer 단위로 따로 실행. baseline 도 z-block 내 voxel 이 없으면 cad/dscnn 추출
        루프 자체를 skip 하므로 overhang 상태가 reset 되지 않게 동일하게 맞춘다.
        """
        # baseline 은 voxel 이 없는 z-block 에서 prev_cad_layer 도 forward 하지 않는다.
        # 본 캐시도 동일 동작이 필요하므로 여기서는 아무것도 하지 않는다.
        # 함수는 명시적 의도 표시용으로 둠.
        return

    # ------------------------------------------------------------
    # baseline 과 공유되는 유틸 (FeatureExtractor 와 동일 코드)
    # ------------------------------------------------------------
    def _load_temporal(self, f) -> dict:
        data = {}
        for key in common_config.TEMPORAL_FEATURES:
            full_key = f"temporal/{key}"
            if full_key in f:
                data[key] = f[full_key][...]
        return data

    def _load_laser_modules(self, f) -> dict:
        key = "parts/process_parameters/laser_module"
        if key not in f:
            return {}
        lm_data = f[key][...]
        return {
            i: int(lm_data[i])
            for i in range(len(lm_data))
            if not np.isnan(lm_data[i])
        }


# ============================================================
# 빌드 → 캐시 (run.py 가 호출)
# ============================================================
def build_cache(build_ids: list[str], output_path: Path) -> None:
    """5개 빌드의 시퀀스 캐시를 누적해 단일 npz 로 저장.

    baseline ``run_pipeline.merge_all_builds`` 와 동일하게 빌드별 sample_id 오프셋을
    적용해 build 간 sample_id 중복을 방지한다. build_ids 는 ``config.BUILDS`` 순서를
    유지해 baseline ``all_features.npz`` 와 SV 정렬 순서가 같도록 한다.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feats_list, mask_list, cadc_list, meltc_list = [], [], [], []
    sids_list, parts_list, voxels_list, builds_list = [], [], [], []
    targets_acc = {p: [] for p in common_config.TARGET_PROPERTIES}

    sample_offset = 0
    for bid in build_ids:
        hdf5 = str(common_config.hdf5_path(bid))
        print(f"\n{'='*60}\n[features_seq] {bid}\n{'='*60}")
        print(f"  hdf5: {hdf5}")

        grid = SuperVoxelGrid.from_hdf5(hdf5)
        s = grid.summary()
        print(f"  grid: {s['nx']}x{s['ny']}x{s['nz']} = {s['total_voxels']}")

        valid = find_valid_supervoxels(grid, hdf5)
        n = len(valid["sample_ids"])
        print(f"  valid SVs: {n}")
        if n == 0:
            print(f"  no valid SVs in {bid}, skipping")
            continue

        extractor = FeatureSequenceExtractor(grid, hdf5)
        out = extractor.extract_sequences(valid)

        feats_list.append(out["features_seq"])
        mask_list.append(out["valid_layer_mask"])
        cadc_list.append(out["cad_count_per_layer"])
        meltc_list.append(out["melt_count_per_layer"])

        sids_list.append(valid["sample_ids"].astype(np.int64) + sample_offset)
        parts_list.append(valid["part_ids"])
        voxels_list.append(valid["voxel_indices"])
        builds_list.append(np.full(n, list(common_config.BUILDS.keys()).index(bid), dtype=np.int32))

        # 인장 타겟 — baseline pipeline 과 동일하게 sample_id 인덱싱
        with h5py.File(hdf5, "r") as f:
            for prop in common_config.TARGET_PROPERTIES:
                key = f"samples/test_results/{prop}"
                if key in f:
                    all_vals = f[key][...]
                    t = np.array([
                        all_vals[sid] if sid < len(all_vals) else np.nan
                        for sid in valid["sample_ids"]
                    ], dtype=np.float32)
                else:
                    t = np.full(n, np.nan, dtype=np.float32)
                targets_acc[prop].append(t)

        max_sid = int(valid["sample_ids"].max())
        sample_offset += max_sid + 1

    if not feats_list:
        raise RuntimeError("어떤 빌드에서도 valid SV 가 추출되지 않았습니다.")

    features_seq = np.concatenate(feats_list, axis=0)
    valid_layer_mask = np.concatenate(mask_list, axis=0)
    cad_count = np.concatenate(cadc_list, axis=0)
    melt_count = np.concatenate(meltc_list, axis=0)
    sample_ids = np.concatenate(sids_list, axis=0).astype(np.int32)
    part_ids = np.concatenate(parts_list, axis=0)
    voxel_indices = np.concatenate(voxels_list, axis=0)
    build_ids_arr = np.concatenate(builds_list, axis=0)
    targets = {p: np.concatenate(targets_acc[p], axis=0) for p in common_config.TARGET_PROPERTIES}

    print(f"\n[features_seq] saving cache → {output_path}")
    print(f"  N_sv = {len(features_seq)}, shape = {features_seq.shape}")
    np.savez_compressed(
        output_path,
        features_seq=features_seq,
        valid_layer_mask=valid_layer_mask,
        cad_count_per_layer=cad_count,
        melt_count_per_layer=melt_count,
        sample_ids=sample_ids,
        part_ids=part_ids,
        voxel_indices=voxel_indices,
        build_ids=build_ids_arr,
        **{f"target_{k}": v for k, v in targets.items()},
    )


# ============================================================
# 검증 — z-평균이 baseline all_features.npz 와 일치하는지
# ============================================================
def validate_against_baseline(seq_npz: Path, baseline_npz: Path,
                              tol: float = 1e-5) -> dict:
    """채널별 z-평균 vs baseline 21-feat 비교.

    P1 (CAD-가중평균): cad_count_per_layer 가중
    P2 (단순 평균): melt_count_per_layer > 0 layer 만 합산
    P3 (단순 평균): 70 layer 평균
    P4 (broadcast 상수): 어느 layer 값이든 동일 → 평균 = 자기 자신

    baseline all_features.npz 와 SV 정렬이 다를 수 있으니 sample_id+voxel_index 로
    매칭한다 (baseline 도 build 순 concat 이므로 build 인덱스가 일치하면 그대로 비교 가능).
    """
    print(f"[validate] seq:      {seq_npz}")
    print(f"[validate] baseline: {baseline_npz}")
    seq = np.load(seq_npz)
    base = np.load(baseline_npz)

    seq_seq = seq["features_seq"]                # (N, 70, 21)
    seq_cadc = seq["cad_count_per_layer"]        # (N, 70)
    seq_meltc = seq["melt_count_per_layer"]      # (N, 70)
    seq_sids = seq["sample_ids"]
    seq_bids = seq["build_ids"]

    base_feats = base["features"]                # (N', 21)
    base_sids = base["sample_ids"]
    base_bids = base["build_ids"]

    # build 별 같은 순서로 concat 되어 있는지 1차 확인
    if len(seq_seq) != len(base_feats):
        print(f"[validate][warn] N mismatch: seq={len(seq_seq)} vs base={len(base_feats)}")
    if not (np.array_equal(seq_sids[: min(len(seq_sids), len(base_sids))],
                           base_sids[: min(len(seq_sids), len(base_sids))])):
        print("[validate][warn] sample_ids ordering differs — 매칭 실패 가능성")

    N_min = min(len(seq_seq), len(base_feats))
    seq_seq = seq_seq[:N_min]
    seq_cadc = seq_cadc[:N_min]
    seq_meltc = seq_meltc[:N_min]
    base_feats = base_feats[:N_min]

    # ---- 채널별 z-평균 재구성 ----
    recon = np.zeros_like(base_feats, dtype=np.float64)

    # P1: CAD-가중 평균
    cad_w = seq_cadc.astype(np.float64)                          # (N, 70)
    cad_w_sum = cad_w.sum(axis=1)                                # (N,)
    cad_valid = cad_w_sum > 0
    for ch in exp_config.P1_INDICES:
        weighted = (seq_seq[:, :, ch].astype(np.float64) * cad_w).sum(axis=1)
        col = np.zeros(seq_seq.shape[0], dtype=np.float64)
        col[cad_valid] = weighted[cad_valid] / cad_w_sum[cad_valid]
        recon[:, ch] = col

    # P2: melt 픽셀 ≥ 1 layer 만 단순 평균
    melt_layer_valid = (seq_meltc > 0).astype(np.float64)        # (N, 70)
    melt_count = melt_layer_valid.sum(axis=1)                    # (N,)
    melt_ok = melt_count > 0
    for ch in exp_config.P2_INDICES:
        s = (seq_seq[:, :, ch].astype(np.float64) * melt_layer_valid).sum(axis=1)
        col = np.zeros(seq_seq.shape[0], dtype=np.float64)
        col[melt_ok] = s[melt_ok] / melt_count[melt_ok]
        # baseline 은 melt 없는 SV 도 0 으로 채움 → recon 도 0 그대로 유지
        recon[:, ch] = col

    # P3: 70 layer 단순 평균 — z-block 끝 잘림 시 NaN 이 들어 있을 수 있음
    for ch in exp_config.P3_INDICES:
        seq_ch = seq_seq[:, :, ch].astype(np.float64)
        recon[:, ch] = np.nanmean(seq_ch, axis=1)

    # P4: broadcast 상수 — 어느 layer 값이든 동일
    for ch in exp_config.P4_INDICES:
        recon[:, ch] = seq_seq[:, 0, ch].astype(np.float64)

    # ---- 비교 ----
    diff = np.abs(recon - base_feats.astype(np.float64))
    # baseline 의 NaN (drop 대상 SV) 은 비교에서 제외
    nan_mask = np.isnan(base_feats)
    diff[nan_mask] = 0.0

    per_ch = diff.max(axis=0)
    print("\n[validate] per-channel max abs diff (z-mean reconstructed vs baseline):")
    for ch in range(common_config.N_FEATURES):
        flag = "  OK" if per_ch[ch] <= tol else "  FAIL"
        print(f"  ch {ch:>2d}: {per_ch[ch]: .3e} {flag}")

    overall = float(per_ch.max())
    print(f"\n[validate] overall max abs diff = {overall:.3e}  (tol={tol:.0e})")

    return {
        "per_channel_max_abs_diff": per_ch.tolist(),
        "overall_max_abs_diff": overall,
        "tolerance": tol,
        "passed": bool(overall <= tol),
    }


# ============================================================
# CLI — features 빌드 / 검증
# ============================================================
def _cli() -> None:
    parser = argparse.ArgumentParser(description="VPPM-1DCNN feature sequence cache")
    parser.add_argument("--validate", action="store_true",
                        help="기존 캐시를 baseline all_features.npz 와 비교 (z-평균 재구성)")
    parser.add_argument("--builds", nargs="+", default=list(common_config.BUILDS.keys()),
                        help="추출할 빌드 ID")
    parser.add_argument("--out", type=Path, default=exp_config.FEATURES_SEQ_NPZ,
                        help="출력 npz 경로")
    parser.add_argument("--baseline", type=Path, default=exp_config.BASELINE_FEATURES_NPZ,
                        help="검증용 baseline all_features.npz 경로")
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    if args.validate:
        if not args.out.exists():
            raise FileNotFoundError(f"캐시 파일이 없습니다: {args.out}")
        if not args.baseline.exists():
            raise FileNotFoundError(f"baseline 캐시가 없습니다: {args.baseline}")
        validate_against_baseline(args.out, args.baseline, tol=args.tol)
        return

    build_cache(args.builds, args.out)


if __name__ == "__main__":
    _cli()
