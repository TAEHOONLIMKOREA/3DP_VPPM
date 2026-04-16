"""
Phase 2: 21개 피처 엔지니어링
논문 Section 2.10 & Appendix D

피처 구성:
  #1-3   : CAD 기반 (distance_from_edge, distance_from_overhang, build_height)
  #4-11  : DSCNN 세그멘테이션 비율 (8클래스)
  #12-18 : 프린터 로그 센서 (7개)
  #19-21 : 레이저 스캔 경로 (laser_module, return_delay, stripe_boundaries)
"""
import numpy as np
import h5py
from scipy.ndimage import distance_transform_edt, gaussian_filter
from tqdm import tqdm

from .. import config
from ..supervoxel import SuperVoxelGrid


FEATURE_NAMES = [
    "distance_from_edge",        # 1
    "distance_from_overhang",    # 2
    "build_height",              # 3
    "seg_powder",                # 4
    "seg_printed",               # 5
    "seg_recoater_streaking",    # 6
    "seg_edge_swelling",         # 7
    "seg_debris",                # 8
    "seg_super_elevation",       # 9
    "seg_soot",                  # 10
    "seg_excessive_melting",     # 11
    "layer_print_time",          # 12
    "top_gas_flow_rate",         # 13
    "bottom_gas_flow_rate",      # 14
    "module_oxygen",             # 15
    "build_plate_temperature",   # 16
    "bottom_flow_temperature",   # 17
    "actual_ventilator_flow_rate",  # 18
    "laser_module",              # 19
    "laser_return_delay",        # 20
    "laser_stripe_boundaries",   # 21
]


class FeatureExtractor:
    """슈퍼복셀 단위로 21개 피처를 추출"""

    def __init__(self, grid: SuperVoxelGrid, hdf5_path: str):
        self.grid = grid
        self.hdf5_path = hdf5_path
        self.sigma_px = config.GAUSSIAN_STD_PIXELS

    def extract_features(self, valid_voxels: dict) -> np.ndarray:
        """유효 슈퍼복셀에 대해 21개 피처 추출

        Args:
            valid_voxels: find_valid_supervoxels()의 반환값

        Returns:
            (N_voxels, 21) 피처 배열
        """
        indices = valid_voxels["voxel_indices"]
        n_voxels = len(indices)
        features = np.full((n_voxels, 21), np.nan, dtype=np.float32)

        with h5py.File(self.hdf5_path, "r") as f:
            # 시간적 데이터 미리 로드 (작음)
            temporal_data = self._load_temporal(f)
            # 파트별 레이저 모듈 정보
            laser_modules = self._load_laser_modules(f)

            # z-block 단위로 처리
            for iz in tqdm(range(self.grid.nz), desc="z-blocks"):
                l0, l1 = self.grid.get_layer_range(iz)
                voxel_mask = indices[:, 2] == iz
                if not voxel_mask.any():
                    continue

                block_indices = np.where(voxel_mask)[0]
                block_voxels = indices[voxel_mask]

                # --- CAD 피처 (#1-2): 레이어별 처리 후 z평균 ---
                cad_accum = self._extract_cad_features_block(
                    f, block_voxels, block_indices, l0, l1,
                    valid_voxels["part_ids"][voxel_mask]
                )
                features[block_indices, 0] = cad_accum[:, 0]  # dist_edge
                features[block_indices, 1] = cad_accum[:, 1]  # dist_overhang

                # --- 피처 #3: Build height ---
                features[block_indices, 2] = self.grid.get_z_center_mm(iz)

                # --- DSCNN 피처 (#4-11) ---
                dscnn_accum = self._extract_dscnn_features_block(
                    f, block_voxels, block_indices, l0, l1
                )
                features[block_indices, 3:11] = dscnn_accum

                # --- Temporal 피처 (#12-18) ---
                for ti, key in enumerate(config.TEMPORAL_FEATURES):
                    if key in temporal_data:
                        vals = temporal_data[key][l0:l1]
                        features[block_indices, 11 + ti] = np.mean(vals)

                # --- 피처 #19: Laser module ---
                part_ids_block = valid_voxels["part_ids"][voxel_mask]
                for i, pidx in enumerate(block_indices):
                    pid = part_ids_block[i - block_indices[0]] if i > block_indices[0] else valid_voxels["part_ids"][pidx]
                    pid = valid_voxels["part_ids"][pidx]
                    if pid in laser_modules:
                        lm = laser_modules[pid]
                        features[pidx, 18] = 0.0 if lm == 1 else 1.0

                # --- 피처 #20-21: 스캔 경로 기반 (단순화 버전) ---
                features[block_indices, 19] = 0.0  # placeholder
                features[block_indices, 20] = 0.0  # placeholder

        return features

    def _load_temporal(self, f: h5py.File) -> dict:
        """시간적 센서 데이터를 한번에 로드 (메모리 작음)"""
        data = {}
        for key in config.TEMPORAL_FEATURES:
            full_key = f"temporal/{key}"
            if full_key in f:
                data[key] = f[full_key][...]
        return data

    def _load_laser_modules(self, f: h5py.File) -> dict:
        """파트별 레이저 모듈 정보 로드"""
        key = "parts/process_parameters/laser_module"
        if key not in f:
            return {}
        lm_data = f[key][...]
        return {i: int(lm_data[i]) for i in range(len(lm_data)) if not np.isnan(lm_data[i])}

    def _extract_cad_features_block(self, f, block_voxels, block_indices,
                                     l0, l1, part_ids_arr):
        """z-block 내 CAD 피처 (distance_from_edge, distance_from_overhang) 추출"""
        n = len(block_indices)
        accum = np.zeros((n, 2), dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)

        part_ids_ds = f["slices/part_ids"]
        prev_cad = None

        for layer in range(l0, l1):
            part_layer = part_ids_ds[layer]
            cad_mask = part_layer > 0

            # 피처 1: distance from edge
            if cad_mask.any():
                dist = distance_transform_edt(cad_mask) * config.PIXEL_SIZE_MM
                dist = np.minimum(dist, config.DIST_EDGE_SATURATION_MM)
                dist_smooth = gaussian_filter(dist.astype(np.float32), sigma=self.sigma_px)
            else:
                dist_smooth = np.zeros_like(part_layer, dtype=np.float32)

            # 피처 2: distance from overhang (simplified)
            # 오버행 = 현재 레이어에 CAD 있지만 바로 아래에는 없는 영역
            if prev_cad is not None:
                overhang = cad_mask & (~prev_cad)
                if overhang.any():
                    dist_oh = distance_transform_edt(~overhang).astype(np.float32)
                    dist_oh = np.minimum(dist_oh, config.DIST_OVERHANG_SATURATION_LAYERS)
                    dist_oh_smooth = gaussian_filter(dist_oh, sigma=self.sigma_px)
                else:
                    dist_oh_smooth = np.full_like(part_layer, config.DIST_OVERHANG_SATURATION_LAYERS, dtype=np.float32)
            else:
                dist_oh_smooth = np.full_like(part_layer, config.DIST_OVERHANG_SATURATION_LAYERS, dtype=np.float32)

            prev_cad = cad_mask.copy()

            # 각 슈퍼복셀 영역에서 평균
            for i, vi in enumerate(range(n)):
                ix, iy = block_voxels[vi, 0], block_voxels[vi, 1]
                r0, r1, c0, c1 = self.grid.get_pixel_range(ix, iy)
                patch_cad = cad_mask[r0:r1, c0:c1]
                n_cad = patch_cad.sum()
                if n_cad > 0:
                    accum[vi, 0] += dist_smooth[r0:r1, c0:c1][patch_cad].mean() * n_cad
                    accum[vi, 1] += dist_oh_smooth[r0:r1, c0:c1][patch_cad].mean() * n_cad
                    counts[vi] += n_cad

        # z-방향 가중 평균
        valid = counts > 0
        accum[valid] /= counts[valid, np.newaxis]
        return accum.astype(np.float32)

    def _extract_dscnn_features_block(self, f, block_voxels, block_indices, l0, l1):
        """z-block 내 DSCNN 피처 8개 추출"""
        n = len(block_indices)
        accum = np.zeros((n, 8), dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)

        part_ids_ds = f["slices/part_ids"]
        dscnn_class_ids = [v[0] for v in config.DSCNN_FEATURE_MAP.values()]

        for layer in range(l0, l1):
            part_layer = part_ids_ds[layer]
            cad_mask = part_layer > 0

            if not cad_mask.any():
                continue

            # 8개 DSCNN 클래스 로드 및 가우시안 블러
            seg_smoothed = []
            for cls_id in dscnn_class_ids:
                seg_key = f"slices/segmentation_results/{cls_id}"
                if seg_key in f:
                    seg = f[seg_key][layer].astype(np.float32)
                    seg = gaussian_filter(seg, sigma=self.sigma_px)
                else:
                    seg = np.zeros((self.grid.image_h, self.grid.image_w), dtype=np.float32)
                seg_smoothed.append(seg)

            # 각 슈퍼복셀 영역에서 CAD 내 평균
            for vi in range(n):
                ix, iy = block_voxels[vi, 0], block_voxels[vi, 1]
                r0, r1, c0, c1 = self.grid.get_pixel_range(ix, iy)
                patch_cad = cad_mask[r0:r1, c0:c1]
                n_cad = patch_cad.sum()
                if n_cad > 0:
                    for ci, seg in enumerate(seg_smoothed):
                        accum[vi, ci] += seg[r0:r1, c0:c1][patch_cad].mean() * n_cad
                    counts[vi] += n_cad

        valid = counts > 0
        accum[valid] /= counts[valid, np.newaxis]
        return accum.astype(np.float32)
