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

from ..common import config
from ..common.supervoxel import SuperVoxelGrid
from .scan_features import (
    build_melt_time_map,
    compute_return_delay_map,
    compute_stripe_boundaries_map,
)


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

        # #2 distance_from_overhang: vertical-column 거리용 누적 상태
        #   - prev_cad_layer  : 직전 레이어의 CAD 마스크 (z-block 경계 넘어 carry-over)
        #   - last_overhang_layer : 각 픽셀에서 가장 최근 오버행이 발생한 layer index.
        #     아직 오버행이 검출되지 않은 픽셀은 -inf → distance = +inf → saturate(71).
        self._prev_cad_layer = None
        self._last_overhang_layer = np.full(
            (self.grid.image_h, self.grid.image_w),
            -np.inf,
            dtype=np.float32,
        )

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

                # --- 피처 #20-21: 스캔 경로 기반 (실 구현) ---
                scan_accum = self._extract_scan_features_block(
                    f, block_voxels, l0, l1
                )
                features[block_indices, 19] = scan_accum[:, 0]  # return_delay
                features[block_indices, 20] = scan_accum[:, 1]  # stripe_boundaries

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
        """z-block 내 CAD 피처 (distance_from_edge, distance_from_overhang) 추출.

        피처 #2 (`distance_from_overhang`) 는 Scime et al. 2023 Appendix D Table A2
        정의에 따라 **수직(z-축) column 거리** 로 계산:
          - 오버행 = 분말 위에 새로 출력된 영역 = (current CAD) ∧ ¬(prev CAD)
          - 빌드 첫 레이어는 빌드 플레이트 위에 출력 → overhang 미검출
          - 각 픽셀에서 dist = (현재 layer) − (가장 최근 overhang 발생 layer)
          - 71 layer 이상은 saturate
        상태(`self._prev_cad_layer`, `self._last_overhang_layer`)는 z-block 경계 넘어
        carry-over 되며, `extract_features` 진입 시 1회 초기화.
        """
        n = len(block_indices)
        accum = np.zeros((n, 2), dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)

        part_ids_ds = f["slices/part_ids"]
        sat_layers = float(config.DIST_OVERHANG_SATURATION_LAYERS)

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

            # 피처 2: vertical-column distance from overhang
            # 빌드 첫 레이어(prev = None)는 빌드 플레이트 위 출력 → overhang 미검출
            if self._prev_cad_layer is not None:
                overhang = cad_mask & (~self._prev_cad_layer)
                if overhang.any():
                    self._last_overhang_layer[overhang] = float(layer)

            # dist_oh = 현재 layer − 가장 최근 overhang layer (미검출 픽셀: +inf → saturate)
            dist_oh_layers = float(layer) - self._last_overhang_layer
            dist_oh_layers = np.minimum(dist_oh_layers, sat_layers).astype(np.float32)
            dist_oh_smooth = gaussian_filter(dist_oh_layers, sigma=self.sigma_px)

            self._prev_cad_layer = cad_mask.copy()

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

    def _extract_scan_features_block(self, f, block_voxels, l0, l1):
        """z-block 내 스캔 경로 피처 (return_delay, stripe_boundaries) 추출.

        각 레이어별로 melt-time 맵을 즉시 계산 → 이웃 max-min 필터 / Sobel RMS →
        해당 z-block 내 모든 슈퍼복셀 영역의 평균을 누적. 영구 캐시 없이 메모리만 사용.

        Returns:
            (n_voxels, 2) — [return_delay, stripe_boundaries], NaN 가능 (스캔 데이터 없음).
        """
        n = len(block_voxels)
        accum = np.zeros((n, 2), dtype=np.float64)
        counts = np.zeros(n, dtype=np.int64)

        # 1mm 커널 = round(1mm / pixel_size_mm) — 보통 8 px
        kernel_px = max(1, int(round(1.0 / config.PIXEL_SIZE_MM)))
        img_shape = (self.grid.image_h, self.grid.image_w)

        for layer in range(l0, l1):
            key = f"scans/{layer}"
            if key not in f:
                continue
            scans = f[key][...]
            if len(scans) == 0:
                continue

            mt = build_melt_time_map(scans, img_shape, config.PIXEL_SIZE_MM)
            rd_map = compute_return_delay_map(mt, kernel_px=kernel_px, sat_s=0.75)
            sb_map = compute_stripe_boundaries_map(mt)

            for vi in range(n):
                ix, iy = block_voxels[vi, 0], block_voxels[vi, 1]
                r0, r1, c0, c1 = self.grid.get_pixel_range(ix, iy)
                rd_patch = rd_map[r0:r1, c0:c1]
                sb_patch = sb_map[r0:r1, c0:c1]

                # 슈퍼복셀 영역 안에 melt 된 픽셀이 하나라도 있으면 평균 누적
                rd_valid = ~np.isnan(rd_patch)
                if rd_valid.any():
                    accum[vi, 0] += rd_patch[rd_valid].mean()
                    # sb_map 은 NaN 없이 0 으로 채워짐 — 같은 valid mask 사용
                    accum[vi, 1] += sb_patch[rd_valid].mean()
                    counts[vi] += 1

        # 스캔 데이터가 전혀 없는 슈퍼복셀 → 0 ("활동 없음" 의미).
        # NaN 으로 두면 build_dataset 의 NaN 마스크가 해당 슈퍼복셀 전체를 드롭함.
        out = np.zeros((n, 2), dtype=np.float32)
        ok = counts > 0
        out[ok] = (accum[ok] / counts[ok, np.newaxis]).astype(np.float32)
        return out
