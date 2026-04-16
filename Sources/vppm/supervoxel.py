"""
Phase 1: 슈퍼복셀 그리드 구축
논문 Section 2.10 — 빌드 볼륨을 1.0×1.0×3.5mm 직육면체 격자로 분할
"""
import numpy as np
import h5py
from . import config


class SuperVoxelGrid:
    """빌드 볼륨을 고정 크기 슈퍼복셀 격자로 분할"""

    def __init__(self, num_layers: int = None, image_shape: tuple = None):
        self.pixel_size_mm = config.PIXEL_SIZE_MM
        self.sv_xy_mm = config.SV_XY_MM
        self.sv_z_layers = config.SV_Z_LAYERS
        self.layer_thickness_mm = config.LAYER_THICKNESS_MM

        # 슈퍼복셀의 픽셀 단위 크기 (정수로 반올림)
        self.sv_xy_pixels = int(round(self.sv_xy_mm / self.pixel_size_mm))

        # 이미지/레이어 크기 (setup_from_hdf5 또는 직접 설정)
        self.image_h = image_shape[0] if image_shape else config.IMAGE_PIXELS
        self.image_w = image_shape[1] if image_shape else config.IMAGE_PIXELS
        self.num_layers = num_layers

        # 그리드 크기 계산
        self.nx = self.image_w // self.sv_xy_pixels
        self.ny = self.image_h // self.sv_xy_pixels
        self.nz = (num_layers // self.sv_z_layers) if num_layers else 0

    @classmethod
    def from_hdf5(cls, hdf5_path: str) -> "SuperVoxelGrid":
        """HDF5 파일에서 좌표 정보를 읽어 그리드 생성"""
        with h5py.File(hdf5_path, "r") as f:
            cam_shape = f["slices/camera_data/visible/0"].shape
            num_layers = cam_shape[0]
            image_shape = (cam_shape[1], cam_shape[2])
        grid = cls(num_layers=num_layers, image_shape=image_shape)
        return grid

    def get_pixel_range(self, ix: int, iy: int) -> tuple:
        """슈퍼복셀 (ix, iy)의 픽셀 범위 반환

        Returns:
            (row_start, row_end, col_start, col_end)
        """
        c0 = ix * self.sv_xy_pixels
        c1 = min(c0 + self.sv_xy_pixels, self.image_w)
        r0 = iy * self.sv_xy_pixels
        r1 = min(r0 + self.sv_xy_pixels, self.image_h)
        return r0, r1, c0, c1

    def get_layer_range(self, iz: int) -> tuple:
        """슈퍼복셀 iz의 레이어 범위 반환

        Returns:
            (layer_start, layer_end)
        """
        l0 = iz * self.sv_z_layers
        l1 = min(l0 + self.sv_z_layers, self.num_layers)
        return l0, l1

    def get_z_center_mm(self, iz: int) -> float:
        """슈퍼복셀 iz의 z 방향 중심 높이 (mm)"""
        l0, l1 = self.get_layer_range(iz)
        center_layer = (l0 + l1) / 2.0
        return center_layer * self.layer_thickness_mm

    def iter_xy(self):
        """모든 (ix, iy) 인덱스를 순회"""
        for iy in range(self.ny):
            for ix in range(self.nx):
                yield ix, iy

    def summary(self) -> dict:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "total_voxels": self.nx * self.ny * self.nz,
            "sv_xy_pixels": self.sv_xy_pixels,
            "sv_z_layers": self.sv_z_layers,
            "pixel_size_mm": self.pixel_size_mm,
            "image_shape": (self.image_h, self.image_w),
            "num_layers": self.num_layers,
        }


def find_valid_supervoxels(grid: SuperVoxelGrid, hdf5_path: str,
                           min_sample_overlap: float = config.SAMPLE_OVERLAP_THRESHOLD):
    """학습에 사용할 유효 슈퍼복셀 식별

    유효 조건:
    1. CAD 형상(part_ids > 0)과 교차
    2. SS-J3 게이지 섹션(sample_ids > 0)과 min_sample_overlap 이상 교차

    Returns:
        dict with keys:
            voxel_indices: (N, 3) — (ix, iy, iz)
            sample_ids: (N,) — 대표 sample_id
            part_ids: (N,) — 대표 part_id
            cad_ratio: (N,) — CAD 교차 비율
    """
    voxel_list = []
    sample_list = []
    part_list = []
    cad_ratio_list = []

    sv_area = grid.sv_xy_pixels * grid.sv_xy_pixels

    with h5py.File(hdf5_path, "r") as f:
        part_ids_ds = f["slices/part_ids"]
        sample_ids_ds = f["slices/sample_ids"]

        for iz in range(grid.nz):
            l0, l1 = grid.get_layer_range(iz)
            # 이 z-block의 중간 레이어에서 sample/part ID 확인 (메모리 절약)
            mid_layer = (l0 + l1) // 2
            part_layer = part_ids_ds[mid_layer]
            sample_layer = sample_ids_ds[mid_layer]

            for ix, iy in grid.iter_xy():
                r0, r1, c0, c1 = grid.get_pixel_range(ix, iy)
                part_patch = part_layer[r0:r1, c0:c1]
                sample_patch = sample_layer[r0:r1, c0:c1]

                # CAD 교차 확인
                cad_mask = part_patch > 0
                cad_count = cad_mask.sum()
                if cad_count == 0:
                    continue

                # Sample 교차 확인
                sample_mask = sample_patch > 0
                sample_count = sample_mask.sum()
                overlap_ratio = sample_count / sv_area

                if overlap_ratio < min_sample_overlap:
                    continue

                # 대표 sample_id: 가장 많은 픽셀을 차지하는 ID
                sample_vals = sample_patch[sample_mask]
                unique, counts = np.unique(sample_vals, return_counts=True)
                dominant_sample = unique[counts.argmax()]

                part_vals = part_patch[cad_mask]
                unique_p, counts_p = np.unique(part_vals, return_counts=True)
                dominant_part = unique_p[counts_p.argmax()]

                voxel_list.append((ix, iy, iz))
                sample_list.append(int(dominant_sample))
                part_list.append(int(dominant_part))
                cad_ratio_list.append(cad_count / sv_area)

    return {
        "voxel_indices": np.array(voxel_list, dtype=np.int32),
        "sample_ids": np.array(sample_list, dtype=np.int32),
        "part_ids": np.array(part_list, dtype=np.int32),
        "cad_ratio": np.array(cad_ratio_list, dtype=np.float32),
    }
