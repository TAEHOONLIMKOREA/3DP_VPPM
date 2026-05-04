"""[new_v2] 빌드용 — sample_ids 없이 part_ids 만 보는 SV 유효성 판정.

기존 `Sources.vppm.common.supervoxel.find_valid_supervoxels` 는
`sample_ids > 0` overlap 비율을 요구한다. new_v2 빌드는 sample_ids 가 사실상
모두 0 이라 그 함수를 그대로 쓰면 N=0 이 된다.

본 모듈은 part overlap 만 보는 동치 함수를 제공:
  유효 조건: SV xy 영역 안에서 part_ids > 0 픽셀 비율 ≥ NEW_V2_CAD_OVERLAP_THRESHOLD
            (z-block 의 mid_layer 기준)

반환 dict 의 `sample_ids` 는 dominant **part_id** 로 채운다 — 기존 캐시 파이프라인이
"sample_ids" 라는 이름을 sv 메타로 저장하기 때문에 그 컨벤션은 유지하되, 의미는
new_v2 평가에서 part_id 로 쓴다 (다운스트림 evaluator 가 알고 있음).
"""
from __future__ import annotations

import h5py
import numpy as np

from ..common import config
from ..common.supervoxel import SuperVoxelGrid


def find_valid_supervoxels_partbased(
    grid: SuperVoxelGrid,
    hdf5_path: str,
    min_part_overlap: float = config.NEW_V2_CAD_OVERLAP_THRESHOLD,
) -> dict:
    """part overlap 기반 유효 SV 식별.

    Returns:
        dict with keys:
            voxel_indices: (N, 3) int32 — (ix, iy, iz)
            sample_ids:    (N,)   int32 — dominant part_id (캐시 호환용 alias)
            part_ids:      (N,)   int32 — dominant part_id (실 의미)
            cad_ratio:     (N,)   float32 — part_ids>0 픽셀 비율
    """
    voxel_list = []
    part_list = []
    cad_ratio_list = []

    sv_area = grid.sv_xy_pixels * grid.sv_xy_pixels

    with h5py.File(hdf5_path, "r") as f:
        part_ids_ds = f["slices/part_ids"]

        for iz in range(grid.nz):
            l0, l1 = grid.get_layer_range(iz)
            mid_layer = (l0 + l1) // 2
            part_layer = part_ids_ds[mid_layer]

            for ix, iy in grid.iter_xy():
                r0, r1, c0, c1 = grid.get_pixel_range(ix, iy)
                part_patch = part_layer[r0:r1, c0:c1]

                cad_mask = part_patch > 0
                cad_count = int(cad_mask.sum())
                ratio = cad_count / sv_area
                if ratio < min_part_overlap:
                    continue

                part_vals = part_patch[cad_mask]
                unique_p, counts_p = np.unique(part_vals, return_counts=True)
                dominant_part = int(unique_p[counts_p.argmax()])

                voxel_list.append((ix, iy, iz))
                part_list.append(dominant_part)
                cad_ratio_list.append(ratio)

    if not voxel_list:
        return {
            "voxel_indices": np.zeros((0, 3), dtype=np.int32),
            "sample_ids": np.zeros((0,), dtype=np.int32),
            "part_ids": np.zeros((0,), dtype=np.int32),
            "cad_ratio": np.zeros((0,), dtype=np.float32),
        }

    indices = np.array(voxel_list, dtype=np.int32)
    parts = np.array(part_list, dtype=np.int32)
    return {
        "voxel_indices": indices,
        "sample_ids": parts.copy(),                 # 캐시 호환 alias (실 의미는 part_id)
        "part_ids": parts,
        "cad_ratio": np.array(cad_ratio_list, dtype=np.float32),
    }
