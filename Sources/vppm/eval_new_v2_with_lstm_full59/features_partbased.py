"""[new_v2] 빌드용 — part 기반 21-feat 추출.

기존 `Sources.vppm.baseline.features.FeatureExtractor` 의 `extract_features` 를
직접 재사용 가능 (CAD/DSCNN/Sensor/Scan 모든 추출 로직이 part_ids 만 본다).
다만 **#19 laser_module** 만 new_v2 에는 `parts/process_parameters/laser_module`
키가 없어 default 0 으로 채워야 한다.

valid_voxels 는 supervoxel_partbased.find_valid_supervoxels_partbased 의 반환값을 그대로 사용.

반환:
    features: (N_sv, 21) float32 — features[:, 18] (laser_module) 는 항상 0.0
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..baseline.features import FeatureExtractor, FEATURE_NAMES
from ..common import config
from ..common.supervoxel import SuperVoxelGrid
from .supervoxel_partbased import find_valid_supervoxels_partbased


def extract_features_new_v2(
    hdf5_path: str | Path | None = None,
    valid_voxels: dict | None = None,
) -> tuple[np.ndarray, dict, SuperVoxelGrid]:
    """new_v2 빌드 단일 파일에 대해 21-feat 추출.

    laser_module(#19) 은 new_v2 에 키 부재 → default 0 (single-laser interpretation).
    다른 20개 feature 는 baseline 추출 그대로.

    Returns:
        features:     (N_sv, 21) float32
        valid_voxels: supervoxel_partbased.find_valid_supervoxels_partbased 반환값
        grid:         SuperVoxelGrid
    """
    hdf5 = str(hdf5_path) if hdf5_path is not None else str(config.new_v2_hdf5_path())

    grid = SuperVoxelGrid.from_hdf5(hdf5)
    if valid_voxels is None:
        valid_voxels = find_valid_supervoxels_partbased(grid, hdf5)

    n = len(valid_voxels["voxel_indices"])
    print(f"[features_new_v2] grid: nx={grid.nx} ny={grid.ny} nz={grid.nz}, valid SVs={n}")

    extractor = FeatureExtractor(grid, hdf5)
    features = extractor.extract_features(valid_voxels)

    # laser_module: extract_features 가 _load_laser_modules() 에서 빈 dict 반환 →
    # features[:, 18] 은 NaN 으로 남음. dataset.py 의 NaN drop 을 회피하려면 0 으로 채움.
    if np.isnan(features[:, 18]).all():
        features[:, 18] = 0.0
        print("[features_new_v2] laser_module(#19) 키 부재 → 모든 SV 에 0.0 채움")

    return features.astype(np.float32), valid_voxels, grid


def save_features(
    features: np.ndarray,
    valid_voxels: dict,
    targets: dict,
    out_path: Path,
) -> Path:
    """new_v2 21-feat + part_ids + GT 를 npz 로 저장.

    Args:
        features:     (N, 21) float32
        valid_voxels: supervoxel_partbased.find_valid_supervoxels_partbased 반환
        targets:      {prop: (N_parts,) GT 배열} — parts 인덱스 정렬 (id 0 reserved)
        out_path:     저장 경로
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        features=features.astype(np.float32),
        voxel_indices=valid_voxels["voxel_indices"].astype(np.int32),
        part_ids=valid_voxels["part_ids"].astype(np.int32),
        sample_ids=valid_voxels["sample_ids"].astype(np.int32),
        cad_ratio=valid_voxels["cad_ratio"].astype(np.float32),
        feature_names=np.array(FEATURE_NAMES, dtype="S"),
        # GT (parts/test_results) — 21-feat 과 무관, 평가 시 사용
        target_yield_strength=targets.get("yield_strength", np.zeros(0)).astype(np.float32),
        target_ultimate_tensile_strength=targets.get("ultimate_tensile_strength", np.zeros(0)).astype(np.float32),
        target_total_elongation=targets.get("total_elongation", np.zeros(0)).astype(np.float32),
    )
    print(f"[features_new_v2] saved {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path
