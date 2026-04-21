"""
Phase 3: 데이터셋 구성 — 정규화, 교차검증 분할, PyTorch Dataset
논문 Section 2.10-2.11
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from pathlib import Path

from . import config


class VPPMDataset(Dataset):
    """슈퍼복셀 피처/타겟 데이터셋"""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float().unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def normalize(data: np.ndarray, d_min: np.ndarray, d_max: np.ndarray) -> np.ndarray:
    """Zero-center normalization to [-1, 1]"""
    eps = 1e-8
    return 2.0 * (data - d_min) / (d_max - d_min + eps) - 1.0


def denormalize(data: np.ndarray, d_min: np.ndarray, d_max: np.ndarray) -> np.ndarray:
    """[-1, 1] 정규화 역변환"""
    return (data + 1.0) / 2.0 * (d_max - d_min) + d_min


def build_dataset(all_features: np.ndarray, all_sample_ids: np.ndarray,
                  all_targets: dict, build_ids: np.ndarray = None):
    """전체 데이터셋 구축

    Args:
        all_features: (N, 21) 피처 배열
        all_sample_ids: (N,) 샘플 ID 배열
        all_targets: {property_name: (N,) 타겟 배열}
        build_ids: (N,) 빌드 ID (시각화용, optional)

    Returns:
        dict with normalized data and normalization params
    """
    # NaN 타겟 제거 (UTS < 50 MPa 제외 — 논문 Section 2.9)
    uts = all_targets.get("ultimate_tensile_strength", np.zeros(len(all_features)))
    valid_mask = ~np.isnan(uts) & (uts >= 50.0)

    # 모든 타겟이 유효한 샘플만
    for prop in config.TARGET_PROPERTIES:
        if prop in all_targets:
            valid_mask &= ~np.isnan(all_targets[prop])

    # NaN 피처 제거
    valid_mask &= ~np.isnan(all_features).any(axis=1)

    features = all_features[valid_mask]
    sample_ids = all_sample_ids[valid_mask]
    targets = {k: v[valid_mask] for k, v in all_targets.items()}

    # 정규화 파라미터 계산
    f_min = features.min(axis=0)
    f_max = features.max(axis=0)
    features_norm = normalize(features, f_min, f_max)

    targets_norm = {}
    t_min, t_max = {}, {}
    for prop in config.TARGET_PROPERTIES:
        if prop in targets:
            t_min[prop] = targets[prop].min()
            t_max[prop] = targets[prop].max()
            targets_norm[prop] = normalize(targets[prop],
                                           t_min[prop], t_max[prop])

    norm_params = {
        "feature_min": f_min.tolist(),
        "feature_max": f_max.tolist(),
        "target_min": {k: float(v) for k, v in t_min.items()},
        "target_max": {k: float(v) for k, v in t_max.items()},
    }

    return {
        "features": features_norm,
        "features_raw": features,
        "sample_ids": sample_ids,
        "targets": targets_norm,
        "targets_raw": targets,
        "norm_params": norm_params,
        "build_ids": build_ids[valid_mask] if build_ids is not None else None,
    }


def create_cv_splits(sample_ids: np.ndarray, n_splits: int = config.N_FOLDS,
                     seed: int = config.RANDOM_SEED):
    """샘플 단위 K-Fold 분할

    같은 시편의 모든 슈퍼복셀은 같은 fold에 배정

    Returns:
        list of (train_mask, val_mask) boolean arrays
    """
    unique_samples = np.unique(sample_ids)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []

    for train_idx, val_idx in kf.split(unique_samples):
        train_samples = set(unique_samples[train_idx].tolist())
        val_samples = set(unique_samples[val_idx].tolist())
        train_mask = np.array([s in train_samples for s in sample_ids])
        val_mask = np.array([s in val_samples for s in sample_ids])
        splits.append((train_mask, val_mask))

    return splits


def save_norm_params(norm_params: dict, path: Path):
    """정규화 파라미터를 JSON으로 저장"""
    with open(path, "w") as f:
        json.dump(norm_params, f, indent=2)


def load_norm_params(path: Path) -> dict:
    """정규화 파라미터 로드"""
    with open(path) as f:
        return json.load(f)
