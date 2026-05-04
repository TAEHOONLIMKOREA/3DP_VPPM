"""VPPM-1DCNN 데이터셋 — (N, 21, 70) 시퀀스 + 채널별 [-1, 1] 정규화.

baseline ``common.dataset`` 과의 차이:
  - 입력이 (N, 70, 21) → Conv1d 형식 (N, 21, 70) 로 재배치
  - 정규화 통계는 (N × 70) 을 채널별로 풀어서 산출
  - NaN SV 드롭 정책은 baseline 과 동일 (시퀀스에 NaN 있으면 SV 단위 드롭)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import config as common_config
from ..common.dataset import normalize


def load_features_seq(path: Path) -> dict:
    """features_seq.npz 로드 → numpy dict.

    Returns:
        features_seq         : (N, 70, 21) float32
        valid_layer_mask     : (N, 70) bool
        cad_count_per_layer  : (N, 70) int32
        melt_count_per_layer : (N, 70) int32
        sample_ids           : (N,)  int32
        build_ids            : (N,)  int32
        targets              : {prop: (N,) float32}
    """
    data = np.load(path)
    targets = {
        p: data[f"target_{p}"]
        for p in common_config.TARGET_PROPERTIES
        if f"target_{p}" in data.files
    }
    return {
        "features_seq": data["features_seq"],
        "valid_layer_mask": data["valid_layer_mask"],
        "cad_count_per_layer": data["cad_count_per_layer"],
        "melt_count_per_layer": data["melt_count_per_layer"],
        "sample_ids": data["sample_ids"],
        "build_ids": data["build_ids"] if "build_ids" in data.files else None,
        "voxel_indices": data["voxel_indices"] if "voxel_indices" in data.files else None,
        "targets": targets,
    }


def build_normalized_dataset_seq(raw: dict) -> dict:
    """채널별 [-1, 1] 정규화 + NaN SV 드롭 + 타겟 정규화.

    baseline ``build_dataset`` 과 동일한 valid_mask 규칙 적용:
      - UTS NaN / UTS < 50 MPa 제외
      - 모든 타겟 NaN 아님
      - 시퀀스에 NaN 없음 (baseline 의 ``~np.isnan(features).any(axis=1)`` 대응)
    """
    seq = raw["features_seq"]                      # (N, 70, 21)
    sids = raw["sample_ids"]
    bids = raw["build_ids"]
    tgts = raw["targets"]
    cadc = raw["cad_count_per_layer"]
    meltc = raw["melt_count_per_layer"]
    vmask = raw["valid_layer_mask"]

    # ---- valid mask ----
    uts = tgts.get("ultimate_tensile_strength", np.zeros(len(seq)))
    valid = ~np.isnan(uts) & (uts >= 50.0)
    for prop in common_config.TARGET_PROPERTIES:
        if prop in tgts:
            valid &= ~np.isnan(tgts[prop])
    # SV 시퀀스 어딘가에 NaN 이 있으면 드롭 (baseline 과 동일 정책)
    valid &= ~np.isnan(seq).any(axis=(1, 2))

    seq = seq[valid]
    sids = sids[valid]
    bids = bids[valid] if bids is not None else None
    tgts = {k: v[valid] for k, v in tgts.items()}
    cadc = cadc[valid]
    meltc = meltc[valid]
    vmask = vmask[valid]

    # ---- 채널별 [-1, 1] min-max 정규화 ----
    # seq: (N, 70, 21) → 채널별 min/max 는 (N × 70) 을 풀어서 산출
    f_min = seq.reshape(-1, seq.shape[-1]).min(axis=0)   # (21,)
    f_max = seq.reshape(-1, seq.shape[-1]).max(axis=0)   # (21,)
    seq_norm = normalize(seq, f_min, f_max).astype(np.float32)

    # ---- (N, 70, 21) → (N, 21, 70) for Conv1d ----
    seq_norm = np.transpose(seq_norm, (0, 2, 1))         # (N, 21, 70)
    seq_raw = np.transpose(seq, (0, 2, 1)).astype(np.float32)

    # ---- 타겟 정규화 ----
    tgt_norm = {}
    t_min, t_max = {}, {}
    for prop in common_config.TARGET_PROPERTIES:
        if prop in tgts:
            t_min[prop] = float(tgts[prop].min())
            t_max[prop] = float(tgts[prop].max())
            tgt_norm[prop] = normalize(tgts[prop], t_min[prop], t_max[prop]).astype(np.float32)

    norm_params = {
        "feature_min": f_min.tolist(),
        "feature_max": f_max.tolist(),
        "target_min": t_min,
        "target_max": t_max,
    }

    return {
        "features": seq_norm,                # (N, 21, 70) float32 normalized
        "features_raw": seq_raw,             # (N, 21, 70) float32 raw
        "sample_ids": sids,
        "build_ids": bids,
        "targets": tgt_norm,
        "targets_raw": tgts,
        "cad_count_per_layer": cadc,
        "melt_count_per_layer": meltc,
        "valid_layer_mask": vmask,
        "norm_params": norm_params,
    }


class VPPM1DCNNDataset(Dataset):
    """(features (21, 70), target) 한 쌍 반환."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        # features: (N, 21, 70) float32 normalized
        # targets:  (N,)       float32 normalized
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float().unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
