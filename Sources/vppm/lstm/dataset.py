"""
Phase L4 — VPPM-LSTM 결합 Dataset.

기존 all_features.npz (21-feat, sample_ids, targets) 와
stacks_all.h5 (per-supervoxel image sequence) 를 동일 row index 로 묶는다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import config


def load_aligned_arrays(
    features_npz: Path | str | None = None,
    stacks_h5: Path | str | None = None,
) -> dict:
    """all_features.npz 와 stacks_all.h5 를 로드해 정렬 검증.

    Returns dict containing numpy arrays for 21-feat / targets / sample_ids /
    build_ids and the h5py path (lazy 접근).
    """
    features_npz = Path(features_npz or config.FEATURES_DIR / "all_features.npz")
    stacks_h5 = Path(stacks_h5 or Path(config.LSTM_CACHE_DIR) / "stacks_all.h5")
    if not features_npz.exists():
        raise FileNotFoundError(f"{features_npz} 없음")
    if not stacks_h5.exists():
        raise FileNotFoundError(
            f"{stacks_h5} 없음 — 먼저 lstm/image_stack.py::build_stacks_cache() 를 실행하세요."
        )

    npz = np.load(features_npz)
    features = npz["features"].astype(np.float32)
    sample_ids = npz["sample_ids"].astype(np.int64)
    build_ids = npz["build_ids"].astype(np.int32) if "build_ids" in npz.files else None
    targets = {
        p: npz[f"target_{p}"].astype(np.float32)
        for p in config.TARGET_PROPERTIES
        if f"target_{p}" in npz.files
    }

    with h5py.File(stacks_h5, "r") as h5:
        n_stack = h5["stacks"].shape[0]
        channels = h5.attrs.get("channels", "?")
        patch_px = int(h5.attrs.get("patch_px", -1))
        n_channels = int(h5.attrs.get("n_channels", -1))
        T = int(h5.attrs.get("T", -1))

    if n_stack != len(features):
        raise AssertionError(
            f"행 수 불일치: stacks={n_stack}, features={len(features)} — "
            f"features.npz 와 stacks_all.h5 가 같은 supervoxel 순서로 생성되었는지 확인."
        )

    return {
        "features": features,
        "sample_ids": sample_ids,
        "build_ids": build_ids,
        "targets": targets,
        "stacks_h5": str(stacks_h5),
        "n": n_stack,
        "T": T,
        "C": n_channels,
        "patch_px": patch_px,
        "channels_name": channels,
    }


def build_valid_mask(features: np.ndarray, targets: dict) -> np.ndarray:
    """기존 dataset.py 와 동일 규칙: UTS < 50 제거, NaN 제거."""
    uts = targets.get("ultimate_tensile_strength", np.zeros(len(features)))
    valid = ~np.isnan(uts) & (uts >= 50.0)
    for p in config.TARGET_PROPERTIES:
        if p in targets:
            valid &= ~np.isnan(targets[p])
    valid &= ~np.isnan(features).any(axis=1)
    return valid


class VppmLstmDataset(Dataset):
    """결합 Dataset — row-wise 정렬 전제.

    Args:
        features_norm: (N, 21) float32  [-1,1] 로 정규화된 기존 피처
        target_norm:   (N,)    float32  해당 property 의 정규화된 타겟
        stacks_h5:     path    이미지 스택 h5 경로 (lazy open)
        row_indices:   (M,)    int      전역 행 중 해당 fold 가 쓸 row
    """

    def __init__(
        self,
        features_norm: np.ndarray,
        target_norm: np.ndarray,
        stacks_h5: str,
        row_indices: np.ndarray,
    ):
        self.x21 = features_norm
        self.y = target_norm
        self.stacks_h5 = stacks_h5
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self._h5: Optional[h5py.File] = None
        self._stacks = None
        self._masks = None

    def _lazy_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.stacks_h5, "r", libver="latest", swmr=False)
            self._stacks = self._h5["stacks"]
            self._masks = self._h5["masks"]

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, idx: int) -> dict:
        self._lazy_open()
        row = int(self.row_indices[idx])
        img = np.asarray(self._stacks[row], dtype=np.float32)        # (T,C,H,W)
        mask = np.asarray(self._masks[row], dtype=bool)              # (T,)
        return {
            "x21": torch.from_numpy(self.x21[row]),
            "img": torch.from_numpy(img),
            "mask": torch.from_numpy(mask),
            "y": torch.tensor(self.y[row], dtype=torch.float32),
        }

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass


def collate(batch: list) -> dict:
    return {
        "x21": torch.stack([b["x21"] for b in batch]),
        "img": torch.stack([b["img"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]).unsqueeze(1),
    }
