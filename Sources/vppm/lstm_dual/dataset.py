"""
VPPMLSTMDualDataset — baseline 21 피처 + visible/0 시퀀스 + visible/1 시퀀스 + 타겟 동시 제공.

- baseline `all_features.npz` 의 21 피처 (sample-id 정렬) 와
  v0 캐시 `experiments/vppm_lstm/cache/crop_stacks_B1.x.h5`,
  v1 캐시 `experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.x.h5` 를 결합.
- 두 채널의 lengths/sv_indices/sample_ids 가 비트 단위 일치한다고 가정 (verify_v0_v1_consistency 로 사전 검증).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import config
from ..common.dataset import normalize


def _load_v0_cache(cache_path: Path) -> dict:
    with h5py.File(cache_path, "r") as f:
        return {
            "stacks": f["stacks"][...],
            "lengths": f["lengths"][...],
            "sv_indices": f["sv_indices"][...],
            "sample_ids": f["sample_ids"][...],
        }


def _load_v1_cache(cache_path: Path) -> dict:
    with h5py.File(cache_path, "r") as f:
        return {
            "stacks": f["stacks"][...],
            "lengths": f["lengths"][...],
            "sv_indices": f["sv_indices"][...],
            "sample_ids": f["sample_ids"][...],
        }


def load_dual_dataset(
    features_npz: Path = config.FEATURES_DIR / "all_features.npz",
    cache_v0_dir: Path = config.LSTM_CACHE_DIR,
    cache_v1_dir: Path = config.LSTM_DUAL_CACHE_DIR,
    build_ids: list[str] | None = None,
) -> dict:
    """baseline features.npz + v0 캐시 + v1 캐시를 합쳐 학습용 dict 반환.

    반환:
        features:    (N, 21) float32   — baseline 21 피처 (raw, 정규화 전)
        sample_ids:  (N,) int32
        build_ids:   (N,) int32
        targets:     {prop: (N,) float32}
        stacks_v0:   (N, 70, 8, 8) float16   — visible/0 padded 시퀀스
        stacks_v1:   (N, 70, 8, 8) float16   — visible/1 padded 시퀀스
        lengths:     (N,) int64               — 두 채널 공통
    """
    if build_ids is None:
        build_ids = list(config.BUILDS.keys())

    feat_data = np.load(features_npz)
    features = feat_data["features"]
    sample_ids = feat_data["sample_ids"]
    build_idx_arr = feat_data["build_ids"]
    targets = {
        p: feat_data[f"target_{p}"]
        for p in config.TARGET_PROPERTIES
        if f"target_{p}" in feat_data.files
    }
    N = len(features)

    cache_v0_paths = [Path(cache_v0_dir) / f"crop_stacks_{bid}.h5" for bid in build_ids]
    cache_v1_paths = [Path(cache_v1_dir) / f"crop_stacks_v1_{bid}.h5" for bid in build_ids]
    for p in cache_v0_paths:
        if not p.exists():
            raise FileNotFoundError(
                f"v0 캐시 누락: {p} — `python -m Sources.vppm.lstm.run --phase cache` 먼저 실행"
            )
    for p in cache_v1_paths:
        if not p.exists():
            raise FileNotFoundError(
                f"v1 캐시 누락: {p} — `python -m Sources.vppm.lstm_dual.run --phase cache_v1` 먼저 실행"
            )

    v0_stacks_list, v1_stacks_list, lengths_list = [], [], []
    cache_sv_count_per_build = []
    for v0p, v1p in zip(cache_v0_paths, cache_v1_paths):
        c0 = _load_v0_cache(v0p)
        c1 = _load_v1_cache(v1p)

        # 빌드 내부에서 v0/v1 일치 검증
        if not np.array_equal(c0["lengths"], c1["lengths"]):
            raise RuntimeError(
                f"v0/v1 lengths 불일치 — {v0p.name} vs {v1p.name}.\n"
                "캐시를 다시 빌드하세요 (lstm_dual.crop_stacks_v1.verify_v0_v1_consistency)"
            )
        if not np.array_equal(c0["sv_indices"], c1["sv_indices"]):
            raise RuntimeError(f"v0/v1 sv_indices 불일치 — {v0p.name} vs {v1p.name}")
        if not np.array_equal(c0["sample_ids"], c1["sample_ids"]):
            raise RuntimeError(f"v0/v1 sample_ids 불일치 — {v0p.name} vs {v1p.name}")

        v0_stacks_list.append(c0["stacks"])
        v1_stacks_list.append(c1["stacks"])
        lengths_list.append(c0["lengths"])
        cache_sv_count_per_build.append(len(c0["lengths"]))

    cache_stacks_v0 = np.concatenate(v0_stacks_list, axis=0)
    cache_stacks_v1 = np.concatenate(v1_stacks_list, axis=0)
    cache_lengths = np.concatenate(lengths_list, axis=0)

    # features.npz 와 캐시의 빌드별 SV 수 일치 검증
    build_id_to_idx = {bid: i for i, bid in enumerate(config.BUILDS.keys())}
    feat_count_per_build = []
    for bid in build_ids:
        bi = build_id_to_idx[bid]
        feat_count_per_build.append(int((build_idx_arr == bi).sum()))

    if feat_count_per_build != cache_sv_count_per_build:
        raise ValueError(
            "features.npz 와 캐시의 빌드별 SV 수가 다릅니다.\n"
            f"  features: {feat_count_per_build}\n"
            f"  cache:    {cache_sv_count_per_build}"
        )

    # features.npz 가 build_ids 순으로 정렬돼있는지 확인 → 아니면 정렬 후 매칭
    sorted_order = []
    for bid in build_ids:
        bi = build_id_to_idx[bid]
        sorted_order.extend(np.where(build_idx_arr == bi)[0].tolist())
    sorted_order = np.array(sorted_order, dtype=np.int64)

    if not np.array_equal(sorted_order, np.arange(N)):
        features = features[sorted_order]
        sample_ids = sample_ids[sorted_order]
        build_idx_arr = build_idx_arr[sorted_order]
        targets = {k: v[sorted_order] for k, v in targets.items()}

    if len(cache_stacks_v0) != N or len(cache_stacks_v1) != N:
        raise ValueError(
            f"매칭 실패: features N={N}, v0 N={len(cache_stacks_v0)}, v1 N={len(cache_stacks_v1)}"
        )

    return {
        "features": features.astype(np.float32),
        "sample_ids": sample_ids.astype(np.int32),
        "build_ids": build_idx_arr.astype(np.int32),
        "targets": {k: v.astype(np.float32) for k, v in targets.items()},
        "stacks_v0": cache_stacks_v0,
        "stacks_v1": cache_stacks_v1,
        "lengths": cache_lengths.astype(np.int64),
    }


def build_normalized_dataset(raw: dict) -> dict:
    """baseline build_dataset 과 동일한 valid_mask + 정규화. v0/v1 stacks 도 같은 mask 로 슬라이스."""
    feats = raw["features"]
    sids = raw["sample_ids"]
    bids = raw["build_ids"]
    sv0 = raw["stacks_v0"]
    sv1 = raw["stacks_v1"]
    lengths = raw["lengths"]
    tgts = raw["targets"]

    uts = tgts.get("ultimate_tensile_strength", np.zeros(len(feats)))
    valid = ~np.isnan(uts) & (uts >= 50.0)
    for prop in config.TARGET_PROPERTIES:
        if prop in tgts:
            valid &= ~np.isnan(tgts[prop])
    valid &= ~np.isnan(feats).any(axis=1)
    valid &= (lengths > 0)

    feats = feats[valid]
    sids = sids[valid]
    bids = bids[valid]
    sv0 = sv0[valid]
    sv1 = sv1[valid]
    lengths = lengths[valid]
    tgts = {k: v[valid] for k, v in tgts.items()}

    f_min = feats.min(axis=0)
    f_max = feats.max(axis=0)
    feats_norm = normalize(feats, f_min, f_max).astype(np.float32)

    tgt_norm = {}
    t_min, t_max = {}, {}
    for prop in config.TARGET_PROPERTIES:
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
        "features": feats_norm,
        "features_raw": feats,
        "sample_ids": sids,
        "build_ids": bids,
        "stacks_v0": sv0,
        "stacks_v1": sv1,
        "lengths": lengths,
        "targets": tgt_norm,
        "targets_raw": tgts,
        "norm_params": norm_params,
    }


class VPPMLSTMDualDataset(Dataset):
    """(features21, stack_v0, stack_v1, length, target) 한 쌍 반환."""

    def __init__(self, features: np.ndarray,
                 stacks_v0: np.ndarray, stacks_v1: np.ndarray,
                 lengths: np.ndarray, targets: np.ndarray):
        # features:  (N, 21) float32 normalized
        # stacks_v*: (N, 70, 8, 8) float16 padded
        # lengths:   (N,) int64
        # targets:   (N,) float32 normalized
        self.features = torch.from_numpy(features).float()
        self.stacks_v0 = torch.from_numpy(stacks_v0)        # float16 → 학습 시 float32 캐스팅
        self.stacks_v1 = torch.from_numpy(stacks_v1)
        self.lengths = torch.from_numpy(lengths.astype(np.int64))
        self.targets = torch.from_numpy(targets).float().unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return (self.features[i], self.stacks_v0[i], self.stacks_v1[i],
                self.lengths[i], self.targets[i])


def collate_fn(batch):
    feats = torch.stack([b[0] for b in batch], dim=0)
    sv0 = torch.stack([b[1] for b in batch], dim=0).float()   # float16 → float32
    sv1 = torch.stack([b[2] for b in batch], dim=0).float()
    lengths = torch.stack([b[3] for b in batch], dim=0)
    targets = torch.stack([b[4] for b in batch], dim=0)
    return feats, sv0, sv1, lengths, targets
