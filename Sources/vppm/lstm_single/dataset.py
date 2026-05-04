"""
VPPMLSTMDataset — baseline 21 피처 + SV 별 크롭 시퀀스 + 타겟 동시 제공.

- baseline `all_features.npz` 의 (sample_ids, build_ids, features, target_*) 와
  L1 캐시 `crop_stacks_B1.x.h5` (sv_indices, sample_ids 로 1:1 매칭) 을 결합.
- 가변 길이 시퀀스는 (N, 70, 8, 8) zero-padded + lengths(N,) 형태로 메모리에 통째 로드.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import config
from ..common.dataset import normalize


def _load_cache_for_build(cache_path: Path) -> dict:
    """단일 빌드의 H5 캐시를 통째로 로드 → numpy."""
    with h5py.File(cache_path, "r") as f:
        return {
            "stacks": f["stacks"][...],           # (N, 70, 8, 8) float16
            "lengths": f["lengths"][...],         # (N,) int16
            "sv_indices": f["sv_indices"][...],   # (N, 3) int32
            "sample_ids": f["sample_ids"][...],   # (N,) int32
        }


def load_lstm_dataset(
    features_npz: Path = config.FEATURES_DIR / "all_features.npz",
    cache_dir: Path = config.LSTM_CACHE_DIR,
    build_ids: list[str] | None = None,
) -> dict:
    """baseline features.npz + L1 캐시를 합쳐 학습용 dict 반환.

    반환:
        features:    (N, 21) float32   — baseline 21 피처 (raw, 정규화 전)
        sample_ids:  (N,) int32        — sample-wise K-fold 용
        build_ids:   (N,) int32        — 빌드 인덱스
        targets:     {prop: (N,) float32}
        stacks:      (N, 70, 8, 8) float16   — 크롭 시퀀스 (padded)
        lengths:     (N,) int16        — 실제 시퀀스 길이
        valid_mask:  (N,) bool         — features.npz 와 캐시가 모두 매칭된 SV
    """
    if build_ids is None:
        build_ids = list(config.BUILDS.keys())

    # baseline features.npz 로드 (이미 빌드별 sample_id 오프셋 적용됨)
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

    # 빌드별 SV 캐시 로드 → 빌드 인덱스 순으로 concat
    # baseline merge 와 동일 순서를 유지: build_ids 순회 + 빌드 내부에서는 features.npz 의 등장 순서
    # → features.npz 는 빌드별로 sample_id 오프셋만 더해 concat 했고, 빌드 내부 순서는 features.npz 와 캐시가 모두 valid SV 순으로 동일.
    cache_paths = [Path(cache_dir) / f"crop_stacks_{bid}.h5" for bid in build_ids]
    for p in cache_paths:
        if not p.exists():
            raise FileNotFoundError(
                f"L1 캐시 누락: {p} — 먼저 `python -m Sources.vppm.lstm.run --phase cache` 실행"
            )

    stacks_list, lengths_list = [], []
    cache_sv_count_per_build = []
    for bid, cpath in zip(build_ids, cache_paths):
        cache = _load_cache_for_build(cpath)
        stacks_list.append(cache["stacks"])
        lengths_list.append(cache["lengths"])
        cache_sv_count_per_build.append(len(cache["lengths"]))

    cache_stacks = np.concatenate(stacks_list, axis=0)
    cache_lengths = np.concatenate(lengths_list, axis=0)

    # 무결성: features.npz 의 빌드별 SV 수 == 캐시의 빌드별 SV 수
    build_id_to_idx = {bid: i for i, bid in enumerate(config.BUILDS.keys())}
    feat_count_per_build = []
    for bid in build_ids:
        bi = build_id_to_idx[bid]
        feat_count_per_build.append(int((build_idx_arr == bi).sum()))

    if feat_count_per_build != cache_sv_count_per_build:
        raise ValueError(
            "features.npz 와 L1 캐시의 빌드별 SV 수가 다릅니다.\n"
            f"  features: {feat_count_per_build}\n"
            f"  cache:    {cache_sv_count_per_build}\n"
            "캐시를 다시 빌드하거나 features.npz 를 재생성하세요."
        )

    # features.npz 는 build_ids 순으로 concat 되어 있으므로 (B1.1, ..., B1.5),
    # cache 도 동일 순서로 concat 했다면 인덱싱 일치.
    # 단, features.npz 의 build_idx_arr 순서가 정렬되어 있는지 한 번 더 확인.
    sorted_order = []
    for bid in build_ids:
        bi = build_id_to_idx[bid]
        sorted_order.extend(np.where(build_idx_arr == bi)[0].tolist())
    sorted_order = np.array(sorted_order, dtype=np.int64)

    if not np.array_equal(sorted_order, np.arange(N)):
        # features.npz 가 빌드 순으로 정렬되어 있지 않으면 — 정렬해서 stacks 와 매칭
        features = features[sorted_order]
        sample_ids = sample_ids[sorted_order]
        build_idx_arr = build_idx_arr[sorted_order]
        targets = {k: v[sorted_order] for k, v in targets.items()}

    if len(cache_stacks) != N:
        raise ValueError(f"매칭 실패: features N={N}, stacks N={len(cache_stacks)}")

    return {
        "features": features.astype(np.float32),
        "sample_ids": sample_ids.astype(np.int32),
        "build_ids": build_idx_arr.astype(np.int32),
        "targets": {k: v.astype(np.float32) for k, v in targets.items()},
        "stacks": cache_stacks,                       # float16, padded
        "lengths": cache_lengths.astype(np.int64),    # LSTM pack_padded 는 int64/cpu 권장
    }


def build_normalized_dataset(raw: dict) -> dict:
    """baseline build_dataset 과 같은 정규화/필터링을 LSTM 입력에도 적용.

    baseline 의 build_dataset 을 그대로 호출하지 않는 이유:
      - LSTM 데이터셋은 stacks/lengths 까지 같은 valid_mask 로 정렬해야 함
      - 따라서 동일 로직을 인라인.

    반환:
        features (norm), features_raw, sample_ids, build_ids, stacks, lengths,
        targets (norm), targets_raw, norm_params
    """
    feats = raw["features"]
    sids = raw["sample_ids"]
    bids = raw["build_ids"]
    stacks = raw["stacks"]
    lengths = raw["lengths"]
    tgts = raw["targets"]

    # baseline 과 동일한 valid_mask 규칙
    uts = tgts.get("ultimate_tensile_strength", np.zeros(len(feats)))
    valid = ~np.isnan(uts) & (uts >= 50.0)
    for prop in config.TARGET_PROPERTIES:
        if prop in tgts:
            valid &= ~np.isnan(tgts[prop])
    valid &= ~np.isnan(feats).any(axis=1)
    # SV 길이 0 이면 학습 불가 — 안전장치
    valid &= (lengths > 0)

    feats = feats[valid]
    sids = sids[valid]
    bids = bids[valid]
    stacks = stacks[valid]
    lengths = lengths[valid]
    tgts = {k: v[valid] for k, v in tgts.items()}

    # 피처 정규화 (baseline 과 동일 방식 — feature-wise min/max)
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
        "stacks": stacks,
        "lengths": lengths,
        "targets": tgt_norm,
        "targets_raw": tgts,
        "norm_params": norm_params,
    }


class VPPMLSTMDataset(Dataset):
    """(features21, stacks, length, target) 한 쌍 반환."""

    def __init__(self, features: np.ndarray, stacks: np.ndarray,
                 lengths: np.ndarray, targets: np.ndarray):
        # features: (N, 21) float32 normalized
        # stacks:   (N, 70, 8, 8) float16 padded
        # lengths:  (N,) int64
        # targets:  (N,) float32 normalized
        self.features = torch.from_numpy(features).float()
        # float16 그대로 두면 BN/LSTM 에서 dtype 충돌 → 학습 시 float32 로 캐스팅
        self.stacks = torch.from_numpy(stacks)
        self.lengths = torch.from_numpy(lengths.astype(np.int64))
        self.targets = torch.from_numpy(targets).float().unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.stacks[i], self.lengths[i], self.targets[i]


def collate_fn(batch):
    """배치: (feat, stack, length, target) 튜플 리스트 → 텐서 stack."""
    feats = torch.stack([b[0] for b in batch], dim=0)
    stacks = torch.stack([b[1] for b in batch], dim=0).float()   # float16 → float32
    lengths = torch.stack([b[2] for b in batch], dim=0)
    targets = torch.stack([b[3] for b in batch], dim=0)
    return feats, stacks, lengths, targets
