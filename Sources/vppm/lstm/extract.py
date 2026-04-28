"""Phase L3 — 학습된 LSTM 으로 샘플별 임베딩 추출 + 21→37 차원 npz 통합.

- 5 fold 의 모델로 각 샘플의 임베딩 (16-dim) 을 모두 추출 → (n_samples, 5, 16)
- 슈퍼복셀 단위 all_features.npz 와 통합 — voxel 의 fold 배정에 따라 같은 fold 의 임베딩을 concat.
  (data leakage 방지: 한 voxel 이 학습 set 에 들어갈 때 / val set 에 들어갈 때 다른 fold 의 LSTM 임베딩
   이어야 일관성 보장.)

산출물:
  Sources/pipeline_outputs/lstm_embeddings/embeddings.npz
  Sources/pipeline_outputs/features/all_features_with_lstm.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from ..common import config
from .lstm_model import SampleLSTMRegressor
from .sample_dataset import (
    SampleSequenceDataset,
    SampleStackIndex,
    collate_pad,
    fit_target_normalization,
    load_normalization_json,
    load_targets_for_samples,
)


# ── 공용: 동일한 fold split 을 train.py 와 일치하게 재현 ─────


def split_entries(valid_entries, n_splits=config.N_FOLDS, seed=config.RANDOM_SEED):
    n = len(valid_entries)
    idx = np.arange(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(idx))


# ── 임베딩 추출 ──────────────────────────────────────────


@torch.no_grad()
def extract_embeddings(
    model: SampleLSTMRegressor,
    entries,
    cache_dir,
    targets_map,
    target_norm,
    raw_min, raw_max,
    device,
) -> dict:
    ds = SampleSequenceDataset(entries, cache_dir, targets_map, target_norm,
                               raw_min, raw_max)
    loader = DataLoader(ds, batch_size=config.LSTM_BATCH_SIZE, shuffle=False,
                        num_workers=config.LSTM_NUM_WORKERS, collate_fn=collate_pad,
                        pin_memory=True)
    model.eval()
    out = {}  # (build_id, sample_id) -> embedding (d_embed,)
    for batch in loader:
        seq = batch["seq"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device)
        emb = model.encode(seq, lengths)                # (B, d_embed)
        emb_np = emb.detach().cpu().numpy()
        for i, (bid, sid) in enumerate(zip(batch["build_ids"], batch["sample_ids"])):
            out[(bid, sid)] = emb_np[i]
    return out


# ── npz 통합 ─────────────────────────────────────────────


def integrate_features(
    embeddings_per_fold: list[dict],   # [fold0_emb, fold1_emb, ...] each {(bid, sid): vec}
    all_features_npz: Path,
    out_npz: Path,
    fold_assignment: np.ndarray,       # (N_voxels,) — 각 voxel 이 어느 fold 의 val set 에 속하는지
    d_embed: int,
):
    """all_features.npz 의 (N, 21) 을 (N, 21 + d_embed) 로 확장."""
    data = np.load(all_features_npz)
    feats = data["features"]
    sample_ids = data["sample_ids"]
    build_ids = data["build_ids"] if "build_ids" in data.files else None

    N = feats.shape[0]
    feats_v2 = np.zeros((N, feats.shape[1] + d_embed), dtype=np.float32)
    feats_v2[:, : feats.shape[1]] = feats

    if build_ids is None:
        raise RuntimeError("all_features.npz 에 build_ids 필요 (LSTM 임베딩 매핑용).")

    missing = 0
    for v in range(N):
        sid = int(sample_ids[v])
        bid = build_ids[v]
        if isinstance(bid, bytes):
            bid = bid.decode()
        f = int(fold_assignment[v])
        emb_dict = embeddings_per_fold[f]
        emb = emb_dict.get((bid, sid))
        if emb is None:
            missing += 1
            continue
        feats_v2[v, feats.shape[1]:] = emb

    if missing:
        print(f"[L3] WARN: {missing}/{N} voxels 이 임베딩 매핑 실패 (기본 0 으로 채움)")

    np.savez(
        out_npz,
        features=feats_v2,
        sample_ids=sample_ids,
        build_ids=build_ids if build_ids is not None else np.array([]),
        **{k: data[k] for k in data.files if k not in ("features", "sample_ids", "build_ids")},
    )
    print(f"[L3] ✓ {out_npz} 저장 (shape={feats_v2.shape})")


# ── voxel fold assignment ─────────────────────────────────


def voxel_fold_assignment(sample_ids: np.ndarray, n_splits=config.N_FOLDS,
                          seed=config.RANDOM_SEED) -> np.ndarray:
    """각 voxel 의 sample_id 가 어느 fold 의 val set 에 속하는지.

    VPPM 의 origin/dataset.create_cv_splits 와 동일 로직 — KFold(unique_samples).
    """
    unique = np.unique(sample_ids)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sample_to_fold = {}
    for fold, (_, val_idx) in enumerate(kf.split(unique)):
        for s in unique[val_idx]:
            sample_to_fold[int(s)] = fold
    return np.array([sample_to_fold[int(s)] for s in sample_ids], dtype=np.int8)


# ── CLI ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase L3 — 임베딩 추출 + npz 통합")
    parser.add_argument("--mode", choices=list(config.LSTM_VALID_MODES), required=True,
                        help="fwd1 | bidir1 | fwd16 | bidir16")
    parser.add_argument("--cache-dir", default=config.LSTM_CACHE_DIR)
    parser.add_argument("--models-dir", default=None,
                        help="(생략 시) config.lstm_paths(mode)['models_dir']")
    parser.add_argument("--embeddings-out", default=None,
                        help="(생략 시) config.lstm_paths(mode)['embeddings_npz']")
    parser.add_argument("--features-in", default=str(config.FEATURES_DIR / "all_features.npz"))
    parser.add_argument("--features-out", default=None,
                        help="(생략 시) config.lstm_paths(mode)['features_npz']")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    paths = config.lstm_paths(args.mode)
    bidirectional = paths["bidirectional"]
    d_embed = paths["d_embed"]
    cache_dir = Path(args.cache_dir)
    models_dir = Path(args.models_dir) if args.models_dir else paths["models_dir"]
    embeddings_out = Path(args.embeddings_out) if args.embeddings_out else paths["embeddings_npz"]
    features_out = Path(args.features_out) if args.features_out else paths["features_npz"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[L3] mode={args.mode}  bidir={bidirectional}  d_embed={d_embed}  device={device}")

    # 1) 인덱스 + 타겟 (train.py 와 동일)
    index = SampleStackIndex(cache_dir)
    targets_map = load_targets_for_samples()
    valid_entries = []
    for e in index.entries:
        bid, sid, _, _ = e
        tgt = targets_map.get((bid, sid))
        if tgt is None:
            continue
        uts = tgt["ultimate_tensile_strength"]
        if np.isnan(uts) or uts < 50.0:
            continue
        if any(np.isnan(tgt[p]) for p in config.TARGET_PROPERTIES):
            continue
        valid_entries.append(e)
    print(f"[L3] valid entries: {len(valid_entries)}")

    norm_json = load_normalization_json(cache_dir)
    raw_min, raw_max = norm_json["raw_min"], norm_json["raw_max"]

    # 2) fold split (train.py 와 동일 split)
    splits = split_entries(valid_entries)

    # 3) fold 별 모델 로드 + 임베딩 추출
    embeddings_per_fold: list[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        ckpt_path = models_dir / f"lstm_sample_fold{fold_idx}.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_bidir = ckpt.get("bidirectional", bidirectional)
        ckpt_d_embed = ckpt.get("d_embed", d_embed)
        if ckpt_bidir != bidirectional or ckpt_d_embed != d_embed:
            raise RuntimeError(
                f"checkpoint (bidir={ckpt_bidir}, d_embed={ckpt_d_embed}) 와 "
                f"mode={args.mode} (bidir={bidirectional}, d_embed={d_embed}) 가 불일치 — "
                f"잘못된 모드 디렉터리?")
        model = SampleLSTMRegressor(bidirectional=bidirectional, d_embed=d_embed).to(device)
        model.load_state_dict(ckpt["model_state"])
        target_norm = ckpt["target_norm"]
        # 모든 샘플 (train + val) 의 임베딩을 추출 — 추론이라 OK
        emb_dict = extract_embeddings(
            model, valid_entries, cache_dir, targets_map, target_norm,
            raw_min, raw_max, device,
        )
        print(f"[L3] fold {fold_idx}: {len(emb_dict)} embeddings")
        embeddings_per_fold.append(emb_dict)

    # 4) embeddings.npz 저장
    embeddings_out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(embeddings_per_fold[0].keys())
    arr = np.zeros((len(keys), len(splits), d_embed), dtype=np.float32)
    for fi, ed in enumerate(embeddings_per_fold):
        for ki, key in enumerate(keys):
            arr[ki, fi] = ed.get(key, np.zeros(d_embed))
    np.savez(
        embeddings_out,
        embeddings=arr,
        build_ids=np.array([k[0] for k in keys]),
        sample_ids=np.array([k[1] for k in keys], dtype=np.uint32),
    )
    print(f"[L3] ✓ {embeddings_out} 저장 (shape={arr.shape})")

    # 5) all_features_with_lstm_{mode}.npz 통합
    feat_in = Path(args.features_in)
    feat_out = features_out
    feat_out.parent.mkdir(parents=True, exist_ok=True)
    data = np.load(feat_in, allow_pickle=True)
    sample_ids_v = data["sample_ids"]
    fold_assign = voxel_fold_assignment(sample_ids_v)
    integrate_features(embeddings_per_fold, feat_in, feat_out, fold_assign, d_embed=d_embed)


if __name__ == "__main__":
    main()
