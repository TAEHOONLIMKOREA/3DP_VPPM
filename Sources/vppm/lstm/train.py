"""Phase L2 — Sample-LSTM 5-Fold CV 학습.

VPPM 과 동일한 sample-단위 K-Fold (seed=42) 로 데이터를 분할하고, 각 fold 의 LSTM 을 4 인장 물성
multi-task 회귀로 학습. 결과 가중치는 Sources/pipeline_outputs/models_lstm/lstm_sample_fold{i}.pt.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


# ── 분할 ────────────────────────────────────────────────


def make_fold_indices(entries, n_splits=config.N_FOLDS, seed=config.RANDOM_SEED):
    """샘플 단위 K-Fold — VPPM 과 동일한 KFold(seed=42, shuffle=True)."""
    n = len(entries)
    idx = np.arange(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(idx))


# ── 학습 루프 ────────────────────────────────────────────


def train_one_fold(
    train_entries,
    val_entries,
    targets_map,
    target_norm,
    cache_dir,
    raw_min,
    raw_max,
    device,
    out_path: Path,
    bidirectional: bool,
    d_embed: int,
    quick: bool = False,
):
    train_ds = SampleSequenceDataset(train_entries, cache_dir, targets_map, target_norm,
                                     raw_min, raw_max, max_seq_len=200 if quick else None)
    val_ds = SampleSequenceDataset(val_entries, cache_dir, targets_map, target_norm,
                                   raw_min, raw_max, max_seq_len=200 if quick else None)

    train_loader = DataLoader(train_ds, batch_size=config.LSTM_BATCH_SIZE,
                              shuffle=True, num_workers=config.LSTM_NUM_WORKERS,
                              collate_fn=collate_pad, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.LSTM_BATCH_SIZE,
                            shuffle=False, num_workers=config.LSTM_NUM_WORKERS,
                            collate_fn=collate_pad, pin_memory=True)

    model = SampleLSTMRegressor(bidirectional=bidirectional, d_embed=d_embed).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.LSTM_LR,
                             weight_decay=config.LSTM_WEIGHT_DECAY)
    criterion = nn.L1Loss()

    max_epochs = 20 if quick else config.LSTM_MAX_EPOCHS
    patience = config.LSTM_EARLY_STOP_PATIENCE
    best_val = float("inf")
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(max_epochs):
        # train
        model.train()
        train_losses = []
        t0 = time.time()
        for batch in train_loader:
            seq = batch["seq"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device)
            y = batch["y"].to(device)
            preds, _ = model(seq, lengths)
            loss = criterion(preds, y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.LSTM_GRAD_CLIP)
            optim.step()
            train_losses.append(loss.item())

        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(device, non_blocking=True)
                lengths = batch["lengths"].to(device)
                y = batch["y"].to(device)
                preds, _ = model(seq, lengths)
                val_losses.append(criterion(preds, y).item())

        tr = float(np.mean(train_losses))
        vl = float(np.mean(val_losses))
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": vl, "sec": dt})
        print(f"  epoch {epoch:3d}  train={tr:.4f}  val={vl:.4f}  ({dt:.1f}s)")

        if vl < best_val - 1e-5:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stop at epoch {epoch} (best val={best_val:.4f})")
                break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "best_val": best_val,
        "history": history,
        "target_norm": target_norm,
        "bidirectional": bidirectional,
        "d_embed": d_embed,
    }, out_path)
    return {"best_val": best_val, "epochs": len(history)}


# ── CLI ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase L2 — Sample-LSTM 5-Fold CV 학습")
    parser.add_argument("--mode", choices=list(config.LSTM_VALID_MODES), required=True,
                        help="fwd1 | bidir1 | fwd16 | bidir16")
    parser.add_argument("--cache-dir", default=config.LSTM_CACHE_DIR)
    parser.add_argument("--out-dir", default=None,
                        help="(생략 시) config.lstm_paths(mode)['models_dir']")
    parser.add_argument("--device", default=None,
                        help="cuda | cpu (default: auto)")
    parser.add_argument("--folds", nargs="+", type=int, default=None,
                        help="특정 fold 만 학습 (예: --folds 0 1)")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test (max_epochs=20, max_seq_len=200)")
    args = parser.parse_args()

    paths = config.lstm_paths(args.mode)
    bidirectional = paths["bidirectional"]
    d_embed = paths["d_embed"]

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir) if args.out_dir else paths["models_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[L2] mode={args.mode}  bidir={bidirectional}  d_embed={d_embed}  "
          f"device={device}  cache={cache_dir}  out={out_dir}")

    # 인덱스 + 타겟 + 정규화
    print("[L2] indexing cache...")
    index = SampleStackIndex(cache_dir)
    print(f"[L2] {len(index)} samples found")

    print("[L2] loading targets...")
    targets_map = load_targets_for_samples()

    # NaN 타겟 / UTS<50 샘플 제거 (origin/dataset.build_dataset 와 동일 조건)
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
    print(f"[L2] valid entries (non-NaN, UTS>=50): {len(valid_entries)}")

    norm_json = load_normalization_json(cache_dir)
    raw_min, raw_max = norm_json["raw_min"], norm_json["raw_max"]
    print(f"[L2] raw normalization: [{raw_min:.3f}, {raw_max:.3f}]")

    # 5-Fold split
    splits = make_fold_indices(valid_entries)

    # 타겟 정규화는 train fold 통계로
    fold_results = {}
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        if args.folds is not None and fold_idx not in args.folds:
            continue
        print(f"\n[L2] === Fold {fold_idx + 1}/{len(splits)} ===")
        train_entries = [valid_entries[i] for i in train_idx]
        val_entries = [valid_entries[i] for i in val_idx]
        target_norm = fit_target_normalization(train_entries, targets_map)
        print(f"     train={len(train_entries)} val={len(val_entries)}")

        out_path = out_dir / f"lstm_sample_fold{fold_idx}.pt"
        res = train_one_fold(
            train_entries, val_entries, targets_map, target_norm,
            cache_dir, raw_min, raw_max, device, out_path,
            bidirectional=bidirectional, d_embed=d_embed, quick=args.quick,
        )
        fold_results[fold_idx] = res
        print(f"     ✓ saved {out_path}, best_val={res['best_val']:.4f}")

    # 요약
    summary = {
        "mode": args.mode,
        "bidirectional": bidirectional,
        "d_embed": d_embed,
        "fold_results": fold_results,
        "n_valid_samples": len(valid_entries),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "device": device,
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[L2] ✓ all folds done. summary → {out_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
