"""
Phase D2 — VPPM-LSTM-Dual 학습 (4 properties × 5 folds = 20 모델)

기존 `lstm/train.py` 와 동일 골격. 차이는 모델 입력이 (feat21, stack_v0, stack_v1, lengths) 4-tuple.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..baseline.train import EarlyStopper
from ..common import config
from ..common.dataset import create_cv_splits
from .dataset import VPPMLSTMDualDataset, collate_fn
from .model import VPPM_LSTM_Dual


def train_single_fold(
    feat_train: np.ndarray, sv0_train: np.ndarray, sv1_train: np.ndarray,
    lengths_train: np.ndarray, targets_train: np.ndarray,
    feat_val: np.ndarray, sv0_val: np.ndarray, sv1_val: np.ndarray,
    lengths_val: np.ndarray, targets_val: np.ndarray,
    device: str = "cpu",
    grad_clip: float = config.LSTM_GRAD_CLIP,
) -> dict:
    train_ds = VPPMLSTMDualDataset(feat_train, sv0_train, sv1_train, lengths_train, targets_train)
    val_ds = VPPMLSTMDualDataset(feat_val, sv0_val, sv1_val, lengths_val, targets_val)
    train_loader = DataLoader(
        train_ds, batch_size=config.LSTM_BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=config.LSTM_NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.LSTM_BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=config.LSTM_NUM_WORKERS,
    )

    model = VPPM_LSTM_Dual().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LSTM_LR,
        betas=config.ADAM_BETAS,
        eps=config.ADAM_EPS,
        weight_decay=config.LSTM_WEIGHT_DECAY,
    )
    criterion = nn.L1Loss()
    stopper = EarlyStopper(patience=config.LSTM_EARLY_STOP_PATIENCE)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(config.LSTM_MAX_EPOCHS):
        # --- Train ---
        model.train()
        train_losses = []
        for feats, sv0, sv1, lengths, ys in train_loader:
            feats = feats.to(device, non_blocking=True)
            sv0 = sv0.to(device, non_blocking=True)
            sv1 = sv1.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            # lengths 는 cpu 유지 (pack_padded_sequence 요구사항)

            optimizer.zero_grad()
            pred = model(feats, sv0, sv1, lengths)
            loss = criterion(pred, ys)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for feats, sv0, sv1, lengths, ys in val_loader:
                feats = feats.to(device, non_blocking=True)
                sv0 = sv0.to(device, non_blocking=True)
                sv1 = sv1.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True)
                pred = model(feats, sv0, sv1, lengths)
                loss = criterion(pred, ys)
                val_loss_sum += loss.item() * len(ys)
                val_n += len(ys)

        train_loss = float(np.mean(train_losses))
        val_loss = val_loss_sum / max(val_n, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if stopper.check(val_loss, model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    return {
        "model_state": model.state_dict(),
        "history": history,
        "best_val_loss": stopper.best_score,
        "epochs": len(history["train_loss"]),
    }


def train_all(dataset: dict,
              output_dir: Path = config.LSTM_DUAL_MODELS_DIR,
              device: str = "cpu") -> dict:
    """4 properties × 5 fold 학습. 모델 저장 → output_dir/vppm_lstm_dual_{short}_fold{k}.pt."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feats = dataset["features"]
    sv0 = dataset["stacks_v0"]
    sv1 = dataset["stacks_v1"]
    lengths = dataset["lengths"]
    sids = dataset["sample_ids"]
    splits = create_cv_splits(sids)

    all_results = {}
    for prop in config.TARGET_PROPERTIES:
        short = config.TARGET_SHORT[prop]
        if prop not in dataset["targets"]:
            print(f"  skip {short}: no target")
            continue
        targets = dataset["targets"][prop]

        print(f"\n{'='*60}\nTraining VPPM-LSTM-Dual for {short}\n{'='*60}")
        fold_results = []
        for fold, (train_mask, val_mask) in enumerate(splits):
            print(f"  Fold {fold+1}/{config.N_FOLDS}...")
            result = train_single_fold(
                feats[train_mask], sv0[train_mask], sv1[train_mask],
                lengths[train_mask], targets[train_mask],
                feats[val_mask], sv0[val_mask], sv1[val_mask],
                lengths[val_mask], targets[val_mask],
                device=device,
            )
            fold_results.append(result)

            model_path = output_dir / f"vppm_lstm_dual_{short}_fold{fold}.pt"
            torch.save(result["model_state"], model_path)
            print(f"    epochs={result['epochs']}  best_val={result['best_val_loss']:.6f}")

        all_results[prop] = fold_results

    log = {}
    for prop, results in all_results.items():
        short = config.TARGET_SHORT[prop]
        log[short] = {
            "fold_val_losses": [r["best_val_loss"] for r in results],
            "fold_epochs": [r["epochs"] for r in results],
        }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete → {output_dir}")
    return all_results
