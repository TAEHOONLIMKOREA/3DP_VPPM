"""VPPM-1DCNN 학습 — baseline ``baseline/train.py`` 미러.

차이점만:
  - 입력 텐서가 (B, 21, 70) 의 시퀀스
  - 모델은 ``VPPM_1DCNN``
  - 모델 저장 경로 prefix 가 ``vppm_1dcnn_`` (baseline 의 ``vppm_`` 와 구분)
  - grad_clip = 1.0 (baseline 도 동일 골격이지만 명시적으로 적용)

학습 하이퍼는 모두 ``common.config`` 의 baseline 값 그대로:
  L1Loss, Adam(lr=1e-3), batch=BATCH_SIZE, EarlyStopper(patience=50, max=5000).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..baseline.train import EarlyStopper
from ..common import config as common_config
from ..common.dataset import create_cv_splits
from . import config as exp_config
from .dataset import VPPM1DCNNDataset
from .model import VPPM_1DCNN


GRAD_CLIP = 1.0


def train_single_fold(
    feat_train: np.ndarray, targets_train: np.ndarray,
    feat_val: np.ndarray, targets_val: np.ndarray,
    device: str = "cpu",
) -> dict:
    """단일 fold 학습. feat_*: (N, 21, 70) normalized."""
    train_ds = VPPM1DCNNDataset(feat_train, targets_train)
    val_ds = VPPM1DCNNDataset(feat_val, targets_val)
    train_loader = DataLoader(train_ds, batch_size=common_config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=common_config.BATCH_SIZE)

    model = VPPM_1DCNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=common_config.LEARNING_RATE,
        betas=common_config.ADAM_BETAS,
        eps=common_config.ADAM_EPS,
    )
    criterion = nn.L1Loss()
    stopper = EarlyStopper(patience=common_config.EARLY_STOP_PATIENCE)

    history = {"train_loss": [], "val_loss": []}

    for _ in range(common_config.MAX_EPOCHS):
        # --- Train ---
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            if GRAD_CLIP is not None and GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss_sum += loss.item() * len(batch_y)
                val_n += len(batch_y)

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
              output_dir: Path = exp_config.MODELS_DIR,
              device: str = "cpu") -> dict:
    """4 properties × 5 folds = 20 모델 학습."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feats = dataset["features"]      # (N, 21, 70)
    sids = dataset["sample_ids"]
    splits = create_cv_splits(sids)

    all_results = {}
    for prop in common_config.TARGET_PROPERTIES:
        short = common_config.TARGET_SHORT[prop]
        if prop not in dataset["targets"]:
            print(f"  skip {short}: no target")
            continue
        targets = dataset["targets"][prop]

        print(f"\n{'='*60}\nTraining VPPM-1DCNN for {short}\n{'='*60}")
        fold_results = []
        for fold, (train_mask, val_mask) in enumerate(splits):
            print(f"  Fold {fold+1}/{common_config.N_FOLDS}...")
            result = train_single_fold(
                feats[train_mask], targets[train_mask],
                feats[val_mask], targets[val_mask],
                device=device,
            )
            fold_results.append(result)

            model_path = output_dir / f"vppm_1dcnn_{short}_fold{fold}.pt"
            torch.save(result["model_state"], model_path)
            print(f"    epochs={result['epochs']}  best_val={result['best_val_loss']:.6f}")

        all_results[prop] = fold_results

    log = {}
    for prop, results in all_results.items():
        short = common_config.TARGET_SHORT[prop]
        log[short] = {
            "fold_val_losses": [r["best_val_loss"] for r in results],
            "fold_epochs": [r["epochs"] for r in results],
        }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete → {output_dir}")
    return all_results
