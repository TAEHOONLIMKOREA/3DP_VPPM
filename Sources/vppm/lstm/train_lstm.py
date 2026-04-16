"""
Phase L5 — VPPM-LSTM 학습 루프.

- 기존 features.npz 재사용 (재추출 금지)
- 기존 dataset.py 의 정규화·sample-wise CV 분할 재사용
- 4 property × 5 fold = 20 모델 전부 영속화 (재평가용, ~30MB)
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import config
from ..dataset import build_dataset, create_cv_splits, save_norm_params
from ..model import VPPM_LSTM
from .dataset import VppmLstmDataset, collate, load_aligned_arrays


def _make_loader(
    x21_norm: np.ndarray,
    target_norm: np.ndarray,
    stacks_h5: str,
    rows: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = VppmLstmDataset(x21_norm, target_norm, stacks_h5, rows)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.LSTM_NUM_WORKERS,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(config.LSTM_NUM_WORKERS > 0),
    )


def _train_one_fold(
    x21_norm: np.ndarray,
    target_norm: np.ndarray,
    stacks_h5: str,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    in_channels: int,
    device: str,
) -> dict:
    """단일 fold 학습 + best_state 반환."""
    train_loader = _make_loader(
        x21_norm, target_norm, stacks_h5, train_rows,
        batch_size=config.LSTM_BATCH_SIZE, shuffle=True,
    )
    val_loader = _make_loader(
        x21_norm, target_norm, stacks_h5, val_rows,
        batch_size=config.LSTM_BATCH_SIZE, shuffle=False,
    )

    model = VPPM_LSTM(
        n_hand=config.N_FEATURES,
        in_channels=in_channels,
        d_cnn=config.LSTM_D_CNN,
        d_lstm=config.LSTM_D_EMBED,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT_RATE,
        bidirectional=config.LSTM_BIDIRECTIONAL,
        num_lstm_layers=config.LSTM_NUM_LAYERS,
        pooling=config.LSTM_POOLING,
    ).to(device)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=config.LSTM_LR,
        betas=config.ADAM_BETAS,
        eps=config.ADAM_EPS,
        weight_decay=config.LSTM_WEIGHT_DECAY,
    )
    criterion = nn.L1Loss()

    best_val = float("inf")
    best_state = None
    patience = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(config.LSTM_MAX_EPOCHS):
        model.train()
        tr_losses = []
        for batch in train_loader:
            x21 = batch["x21"].to(device, non_blocking=True)
            img = batch["img"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            pred = model(x21, img, mask)
            loss = criterion(pred, y)
            loss.backward()
            if config.LSTM_GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.LSTM_GRAD_CLIP
                )
            optim.step()
            tr_losses.append(loss.item())

        # ---------- validation ----------
        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x21 = batch["x21"].to(device, non_blocking=True)
                img = batch["img"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                pred = model(x21, img, mask)
                val_total += criterion(pred, y).item() * len(y)
                val_count += len(y)
        val_loss = val_total / max(val_count, 1)
        tr_loss = float(np.mean(tr_losses))
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config.LSTM_EARLY_STOP_PATIENCE:
                break

        print(f"    epoch {epoch + 1:3d}  train={tr_loss:.4f}  val={val_loss:.4f}"
              f"  (best={best_val:.4f})")

    return {
        "best_state": best_state,
        "best_val_loss": best_val,
        "epochs": len(history["train_loss"]),
        "history": history,
    }


def train_vppm_lstm(device: str | None = None,
                    builds_skip_existing: bool = True) -> dict:
    """전체 학습 — 4 properties × 5 folds."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[L5] device: {device}")

    # -------- 1) 기존 features 로드 (재추출 없음) ----------
    arr = load_aligned_arrays()
    features_raw = arr["features"]
    sample_ids = arr["sample_ids"]
    targets_raw = arr["targets"]
    build_ids = arr["build_ids"]
    stacks_h5 = arr["stacks_h5"]
    in_channels = arr["C"]
    print(f"[L5] N={arr['n']} T={arr['T']} C={in_channels} "
          f"patch={arr['patch_px']} channels={arr['channels_name']}")

    # -------- 2) 정규화 (기존 dataset.build_dataset 재사용) ----------
    ds = build_dataset(features_raw, sample_ids, targets_raw, build_ids)
    # build_dataset 이 NaN/UTS<50 필터링을 하므로 valid row 를 역산
    valid_count = len(ds["features"])
    # valid row 의 "원본 row index" 를 재구성해야 stacks_h5 인덱싱 가능
    # build_dataset 는 valid_mask 를 반환하지 않음 → 동일 규칙을 여기서 재계산
    uts = targets_raw.get("ultimate_tensile_strength",
                          np.zeros(len(features_raw)))
    valid_mask = ~np.isnan(uts) & (uts >= 50.0)
    for p in config.TARGET_PROPERTIES:
        if p in targets_raw:
            valid_mask &= ~np.isnan(targets_raw[p])
    valid_mask &= ~np.isnan(features_raw).any(axis=1)
    orig_rows = np.where(valid_mask)[0]
    assert len(orig_rows) == valid_count, (
        f"valid_mask 재계산 결과({len(orig_rows)}) ≠ "
        f"build_dataset valid 수({valid_count})"
    )

    features_norm_full = np.empty_like(features_raw)
    features_norm_full[orig_rows] = ds["features"]
    targets_norm_full = {}
    for p, y_norm in ds["targets"].items():
        full = np.zeros(len(features_raw), dtype=np.float32)
        full[orig_rows] = y_norm
        targets_norm_full[p] = full

    # -------- 3) CV 분할 (sample-wise) ----------
    splits = create_cv_splits(ds["sample_ids"])

    # 분할 마스크는 valid row 상의 index → 전역 row 로 매핑
    # train/val row = orig_rows[train_mask] 같은 식
    config.LSTM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.LSTM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_norm_params(ds["norm_params"], config.FEATURES_DIR / "normalization.json")

    training_log = {}
    for prop in config.TARGET_PROPERTIES:
        if prop not in ds["targets"]:
            continue
        short = config.TARGET_SHORT[prop]
        print(f"\n[L5] === {short} ({prop}) ===")
        fold_summary = []
        target_norm = targets_norm_full[prop]

        for fold, (train_mask, val_mask) in enumerate(splits):
            tr_rows = orig_rows[train_mask]
            va_rows = orig_rows[val_mask]
            print(f"\n  fold {fold + 1}/{config.N_FOLDS}  "
                  f"train={len(tr_rows)} val={len(va_rows)}")

            result = _train_one_fold(
                features_norm_full, target_norm, stacks_h5,
                tr_rows, va_rows, in_channels, device,
            )

            out_path = (
                config.LSTM_MODELS_DIR / f"vppm_lstm_{short}_fold{fold}.pt"
            )
            torch.save(result["best_state"], out_path)
            print(f"    saved → {out_path.name}  "
                  f"(best_val={result['best_val_loss']:.4f}, "
                  f"epochs={result['epochs']})")
            fold_summary.append({
                "fold": fold,
                "best_val_loss": float(result["best_val_loss"]),
                "epochs": int(result["epochs"]),
                "history": result["history"],
            })

        # best fold 포인터
        best = min(fold_summary, key=lambda r: r["best_val_loss"])
        best_ptr = {
            "best_fold": int(best["fold"]),
            "best_val_loss": float(best["best_val_loss"]),
            "n_folds": config.N_FOLDS,
        }
        with open(config.LSTM_MODELS_DIR / f"vppm_lstm_{short}_best.json", "w") as f:
            json.dump(best_ptr, f, indent=2)

        training_log[short] = {
            "fold_val_losses": [r["best_val_loss"] for r in fold_summary],
            "fold_epochs": [r["epochs"] for r in fold_summary],
            "best_fold": best["fold"],
        }

    with open(config.LSTM_RESULTS_DIR / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\n[L5] complete. models → {config.LSTM_MODELS_DIR}")
    return training_log
