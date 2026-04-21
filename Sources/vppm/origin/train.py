"""
Phase 5: VPPM 학습 파이프라인
논문 Section 2.11 — 5-Fold CV, L1 Loss, Adam, Early Stopping
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from ..common import config
from ..common.model import VPPM
from ..common.dataset import VPPMDataset, create_cv_splits


class EarlyStopper:
    """검증 오차 plateau 감지"""

    def __init__(self, patience: int = config.EARLY_STOP_PATIENCE, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("inf")
        self.counter = 0
        self.best_state = None

    def check(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_single_fold(features_train: np.ndarray, targets_train: np.ndarray,
                      features_val: np.ndarray, targets_val: np.ndarray,
                      n_feats: int = config.N_FEATURES,
                      device: str = "cpu") -> dict:
    """단일 fold 학습

    Returns:
        dict with model state_dict, training history, best val loss
    """
    train_ds = VPPMDataset(features_train, targets_train)
    val_ds = VPPMDataset(features_val, targets_val)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    model = VPPM(n_feats=n_feats).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.ADAM_BETAS,
        eps=config.ADAM_EPS,
    )
    criterion = nn.L1Loss()
    stopper = EarlyStopper(patience=config.EARLY_STOP_PATIENCE)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(config.MAX_EPOCHS):
        # --- Train ---
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_losses.append(loss.item() * len(batch_x))

        train_loss = np.mean(train_losses)
        val_loss = sum(val_losses) / len(val_ds)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        # --- Early stopping ---
        if stopper.check(val_loss, model):
            break

    # 최적 모델 복원
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    return {
        "model_state": model.state_dict(),
        "history": history,
        "best_val_loss": stopper.best_score,
        "epochs": len(history["train_loss"]),
        "n_feats": n_feats,
    }


def train_all(dataset: dict, output_dir: Path = config.MODELS_DIR,
              n_feats: int = config.N_FEATURES, device: str = "cpu"):
    """전체 학습: 4 properties × 5 folds = 20 모델

    Args:
        dataset: build_dataset()의 반환값
        output_dir: 모델 저장 디렉토리
        n_feats: 피처 수 (전체 21 또는 ablation)
        device: "cpu" or "cuda"
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    features = dataset["features"][:, :n_feats]
    sample_ids = dataset["sample_ids"]
    splits = create_cv_splits(sample_ids)
    all_results = {}

    for prop in config.TARGET_PROPERTIES:
        short = config.TARGET_SHORT[prop]
        if prop not in dataset["targets"]:
            print(f"  Skipping {short}: no target data")
            continue

        targets = dataset["targets"][prop]
        fold_results = []
        print(f"\n{'='*50}")
        print(f"Training VPPM for {short} ({prop})")
        print(f"{'='*50}")

        for fold, (train_mask, val_mask) in enumerate(splits):
            print(f"\n  Fold {fold+1}/{config.N_FOLDS}...")
            result = train_single_fold(
                features[train_mask], targets[train_mask],
                features[val_mask], targets[val_mask],
                n_feats=n_feats, device=device,
            )
            fold_results.append(result)

            # 모델 저장
            model_path = output_dir / f"vppm_{short}_fold{fold}.pt"
            torch.save(result["model_state"], model_path)

            print(f"    Epochs: {result['epochs']}, "
                  f"Best val loss: {result['best_val_loss']:.6f}")

        all_results[prop] = fold_results

    # 학습 로그 저장
    log = {}
    for prop, results in all_results.items():
        short = config.TARGET_SHORT[prop]
        log[short] = {
            "fold_val_losses": [r["best_val_loss"] for r in results],
            "fold_epochs": [r["epochs"] for r in results],
        }
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete. Models saved to {output_dir}")
    return all_results
