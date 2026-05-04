"""VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-Sensor-1 학습.

4 properties (YS / UTS / UE / TE) × 5 folds = 20 모델.
forward 시그니처: (feats_static, stacks_v0, stacks_v1, sensors, dscnn, cad_patch, scan_patch, lengths).

fullstack `_1dcnn_sensor_4/train.py` 와 동일 골격. 차이는 (1) 모델 클래스
`VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_Sensor_1`, (2) 저장 prefix.
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
from .dataset import VPPMLSTMSeptetDataset, collate_fn
from .model import VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_Sensor_1

MODEL_FILE_PREFIX = "vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1"


def train_single_fold(
    fs_train: np.ndarray, sv0_train: np.ndarray, sv1_train: np.ndarray,
    sn_train: np.ndarray, dn_train: np.ndarray,
    cad_train: np.ndarray, scan_train: np.ndarray,
    lengths_train: np.ndarray, targets_train: np.ndarray,
    fs_val: np.ndarray, sv0_val: np.ndarray, sv1_val: np.ndarray,
    sn_val: np.ndarray, dn_val: np.ndarray,
    cad_val: np.ndarray, scan_val: np.ndarray,
    lengths_val: np.ndarray, targets_val: np.ndarray,
    device: str = "cpu",
    grad_clip: float = config.LSTM_GRAD_CLIP,
) -> dict:
    train_ds = VPPMLSTMSeptetDataset(
        fs_train, sv0_train, sv1_train, sn_train, dn_train,
        cad_train, scan_train, lengths_train, targets_train,
    )
    val_ds = VPPMLSTMSeptetDataset(
        fs_val, sv0_val, sv1_val, sn_val, dn_val,
        cad_val, scan_val, lengths_val, targets_val,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.LSTM_BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=config.LSTM_NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.LSTM_BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=config.LSTM_NUM_WORKERS,
    )

    model = VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_Sensor_1().to(device)
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
        model.train()
        train_losses = []
        for fs, sv0, sv1, sn, dn, cad, scan, lengths, ys in train_loader:
            fs = fs.to(device, non_blocking=True)
            sv0 = sv0.to(device, non_blocking=True)
            sv1 = sv1.to(device, non_blocking=True)
            sn = sn.to(device, non_blocking=True)
            dn = dn.to(device, non_blocking=True)
            cad = cad.to(device, non_blocking=True)
            scan = scan.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(fs, sv0, sv1, sn, dn, cad, scan, lengths)
            loss = criterion(pred, ys)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for fs, sv0, sv1, sn, dn, cad, scan, lengths, ys in val_loader:
                fs = fs.to(device, non_blocking=True)
                sv0 = sv0.to(device, non_blocking=True)
                sv1 = sv1.to(device, non_blocking=True)
                sn = sn.to(device, non_blocking=True)
                dn = dn.to(device, non_blocking=True)
                cad = cad.to(device, non_blocking=True)
                scan = scan.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True)
                pred = model(fs, sv0, sv1, sn, dn, cad, scan, lengths)
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


def train_all(
    dataset: dict,
    output_dir: Path = config.LSTM_FULL59_MODELS_DIR,
    device: str = "cpu",
    prop_filter: str | None = None,
) -> dict:
    """4 properties × 5 fold 학습. 모델 저장 → output_dir/{prefix}_{short}_fold{k}.pt.

    prop_filter:
      None | "all"        → 모든 property 학습
      "YS"|"UTS"|"UE"|"TE" → 해당 short 매칭되는 prop 만 학습 (4-GPU 병렬용)
    """
    if prop_filter is not None and prop_filter.lower() != "all":
        valid_shorts = {config.TARGET_SHORT[p] for p in config.TARGET_PROPERTIES}
        if prop_filter not in valid_shorts:
            raise ValueError(
                f"prop_filter={prop_filter!r} 가 유효하지 않습니다. "
                f"가능: {sorted(valid_shorts)} 또는 'all'"
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fs = dataset["features_static"]
    sv0 = dataset["stacks_v0"]
    sv1 = dataset["stacks_v1"]
    sn = dataset["sensors"]
    dn = dataset["dscnn"]
    cad = dataset["cad_patch"]
    scan = dataset["scan_patch"]
    lengths = dataset["lengths"]
    sids = dataset["sample_ids"]
    splits = create_cv_splits(sids)

    all_results = {}
    for prop in config.TARGET_PROPERTIES:
        short = config.TARGET_SHORT[prop]
        if prop_filter is not None and prop_filter.lower() != "all" and short != prop_filter:
            continue
        if prop not in dataset["targets"]:
            print(f"  skip {short}: no target")
            continue
        targets = dataset["targets"][prop]

        print(f"\n{'='*60}\nTraining {MODEL_FILE_PREFIX} for {short}\n{'='*60}")
        fold_results = []
        for fold, (train_mask, val_mask) in enumerate(splits):
            print(f"  Fold {fold+1}/{config.N_FOLDS}...")
            result = train_single_fold(
                fs[train_mask], sv0[train_mask], sv1[train_mask],
                sn[train_mask], dn[train_mask],
                cad[train_mask], scan[train_mask],
                lengths[train_mask], targets[train_mask],
                fs[val_mask], sv0[val_mask], sv1[val_mask],
                sn[val_mask], dn[val_mask],
                cad[val_mask], scan[val_mask],
                lengths[val_mask], targets[val_mask],
                device=device,
            )
            fold_results.append(result)

            model_path = output_dir / f"{MODEL_FILE_PREFIX}_{short}_fold{fold}.pt"
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
    # prop 별 컨테이너가 동시에 돌면 같은 파일 race → 파일명 분리
    if prop_filter is not None and prop_filter.lower() != "all":
        log_path = output_dir / f"training_log_{prop_filter}.json"
    else:
        log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete → {output_dir}")
    return all_results
