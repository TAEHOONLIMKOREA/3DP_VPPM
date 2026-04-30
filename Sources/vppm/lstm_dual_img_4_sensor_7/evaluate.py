"""Phase S4 — VPPM-LSTM-Dual-Img-4-Sensor-7 평가.

`lstm_dual_4/evaluate.py` 와 동일 골격. 차이는 모델 클래스 / 파일 prefix / sensor 입력.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..baseline.evaluate import plot_correlation, plot_scatter_uts, save_metrics  # noqa: F401
from ..common import config
from ..common.dataset import create_cv_splits, denormalize
from .model import VPPM_LSTM_Dual_Img_4_Sensor_7
from .train import MODEL_FILE_PREFIX


def _evaluate_fold(model: VPPM_LSTM_Dual_Img_4_Sensor_7,
                   feats: np.ndarray,
                   stacks_v0: np.ndarray, stacks_v1: np.ndarray,
                   sensors: np.ndarray,
                   lengths: np.ndarray,
                   targets_raw: np.ndarray, sample_ids: np.ndarray,
                   norm_params: dict, prop: str,
                   device: str = "cpu", batch_size: int = 1024) -> dict:
    model.eval()
    N = len(feats)
    preds_norm = np.empty(N, dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            i1 = min(i0 + batch_size, N)
            f = torch.from_numpy(feats[i0:i1]).float().to(device)
            s0 = torch.from_numpy(stacks_v0[i0:i1]).float().to(device)
            s1 = torch.from_numpy(stacks_v1[i0:i1]).float().to(device)
            sn = torch.from_numpy(sensors[i0:i1]).float().to(device)
            l = torch.from_numpy(lengths[i0:i1].astype(np.int64))
            out = model(f, s0, s1, sn, l).cpu().numpy().flatten()
            preds_norm[i0:i1] = out

    t_min = norm_params["target_min"][prop]
    t_max = norm_params["target_max"][prop]
    pred_raw = denormalize(preds_norm, t_min, t_max)

    per_sample_pred: dict[int, list[float]] = {}
    per_sample_true: dict[int, float] = {}
    for i, sid in enumerate(sample_ids):
        sid = int(sid)
        per_sample_pred.setdefault(sid, []).append(float(pred_raw[i]))
        per_sample_true[sid] = float(targets_raw[i])

    sample_ids_sorted = sorted(per_sample_pred.keys())
    preds = np.array([min(per_sample_pred[s]) for s in sample_ids_sorted], dtype=np.float64)
    trues = np.array([per_sample_true[s] for s in sample_ids_sorted], dtype=np.float64)
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))

    return {
        "rmse": rmse,
        "predictions": preds,
        "ground_truths": trues,
        "sample_ids": np.array(sample_ids_sorted, dtype=np.int32),
    }


def evaluate_all(dataset: dict,
                 models_dir: Path = config.LSTM_DUAL_IMG_4_SENSOR_7_MODELS_DIR,
                 device: str = "cpu") -> dict:
    feats = dataset["features"]
    sv0 = dataset["stacks_v0"]
    sv1 = dataset["stacks_v1"]
    sn = dataset["sensors"]
    lengths = dataset["lengths"]
    sids = dataset["sample_ids"]
    splits = create_cv_splits(sids)
    norm_params = dataset["norm_params"]

    results = {}
    for prop in config.TARGET_PROPERTIES:
        short = config.TARGET_SHORT[prop]
        if prop not in dataset["targets_raw"]:
            continue
        targets_raw = dataset["targets_raw"][prop]

        fold_rmses = []
        all_preds, all_trues = [], []
        for fold, (_, val_mask) in enumerate(splits):
            mp = Path(models_dir) / f"{MODEL_FILE_PREFIX}_{short}_fold{fold}.pt"
            if not mp.exists():
                print(f"  warn: {mp} 없음, skip")
                continue
            model = VPPM_LSTM_Dual_Img_4_Sensor_7()
            model.load_state_dict(torch.load(mp, weights_only=True))
            model.to(device)

            fr = _evaluate_fold(
                model,
                feats[val_mask], sv0[val_mask], sv1[val_mask], sn[val_mask],
                lengths[val_mask], targets_raw[val_mask], sids[val_mask],
                norm_params, prop, device,
            )
            fold_rmses.append(fr["rmse"])
            all_preds.extend(fr["predictions"].tolist())
            all_trues.extend(fr["ground_truths"].tolist())

        if not fold_rmses:
            continue

        naive_rmse = float(np.sqrt(np.mean((targets_raw.mean() - targets_raw) ** 2)))
        mean_rmse = float(np.mean(fold_rmses))
        std_rmse = float(np.std(fold_rmses))
        reduction = naive_rmse - mean_rmse

        results[prop] = {
            "vppm_rmse_mean": mean_rmse,
            "vppm_rmse_std": std_rmse,
            "naive_rmse": naive_rmse,
            "reduction": reduction,
            "reduction_pct": reduction / naive_rmse * 100 if naive_rmse > 0 else 0.0,
            "fold_rmses": [float(r) for r in fold_rmses],
            "all_predictions": np.array(all_preds),
            "all_ground_truths": np.array(all_trues),
        }
        print(f"\n{short}:")
        print(f"  VPPM-LSTM-Dual-Img-4-Sensor-7 RMSE: {mean_rmse:.2f} ± {std_rmse:.2f}")
        print(f"  Naive RMSE:                         {naive_rmse:.2f}")
        print(f"  Reduction:                          {reduction:.2f}  ({reduction/naive_rmse*100:.0f}%)")

    return results
