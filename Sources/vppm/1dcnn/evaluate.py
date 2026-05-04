"""VPPM-1DCNN 평가 — baseline ``baseline/evaluate.py`` 미러.

baseline 과 동일한 메트릭/시각화:
  - per-sample min 집계 (보수적 추정, 논문 Section 3.1)
  - 5-fold RMSE 평균 ± 표준편차, naive baseline 대비 reduction%
  - correlation_plots.png, scatter_plot_uts.png
  - per-build (B1.1~B1.5) RMSE 분해

차이점은 모델 forward 시그니처 / 입력 텐서 shape 만:
    pred = model(x: (B, 21, 70))
plot/save 함수는 baseline 의 것을 그대로 재사용.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..baseline.evaluate import plot_correlation, plot_scatter_uts, save_metrics  # noqa: F401
from ..common import config as common_config
from ..common.dataset import create_cv_splits, denormalize
from . import config as exp_config
from .model import VPPM_1DCNN


BUILD_LABELS = {0: "B1.1", 1: "B1.2", 2: "B1.3", 3: "B1.4", 4: "B1.5"}


def _evaluate_fold(
    model: VPPM_1DCNN,
    feats: np.ndarray, targets_raw: np.ndarray,
    sample_ids: np.ndarray, build_ids: np.ndarray | None,
    norm_params: dict, prop: str,
    device: str = "cpu",
    batch_size: int = 1024,
) -> dict:
    """단일 fold val set 예측 → 역정규화 → per-sample min 집계 → RMSE."""
    model.eval()
    N = len(feats)
    preds_norm = np.empty(N, dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            i1 = min(i0 + batch_size, N)
            x = torch.from_numpy(feats[i0:i1]).float().to(device)
            out = model(x).cpu().numpy().flatten()
            preds_norm[i0:i1] = out

    t_min = norm_params["target_min"][prop]
    t_max = norm_params["target_max"][prop]
    pred_raw = denormalize(preds_norm, t_min, t_max)

    # per-sample min 집계 — 보수적 추정 (논문 Section 3.1)
    per_sample_pred: dict[int, list[float]] = {}
    per_sample_true: dict[int, float] = {}
    per_sample_bid: dict[int, int] = {}
    for i, sid in enumerate(sample_ids):
        sid = int(sid)
        per_sample_pred.setdefault(sid, []).append(float(pred_raw[i]))
        per_sample_true[sid] = float(targets_raw[i])
        if build_ids is not None:
            per_sample_bid[sid] = int(build_ids[i])

    sids_sorted = sorted(per_sample_pred.keys())
    preds = np.array([min(per_sample_pred[s]) for s in sids_sorted], dtype=np.float64)
    trues = np.array([per_sample_true[s] for s in sids_sorted], dtype=np.float64)
    bids_arr = (
        np.array([per_sample_bid[s] for s in sids_sorted], dtype=np.int32)
        if per_sample_bid else None
    )
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))

    return {
        "rmse": rmse,
        "predictions": preds,
        "ground_truths": trues,
        "sample_ids": np.array(sids_sorted, dtype=np.int32),
        "build_ids": bids_arr,
    }


def evaluate_all(dataset: dict,
                 models_dir: Path = exp_config.MODELS_DIR,
                 device: str = "cpu") -> dict:
    feats = dataset["features"]      # (N, 21, 70)
    sids = dataset["sample_ids"]
    bids = dataset.get("build_ids")
    splits = create_cv_splits(sids)
    norm_params = dataset["norm_params"]

    results = {}
    for prop in common_config.TARGET_PROPERTIES:
        short = common_config.TARGET_SHORT[prop]
        if prop not in dataset["targets_raw"]:
            continue
        targets_raw = dataset["targets_raw"][prop]

        fold_rmses = []
        all_preds, all_trues, all_bids = [], [], []
        for fold, (_, val_mask) in enumerate(splits):
            mp = Path(models_dir) / f"vppm_1dcnn_{short}_fold{fold}.pt"
            if not mp.exists():
                print(f"  warn: {mp} 없음, skip")
                continue
            model = VPPM_1DCNN()
            model.load_state_dict(torch.load(mp, weights_only=True))
            model.to(device)

            fr = _evaluate_fold(
                model,
                feats[val_mask], targets_raw[val_mask],
                sids[val_mask], bids[val_mask] if bids is not None else None,
                norm_params, prop, device,
            )
            fold_rmses.append(fr["rmse"])
            all_preds.extend(fr["predictions"].tolist())
            all_trues.extend(fr["ground_truths"].tolist())
            if fr["build_ids"] is not None:
                all_bids.extend(fr["build_ids"].tolist())

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
            "all_build_ids": np.array(all_bids, dtype=np.int32) if all_bids else None,
        }
        print(f"\n{short}:")
        print(f"  VPPM-1DCNN RMSE: {mean_rmse:.2f} +/- {std_rmse:.2f}")
        print(f"  Naive RMSE:      {naive_rmse:.2f}")
        print(f"  Reduction:       {reduction:.2f}  ({reduction/naive_rmse*100:.0f}%)")

    return results


def save_per_build_rmse(results: dict, output_dir: Path = exp_config.RESULTS_DIR) -> None:
    """빌드별 RMSE 분해 → per_build_rmse.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for prop, res in results.items():
        short = common_config.TARGET_SHORT[prop]
        bids = res.get("all_build_ids")
        if bids is None:
            continue
        preds = res["all_predictions"]
        trues = res["all_ground_truths"]
        per_build = {}
        for bid in sorted(set(int(b) for b in bids)):
            mask = bids == bid
            if not mask.any():
                continue
            rmse = float(np.sqrt(np.mean((preds[mask] - trues[mask]) ** 2)))
            per_build[BUILD_LABELS.get(bid, str(bid))] = {
                "rmse": rmse,
                "n_samples": int(mask.sum()),
            }
        out[short] = per_build

    path = output_dir / "per_build_rmse.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Per-build RMSE saved to {path}")
