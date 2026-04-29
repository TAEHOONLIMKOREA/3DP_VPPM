"""
Phase 6: 평가 및 시각화
논문 Section 3.1, 3.2 — RMS 오차, correlation plots, scatter plots
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from ..common import config
from ..common.model import VPPM
from ..common.dataset import (
    VPPMDataset, create_cv_splits, denormalize, normalize,
)


def evaluate_fold(model: VPPM, features: np.ndarray, targets_raw: np.ndarray,
                  sample_ids: np.ndarray, norm_params: dict,
                  prop: str, device: str = "cpu") -> dict:
    """단일 fold 평가

    Returns:
        dict with per-sample predictions and ground truths
    """
    model.eval()
    x = torch.from_numpy(features).float().to(device)

    with torch.no_grad():
        pred_norm = model(x).cpu().numpy().flatten()

    # 역정규화
    t_min = norm_params["target_min"][prop]
    t_max = norm_params["target_max"][prop]
    pred_raw = denormalize(pred_norm, t_min, t_max)

    # 샘플별 최소값 취합 (보수적 추정, 논문 Section 3.1)
    per_sample_pred = {}
    per_sample_true = {}
    for i, sid in enumerate(sample_ids):
        sid = int(sid)
        if sid not in per_sample_pred:
            per_sample_pred[sid] = []
        per_sample_pred[sid].append(pred_raw[i])
        per_sample_true[sid] = targets_raw[i]

    final_pred = {sid: min(preds) for sid, preds in per_sample_pred.items()}
    sample_ids_list = sorted(final_pred.keys())
    preds = np.array([final_pred[s] for s in sample_ids_list])
    trues = np.array([per_sample_true[s] for s in sample_ids_list])

    rmse = np.sqrt(np.mean((preds - trues) ** 2))

    return {
        "rmse": rmse,
        "predictions": preds,
        "ground_truths": trues,
        "sample_ids": np.array(sample_ids_list),
    }


def evaluate_all(dataset: dict, models_dir: Path = config.MODELS_DIR,
                 n_feats: int = config.N_FEATURES,
                 device: str = "cpu") -> dict:
    """전체 평가: 5-fold CV 평균 RMS 오차"""
    features = dataset["features"][:, :n_feats]
    sample_ids = dataset["sample_ids"]
    splits = create_cv_splits(sample_ids)
    norm_params = dataset["norm_params"]
    results = {}

    for prop in config.TARGET_PROPERTIES:
        short = config.TARGET_SHORT[prop]
        if prop not in dataset["targets_raw"]:
            continue

        targets_raw = dataset["targets_raw"][prop]
        fold_rmses = []
        all_preds = []
        all_trues = []

        for fold, (train_mask, val_mask) in enumerate(splits):
            model_path = models_dir / f"vppm_{short}_fold{fold}.pt"
            if not model_path.exists():
                print(f"  Warning: {model_path} not found, skipping")
                continue

            model = VPPM(n_feats=n_feats)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.to(device)

            fold_result = evaluate_fold(
                model, features[val_mask], targets_raw[val_mask],
                sample_ids[val_mask], norm_params, prop, device,
            )
            fold_rmses.append(fold_result["rmse"])
            all_preds.extend(fold_result["predictions"].tolist())
            all_trues.extend(fold_result["ground_truths"].tolist())

        if not fold_rmses:
            continue

        # Naive 예측: 전체 평균
        naive_rmse = np.sqrt(np.mean((np.mean(targets_raw) - targets_raw) ** 2))

        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        reduction = naive_rmse - mean_rmse

        results[prop] = {
            "vppm_rmse_mean": float(mean_rmse),
            "vppm_rmse_std": float(std_rmse),
            "naive_rmse": float(naive_rmse),
            "reduction": float(reduction),
            "reduction_pct": float(reduction / naive_rmse * 100),
            "fold_rmses": [float(r) for r in fold_rmses],
            "all_predictions": np.array(all_preds),
            "all_ground_truths": np.array(all_trues),
        }

        print(f"\n{short}:")
        print(f"  VPPM RMSE: {mean_rmse:.1f} +/- {std_rmse:.1f}")
        print(f"  Naive RMSE: {naive_rmse:.1f}")
        print(f"  Reduction: {reduction:.1f} ({reduction/naive_rmse*100:.0f}%)")

    return results


def save_metrics(results: dict, output_dir: Path = config.RESULTS_DIR):
    """메트릭 요약(JSON) + 원본 수치(JSON) + per-sample 예측(CSV) 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    raw = {}
    for prop, res in results.items():
        short = config.TARGET_SHORT[prop]
        summary[short] = {
            "vppm_rmse": f"{res['vppm_rmse_mean']:.1f} +/- {res['vppm_rmse_std']:.1f}",
            "naive_rmse": f"{res['naive_rmse']:.1f}",
            "reduction_pct": f"{res['reduction_pct']:.0f}%",
        }
        raw[short] = {
            "property": prop,
            "vppm_rmse_mean": res["vppm_rmse_mean"],
            "vppm_rmse_std": res["vppm_rmse_std"],
            "naive_rmse": res["naive_rmse"],
            "reduction": res["reduction"],
            "reduction_pct": res["reduction_pct"],
            "fold_rmses": res["fold_rmses"],
            "n_samples": int(len(res["all_predictions"])),
        }

        preds = np.asarray(res["all_predictions"])
        trues = np.asarray(res["all_ground_truths"])
        csv_path = output_dir / f"predictions_{short}.csv"
        with open(csv_path, "w") as f:
            f.write("ground_truth,prediction,residual\n")
            for t, p in zip(trues, preds):
                f.write(f"{t},{p},{p - t}\n")

    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics summary saved to {summary_path}")

    raw_path = output_dir / "metrics_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"Metrics (raw, full precision) saved to {raw_path}")
    print(f"Per-sample predictions saved to {output_dir}/predictions_*.csv")


def plot_correlation(results: dict, output_dir: Path = config.RESULTS_DIR):
    """예측 vs 실측 2D 히스토그램 (논문 Figure 17 재현)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    props = config.TARGET_PROPERTIES

    # 축 하한 고정, 상한은 데이터에서 산출
    axis_lower = {"YS": 177, "UTS": 129, "UE": 0.1, "TE": 4.2}

    for idx, prop in enumerate(props):
        if prop not in results:
            continue
        ax = axes[idx // 2, idx % 2]
        preds = results[prop]["all_predictions"]
        trues = results[prop]["all_ground_truths"]
        short = config.TARGET_SHORT[prop]
        unit = "MPa" if "strength" in prop else "%"

        lo = axis_lower.get(short, 0)
        hi = max(trues.max(), preds.max()) * 1.02
        lims = [lo, hi]

        ax.set_facecolor("black")
        ax.hist2d(trues, preds, bins=80, range=[lims, lims], cmap="hot",
                  cmin=1)
        ax.plot(lims, lims, "w--", alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Ground Truth ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.set_title(short)
        ax.set_aspect("equal")

    plt.tight_layout()
    path = output_dir / "correlation_plots.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Correlation plots saved to {path}")


def plot_scatter_uts(results: dict, build_ids: np.ndarray = None,
                     output_dir: Path = config.RESULTS_DIR):
    """UTS 예측 vs 실측 산점도 (논문 Figure 18 재현)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    prop = "ultimate_tensile_strength"
    if prop not in results:
        return

    preds = results[prop]["all_predictions"]
    trues = results[prop]["all_ground_truths"]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(trues, preds, alpha=0.3, s=10)
    lo, hi = 129, max(trues.max(), preds.max()) * 1.02
    lims = [lo, hi]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Ground Truth UTS (MPa)")
    ax.set_ylabel("Predicted UTS (MPa)")
    ax.set_title("UTS: VPPM Predictions vs Ground Truth")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    path = output_dir / "scatter_plot_uts.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"UTS scatter plot saved to {path}")
