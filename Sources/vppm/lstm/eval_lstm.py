"""
Phase L6 — VPPM-LSTM 평가 + LSTM 임베딩 export.

- 저장된 20개 fold 모델을 로드해 5-fold CV RMSE 산출
- 기존 evaluate.py 와 동일한 per-sample min 집계
- LSTM 임베딩을 npz 로 export (재평가/ablation 용, ~2MB)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import config
from ..dataset import build_dataset, create_cv_splits, denormalize
from ..model import VPPM_LSTM
from .dataset import VppmLstmDataset, collate, load_aligned_arrays


def _build_full_norm(arr: dict):
    """load_aligned_arrays 결과 → 정규화된 전체 (valid row 기준) 배열."""
    features_raw = arr["features"]
    targets_raw = arr["targets"]
    sample_ids = arr["sample_ids"]
    build_ids = arr["build_ids"]

    ds = build_dataset(features_raw, sample_ids, targets_raw, build_ids)

    uts = targets_raw.get("ultimate_tensile_strength",
                          np.zeros(len(features_raw)))
    valid_mask = ~np.isnan(uts) & (uts >= 50.0)
    for p in config.TARGET_PROPERTIES:
        if p in targets_raw:
            valid_mask &= ~np.isnan(targets_raw[p])
    valid_mask &= ~np.isnan(features_raw).any(axis=1)
    orig_rows = np.where(valid_mask)[0]

    f_full = np.empty_like(features_raw)
    f_full[orig_rows] = ds["features"]

    t_full = {}
    for p, y_norm in ds["targets"].items():
        full = np.zeros(len(features_raw), dtype=np.float32)
        full[orig_rows] = y_norm
        t_full[p] = full

    return ds, orig_rows, f_full, t_full


def _predict_rows(model: VPPM_LSTM, x21: np.ndarray, target_norm: np.ndarray,
                  stacks_h5: str, rows: np.ndarray,
                  device: str) -> np.ndarray:
    loader = DataLoader(
        VppmLstmDataset(x21, target_norm, stacks_h5, rows),
        batch_size=config.LSTM_BATCH_SIZE,
        shuffle=False,
        num_workers=config.LSTM_NUM_WORKERS,
        collate_fn=collate,
    )
    preds = np.empty(len(rows), dtype=np.float32)
    cursor = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            n = len(batch["y"])
            p = model(
                batch["x21"].to(device),
                batch["img"].to(device),
                batch["mask"].to(device),
            ).cpu().numpy().flatten()
            preds[cursor:cursor + n] = p
            cursor += n
    return preds


def evaluate_vppm_lstm(device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[L6] device: {device}")

    arr = load_aligned_arrays()
    ds, orig_rows, f_full, t_full = _build_full_norm(arr)
    splits = create_cv_splits(ds["sample_ids"])
    sample_ids_valid = ds["sample_ids"]
    norm_params = ds["norm_params"]
    stacks_h5 = arr["stacks_h5"]
    in_channels = arr["C"]

    results = {}
    for prop in config.TARGET_PROPERTIES:
        if prop not in ds["targets_raw"]:
            continue
        short = config.TARGET_SHORT[prop]
        print(f"\n[L6] === {short} ({prop}) ===")
        targets_raw = ds["targets_raw"][prop]
        t_min = norm_params["target_min"][prop]
        t_max = norm_params["target_max"][prop]

        fold_rmses = []
        all_preds, all_trues = [], []

        for fold, (train_mask, val_mask) in enumerate(splits):
            model_path = (
                config.LSTM_MODELS_DIR / f"vppm_lstm_{short}_fold{fold}.pt"
            )
            if not model_path.exists():
                print(f"  [warn] {model_path} 없음")
                continue
            model = VPPM_LSTM(
                in_channels=in_channels,
                d_lstm=config.LSTM_D_EMBED,
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device,
                                             weights_only=True))

            va_rows = orig_rows[val_mask]
            preds_norm = _predict_rows(
                model, f_full, t_full[prop], stacks_h5, va_rows, device
            )
            preds_raw = denormalize(preds_norm, t_min, t_max)
            trues_raw = targets_raw[val_mask]
            sids = sample_ids_valid[val_mask]

            # per-sample min 집계 (기존 evaluate.py 와 동일)
            per_sample_pred: dict[int, list[float]] = {}
            per_sample_true: dict[int, float] = {}
            for i, sid in enumerate(sids):
                sid = int(sid)
                per_sample_pred.setdefault(sid, []).append(float(preds_raw[i]))
                per_sample_true[sid] = float(trues_raw[i])
            sid_list = sorted(per_sample_pred.keys())
            p_arr = np.array([min(per_sample_pred[s]) for s in sid_list])
            t_arr = np.array([per_sample_true[s] for s in sid_list])
            rmse = float(np.sqrt(np.mean((p_arr - t_arr) ** 2)))
            fold_rmses.append(rmse)
            all_preds.extend(p_arr.tolist())
            all_trues.extend(t_arr.tolist())
            print(f"  fold {fold}: rmse={rmse:.3f}")

        if not fold_rmses:
            continue

        naive_rmse = float(np.sqrt(np.mean(
            (np.mean(targets_raw) - targets_raw) ** 2
        )))
        mean_rmse = float(np.mean(fold_rmses))
        std_rmse = float(np.std(fold_rmses))
        results[prop] = {
            "vppm_lstm_rmse_mean": mean_rmse,
            "vppm_lstm_rmse_std": std_rmse,
            "naive_rmse": naive_rmse,
            "reduction": naive_rmse - mean_rmse,
            "reduction_pct": (naive_rmse - mean_rmse) / naive_rmse * 100,
            "fold_rmses": fold_rmses,
            "all_predictions": all_preds,
            "all_ground_truths": all_trues,
        }
        print(f"  mean rmse = {mean_rmse:.2f} ± {std_rmse:.2f}  "
              f"(naive={naive_rmse:.2f})")

    # -------- save metrics ----------
    config.LSTM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for prop, r in results.items():
        short = config.TARGET_SHORT[prop]
        summary[short] = {
            "vppm_lstm_rmse": f"{r['vppm_lstm_rmse_mean']:.1f} +/- {r['vppm_lstm_rmse_std']:.1f}",
            "naive_rmse": f"{r['naive_rmse']:.1f}",
            "reduction_pct": f"{r['reduction_pct']:.0f}%",
            "fold_rmses": [round(x, 2) for x in r["fold_rmses"]],
        }
    with open(config.LSTM_RESULTS_DIR / "cv_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[L6] metrics → {config.LSTM_RESULTS_DIR / 'cv_metrics.json'}")

    # correlation plot
    _plot_correlation(results)
    return results


def _plot_correlation(results: dict):
    props = [p for p in config.TARGET_PROPERTIES if p in results]
    if not props:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for idx, prop in enumerate(config.TARGET_PROPERTIES):
        ax = axes[idx // 2, idx % 2]
        if prop not in results:
            ax.axis("off")
            continue
        r = results[prop]
        preds = np.array(r["all_predictions"])
        trues = np.array(r["all_ground_truths"])
        short = config.TARGET_SHORT[prop]
        unit = "MPa" if "strength" in prop else "%"
        ax.hist2d(trues, preds, bins=50, cmap="hot")
        lims = [min(trues.min(), preds.min()), max(trues.max(), preds.max())]
        ax.plot(lims, lims, "w--", alpha=0.7)
        ax.set_xlabel(f"Ground Truth ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.set_title(f"{short}  rmse={r['vppm_lstm_rmse_mean']:.1f}")
        ax.set_aspect("equal")
    plt.tight_layout()
    out = config.LSTM_RESULTS_DIR / "correlation_plots.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[L6] correlation plot → {out}")


def export_lstm_embeddings(device: str | None = None) -> Path:
    """LSTM 임베딩을 전체 슈퍼복셀에 대해 계산해 npz 로 저장 (~2MB).

    UTS best fold 를 LSTM 인코더 소스로 사용 (모든 property 가 동일 CNN/LSTM 구조).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    arr = load_aligned_arrays()
    in_channels = arr["C"]

    # UTS 의 best fold 선택
    ptr_path = config.LSTM_MODELS_DIR / "vppm_lstm_UTS_best.json"
    if not ptr_path.exists():
        raise FileNotFoundError(f"{ptr_path} 없음")
    best = json.loads(ptr_path.read_text())
    fold = int(best["best_fold"])
    model_path = config.LSTM_MODELS_DIR / f"vppm_lstm_UTS_fold{fold}.pt"

    model = VPPM_LSTM(in_channels=in_channels,
                      d_lstm=config.LSTM_D_EMBED).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()

    # 전체 row 에 대해 임베딩 추출
    N = arr["n"]
    x21_dummy = np.zeros((N, config.N_FEATURES), dtype=np.float32)
    y_dummy = np.zeros(N, dtype=np.float32)
    all_rows = np.arange(N, dtype=np.int64)
    loader = DataLoader(
        VppmLstmDataset(x21_dummy, y_dummy, arr["stacks_h5"], all_rows),
        batch_size=config.LSTM_BATCH_SIZE,
        shuffle=False,
        num_workers=config.LSTM_NUM_WORKERS,
        collate_fn=collate,
    )
    out = np.empty((N, config.LSTM_D_EMBED), dtype=np.float32)
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(device)
            mask = batch["mask"].to(device)
            emb = model.embed(img, mask).cpu().numpy()
            n = len(emb)
            out[cursor:cursor + n] = emb
            cursor += n

    config.LSTM_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    tag = arr["channels_name"].replace("+", "_")
    out_path = (
        config.LSTM_EMBEDDINGS_DIR
        / f"lstm_emb_{tag}_d{config.LSTM_D_EMBED}.npz"
    )
    np.savez_compressed(out_path, embeddings=out, source_fold=fold)
    print(f"[L6] embeddings saved → {out_path} "
          f"(shape={out.shape}, {out_path.stat().st_size/1e6:.2f} MB)")
    return out_path
