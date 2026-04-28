"""Phase L4 — 37 차원 (21 + 16 LSTM) 입력으로 VPPM 재학습.

기존 origin 의 train_all / evaluate_all 을 그대로 재사용. n_feats 를 21 + LSTM_D_EMBED 로 지정.
산출물: Sources/pipeline_outputs/{models_lstm, results/vppm_lstm}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..common import config
from ..common.dataset import build_dataset, save_norm_params
from ..origin.evaluate import evaluate_all, save_metrics
from ..origin.train import train_all


def main():
    parser = argparse.ArgumentParser(description="Phase L4 — VPPM 재학습 (21 + LSTM 임베딩)")
    parser.add_argument("--mode", choices=list(config.LSTM_VALID_MODES), required=True,
                        help="fwd1 | bidir1 | fwd16 | bidir16")
    parser.add_argument("--features-npz", default=None,
                        help="(생략 시) config.lstm_paths(mode)['features_npz']")
    parser.add_argument("--models-dir", default=None,
                        help="(생략 시) config.lstm_paths(mode)['models_dir']")
    parser.add_argument("--results-dir", default=None,
                        help="(생략 시) config.lstm_paths(mode)['results_dir']")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    paths = config.lstm_paths(args.mode)
    feat_path = Path(args.features_npz) if args.features_npz else paths["features_npz"]
    models_dir = Path(args.models_dir) if args.models_dir else paths["models_dir"]
    results_dir = Path(args.results_dir) if args.results_dir else paths["results_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[L4] mode={args.mode}  features={feat_path.name}  results={results_dir.name}")

    print(f"[L4] loading {feat_path} ...")
    data = np.load(feat_path, allow_pickle=True)
    features = data["features"]
    sample_ids = data["sample_ids"]
    targets = {p: data[f"target_{p}"] for p in config.TARGET_PROPERTIES
               if f"target_{p}" in data.files}
    build_ids = data["build_ids"] if "build_ids" in data.files else None
    n_feats = features.shape[1]
    print(f"[L4] features: {features.shape}, targets: {list(targets.keys())}, n_feats={n_feats}")

    dataset = build_dataset(features, sample_ids, targets, build_ids)
    save_norm_params(dataset["norm_params"], models_dir / "normalization.json")

    import torch as _torch
    device = args.device or ("cuda" if _torch.cuda.is_available() else "cpu")
    print(f"[L4] device={device}")

    print(f"[L4] training 4 props × 5 folds (n_feats={n_feats}) → {models_dir}")
    train_all(dataset, output_dir=models_dir, n_feats=n_feats, device=device)

    print(f"[L4] evaluating → {results_dir}")
    results = evaluate_all(dataset, models_dir=models_dir, n_feats=n_feats, device=device)
    save_metrics(results, output_dir=results_dir)
    try:
        from ..origin.evaluate import plot_correlation, plot_scatter_uts
        plot_correlation(results, output_dir=results_dir)
        plot_scatter_uts(results, output_dir=results_dir)
    except Exception as e:
        print(f"[L4] plot skipped: {e}")

    # baseline 비교 한 줄
    base_metrics = config.RESULTS_DIR / "metrics_raw.json"
    if base_metrics.exists():
        import json
        b = json.loads(base_metrics.read_text())
        print(f"\n[L4] === Baseline (21-feat) vs LSTM-{args.mode} ({n_feats}-feat) RMSE ===")
        for s in ["YS", "UTS", "UE", "TE"]:
            b_rmse = b[s]["vppm_rmse_mean"]
            v_rmse = results[s]["vppm_rmse_mean"] if s in results else float("nan")
            delta = v_rmse - b_rmse
            print(f"  {s}: baseline {b_rmse:.2f} → lstm-{args.mode} {v_rmse:.2f}  ΔRMSE={delta:+.2f}")
    print(f"\n[L4] ✓ done. results → {results_dir}")


if __name__ == "__main__":
    main()
