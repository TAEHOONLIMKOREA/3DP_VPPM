"""
VPPM-LSTM 실행 진입점.

전제: baseline `Sources/pipeline_outputs/features/all_features.npz` 가 이미 생성되어 있어야 함.

Usage:
    python -m Sources.vppm.lstm.run --phase cache              # L1 캐시 빌드 (5 빌드)
    python -m Sources.vppm.lstm.run --phase train              # 학습 (4 props × 5 folds)
    python -m Sources.vppm.lstm.run --phase evaluate           # 평가
    python -m Sources.vppm.lstm.run --all                      # cache → train → evaluate

Smoke test:
    python -m Sources.vppm.lstm.run --all --quick              # epochs=20, patience=10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Sources.vppm.baseline.evaluate import (
    plot_correlation,
    plot_scatter_uts,
    save_metrics,
)
from Sources.vppm.common import config
from Sources.vppm.common.dataset import save_norm_params
from Sources.vppm.lstm.crop_stacks import build_cache
from Sources.vppm.lstm.dataset import build_normalized_dataset, load_lstm_dataset
from Sources.vppm.lstm.evaluate import evaluate_all
from Sources.vppm.lstm.train import train_all


def _ensure_dirs():
    for d in (
        config.LSTM_EXPERIMENT_DIR,
        config.LSTM_CACHE_DIR,
        config.LSTM_MODELS_DIR,
        config.LSTM_RESULTS_DIR,
        config.LSTM_FEATURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def run_cache(build_ids: list[str]):
    print(f"\n[L1 cache] builds={build_ids} → {config.LSTM_CACHE_DIR}")
    build_cache(build_ids, config.LSTM_CACHE_DIR)


def run_train(device: str):
    print("\n[L2 train] loading raw → normalize → 5-fold train")
    raw = load_lstm_dataset(
        features_npz=config.FEATURES_DIR / "all_features.npz",
        cache_dir=config.LSTM_CACHE_DIR,
    )
    dataset = build_normalized_dataset(raw)

    n_valid = len(dataset["features"])
    n_samples = len(np.unique(dataset["sample_ids"]))
    print(f"Dataset: {n_valid} SVs, {n_samples} unique samples")
    print(f"  feature dim: {dataset['features'].shape[1]} + {config.LSTM_D_EMBED} (LSTM embed)")
    print(f"  stack shape: {dataset['stacks'].shape}, lengths range "
          f"[{int(dataset['lengths'].min())}, {int(dataset['lengths'].max())}], "
          f"median={int(np.median(dataset['lengths']))}")

    save_norm_params(dataset["norm_params"], config.LSTM_FEATURES_DIR / "normalization.json")

    train_all(dataset, output_dir=config.LSTM_MODELS_DIR, device=device)

    # 즉시 평가
    print("\n[L3 evaluate]")
    results = evaluate_all(dataset, models_dir=config.LSTM_MODELS_DIR, device=device)
    save_metrics(results, output_dir=config.LSTM_RESULTS_DIR)
    plot_correlation(results, output_dir=config.LSTM_RESULTS_DIR)
    plot_scatter_uts(results, output_dir=config.LSTM_RESULTS_DIR)


def run_evaluate(device: str):
    raw = load_lstm_dataset(
        features_npz=config.FEATURES_DIR / "all_features.npz",
        cache_dir=config.LSTM_CACHE_DIR,
    )
    dataset = build_normalized_dataset(raw)
    results = evaluate_all(dataset, models_dir=config.LSTM_MODELS_DIR, device=device)
    save_metrics(results, output_dir=config.LSTM_RESULTS_DIR)
    plot_correlation(results, output_dir=config.LSTM_RESULTS_DIR)
    plot_scatter_uts(results, output_dir=config.LSTM_RESULTS_DIR)


def _save_experiment_meta():
    meta = {
        "n_baseline_feats": config.N_FEATURES,
        "d_embed": config.LSTM_D_EMBED,
        "n_total_feats": config.N_FEATURES + config.LSTM_D_EMBED,
        "T_max": config.LSTM_T_MAX,
        "crop_h": config.LSTM_CROP_H,
        "crop_w": config.LSTM_CROP_W,
        "camera_channel": config.LSTM_CAMERA_CHANNEL,
        "cnn": {
            "ch1": config.LSTM_CNN_CH1,
            "ch2": config.LSTM_CNN_CH2,
            "kernel": config.LSTM_CNN_KERNEL,
            "d_cnn": config.LSTM_D_CNN,
        },
        "lstm": {
            "d_hidden": config.LSTM_D_HIDDEN,
            "num_layers": config.LSTM_NUM_LAYERS,
            "bidirectional": config.LSTM_BIDIRECTIONAL,
        },
        "train": {
            "lr": config.LSTM_LR,
            "batch_size": config.LSTM_BATCH_SIZE,
            "max_epochs": config.LSTM_MAX_EPOCHS,
            "early_stop_patience": config.LSTM_EARLY_STOP_PATIENCE,
            "grad_clip": config.LSTM_GRAD_CLIP,
            "weight_decay": config.LSTM_WEIGHT_DECAY,
        },
    }
    with open(config.LSTM_EXPERIMENT_DIR / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="VPPM-LSTM Pipeline")
    parser.add_argument(
        "--phase", choices=["cache", "train", "evaluate", "all"],
        help="실행 단계",
    )
    parser.add_argument("--all", action="store_true", help="cache → train → evaluate 일괄")
    parser.add_argument("--builds", nargs="+", default=list(config.BUILDS.keys()))
    parser.add_argument("--device", default=None, help="cpu | cuda (기본: 자동)")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: epochs=20, patience=10")
    args = parser.parse_args()

    if not args.phase and not args.all:
        parser.error("--phase 또는 --all 중 하나는 필수")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _ensure_dirs()

    if args.quick:
        config.LSTM_MAX_EPOCHS = 20
        config.LSTM_EARLY_STOP_PATIENCE = 10
        print("[quick] LSTM_MAX_EPOCHS=20, patience=10")

    _save_experiment_meta()

    if args.all or args.phase == "cache":
        run_cache(args.builds)
    if args.all or args.phase == "train":
        run_train(device)
    if (not args.all) and args.phase == "evaluate":
        run_evaluate(device)


if __name__ == "__main__":
    main()
