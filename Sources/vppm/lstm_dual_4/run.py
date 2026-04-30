"""VPPM-LSTM-Dual-4 실행 진입점.

전제:
  - baseline `Sources/pipeline_outputs/features/all_features.npz` 가 이미 생성됨
  - visible/0 캐시 `experiments/vppm_lstm/cache/crop_stacks_B1.x.h5` 존재
  - visible/1 캐시 `experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.x.h5` 존재
    (없으면 `python -m Sources.vppm.lstm_dual.run --phase cache_v1` 먼저 실행)

캐시는 dual 와 공유 — dual_4 는 별도 캐시를 빌드하지 않는다.

Usage:
    python -m Sources.vppm.lstm_dual_4.run --phase train      # 학습 (4 props × 5 folds)
    python -m Sources.vppm.lstm_dual_4.run --phase evaluate   # 평가만
    python -m Sources.vppm.lstm_dual_4.run --all              # train → evaluate

Smoke test:
    python -m Sources.vppm.lstm_dual_4.run --all --quick      # epochs=20, patience=10
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
from Sources.vppm.lstm_dual.crop_stacks_v1 import verify_v0_v1_consistency
from Sources.vppm.lstm_dual_4.dataset import build_normalized_dataset, load_dual_dataset
from Sources.vppm.lstm_dual_4.evaluate import evaluate_all
from Sources.vppm.lstm_dual_4.train import train_all


def _ensure_dirs():
    for d in (
        config.LSTM_DUAL_4_EXPERIMENT_DIR,
        config.LSTM_DUAL_4_MODELS_DIR,
        config.LSTM_DUAL_4_RESULTS_DIR,
        config.LSTM_DUAL_4_FEATURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def run_train(device: str, build_ids: list[str]):
    print("\n[D2 train] verify caches")
    verify_v0_v1_consistency(
        build_ids,
        cache_v0_dir=config.LSTM_DUAL_4_CACHE_V0_DIR,
        cache_v1_dir=config.LSTM_DUAL_4_CACHE_V1_DIR,
    )

    print("\n[D2 train] loading raw → normalize → 5-fold train")
    raw = load_dual_dataset(
        features_npz=config.FEATURES_DIR / "all_features.npz",
        cache_v0_dir=config.LSTM_DUAL_4_CACHE_V0_DIR,
        cache_v1_dir=config.LSTM_DUAL_4_CACHE_V1_DIR,
        build_ids=build_ids,
    )
    dataset = build_normalized_dataset(raw)

    n_valid = len(dataset["features"])
    n_samples = len(np.unique(dataset["sample_ids"]))
    n_total = (config.N_FEATURES
               + config.LSTM_DUAL_4_D_EMBED_V0
               + config.LSTM_DUAL_4_D_EMBED_V1)
    print(f"Dataset: {n_valid} SVs, {n_samples} unique samples")
    print(f"  feature dim: {dataset['features'].shape[1]} + "
          f"{config.LSTM_DUAL_4_D_EMBED_V0} (v0) + "
          f"{config.LSTM_DUAL_4_D_EMBED_V1} (v1) = {n_total}")
    print(f"  stack_v0 shape: {dataset['stacks_v0'].shape}")
    print(f"  stack_v1 shape: {dataset['stacks_v1'].shape}")
    print(f"  lengths range [{int(dataset['lengths'].min())}, "
          f"{int(dataset['lengths'].max())}], "
          f"median={int(np.median(dataset['lengths']))}")

    save_norm_params(dataset["norm_params"], config.LSTM_DUAL_4_FEATURES_DIR / "normalization.json")

    train_all(dataset, output_dir=config.LSTM_DUAL_4_MODELS_DIR, device=device)

    print("\n[D3 evaluate]")
    results = evaluate_all(dataset, models_dir=config.LSTM_DUAL_4_MODELS_DIR, device=device)
    save_metrics(results, output_dir=config.LSTM_DUAL_4_RESULTS_DIR)
    plot_correlation(results, output_dir=config.LSTM_DUAL_4_RESULTS_DIR)
    plot_scatter_uts(results, output_dir=config.LSTM_DUAL_4_RESULTS_DIR)


def run_evaluate(device: str, build_ids: list[str]):
    raw = load_dual_dataset(
        features_npz=config.FEATURES_DIR / "all_features.npz",
        cache_v0_dir=config.LSTM_DUAL_4_CACHE_V0_DIR,
        cache_v1_dir=config.LSTM_DUAL_4_CACHE_V1_DIR,
        build_ids=build_ids,
    )
    dataset = build_normalized_dataset(raw)
    results = evaluate_all(dataset, models_dir=config.LSTM_DUAL_4_MODELS_DIR, device=device)
    save_metrics(results, output_dir=config.LSTM_DUAL_4_RESULTS_DIR)
    plot_correlation(results, output_dir=config.LSTM_DUAL_4_RESULTS_DIR)
    plot_scatter_uts(results, output_dir=config.LSTM_DUAL_4_RESULTS_DIR)


def _save_experiment_meta():
    meta = {
        "n_baseline_feats": config.N_FEATURES,
        "d_embed_v0": config.LSTM_DUAL_4_D_EMBED_V0,
        "d_embed_v1": config.LSTM_DUAL_4_D_EMBED_V1,
        "n_total_feats": (config.N_FEATURES
                          + config.LSTM_DUAL_4_D_EMBED_V0
                          + config.LSTM_DUAL_4_D_EMBED_V1),
        "T_max": config.LSTM_T_MAX,
        "crop_h": config.LSTM_CROP_H,
        "crop_w": config.LSTM_CROP_W,
        "camera_channels": [config.LSTM_CAMERA_CHANNEL, config.LSTM_DUAL_CAMERA_CHANNEL_V1],
        "share_cnn": config.LSTM_DUAL_SHARE_CNN,
        "share_lstm": config.LSTM_DUAL_SHARE_LSTM,
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
    with open(config.LSTM_DUAL_4_EXPERIMENT_DIR / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="VPPM-LSTM-Dual-4 Pipeline")
    parser.add_argument(
        "--phase", choices=["train", "evaluate", "all"],
        help="실행 단계",
    )
    parser.add_argument("--all", action="store_true",
                        help="train → evaluate 일괄")
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

    if args.all or args.phase == "train":
        run_train(device, args.builds)
    if (not args.all) and args.phase == "evaluate":
        run_evaluate(device, args.builds)


if __name__ == "__main__":
    main()
