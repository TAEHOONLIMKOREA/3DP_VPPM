"""VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-1DCNN-Sensor-4 실행 진입점.

전제:
  - baseline `Sources/pipeline_outputs/features/all_features.npz` 가 이미 생성됨
  - visible/0 캐시 `experiments/vppm_lstm/cache/crop_stacks_B1.x.h5` 존재
  - visible/1 캐시 `experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.x.h5` 존재
  - sensor 캐시 `experiments/vppm_lstm_dual_img_4_sensor_7/cache/sensor_stacks_B1.x.h5` 존재
  - DSCNN 캐시 `experiments/vppm_lstm_dual_img_4_sensor_7_dscnn_8/cache/dscnn_stacks_B1.x.h5` 존재
  - cad_patch / scan_patch 캐시 (없으면 `--phase cache_cad_patch` / `cache_scan_patch` 가 자동 빌드)

Usage:
    # CAD/Scan 패치 캐시 빌드만
    python -m Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.run --phase cache_cad_patch
    python -m Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.run --phase cache_scan_patch

    # 학습 (4 props × 5 folds)
    python -m Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.run --phase train

    # 평가만
    python -m Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.run --phase evaluate

    # 전체 (cache_cad_patch → cache_scan_patch → train → evaluate)
    python -m Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.run --all

Smoke test:
    python -m Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.run --all --quick    # epochs=20, patience=10
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
from Sources.vppm.lstm_dual_img_4_sensor_7.cache_sensor import (
    verify_v0_consistency as verify_v0_sensor_consistency,
)
from Sources.vppm.lstm_dual_img_4_sensor_7_dscnn_8.cache_dscnn import (
    verify_dscnn_v0_consistency,
)
from Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.cache_cad_patch import (
    build_cache as build_cad_patch_cache,
    verify_cad_patch_v0_consistency,
)
from Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.cache_scan_patch import (
    build_cache as build_scan_patch_cache,
    verify_scan_patch_v0_consistency,
)
from Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.dataset import (
    build_normalized_dataset,
    load_septet_dataset,
)
from Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.evaluate import evaluate_all
from Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.train import train_all


def _ensure_dirs():
    for d in (
        config.LSTM_FULL86_EXPERIMENT_DIR,
        config.LSTM_FULL86_CACHE_DIR,
        config.LSTM_FULL86_MODELS_DIR,
        config.LSTM_FULL86_RESULTS_DIR,
        config.LSTM_FULL86_FEATURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def run_cache_cad_patch(build_ids: list[str]):
    print("\n[S1a cache_cad_patch] building CAD 8×8 patch caches (inversion + cad_mask 픽셀곱)")
    build_cad_patch_cache(build_ids, out_dir=config.LSTM_FULL86_CACHE_CAD_DIR)
    verify_cad_patch_v0_consistency(
        build_ids,
        cache_v0_dir=config.LSTM_FULL86_CACHE_V0_DIR,
        cache_cad_dir=config.LSTM_FULL86_CACHE_CAD_DIR,
    )


def run_cache_scan_patch(build_ids: list[str]):
    print("\n[S1b cache_scan_patch] building Scan 8×8 patch caches (raw + NaN→0)")
    build_scan_patch_cache(build_ids, out_dir=config.LSTM_FULL86_CACHE_SCAN_DIR)
    verify_scan_patch_v0_consistency(
        build_ids,
        cache_v0_dir=config.LSTM_FULL86_CACHE_V0_DIR,
        cache_scan_dir=config.LSTM_FULL86_CACHE_SCAN_DIR,
    )


def run_train(device: str, build_ids: list[str]):
    print("\n[S2 train] verify caches (6-way)")
    verify_v0_v1_consistency(
        build_ids,
        cache_v0_dir=config.LSTM_FULL86_CACHE_V0_DIR,
        cache_v1_dir=config.LSTM_FULL86_CACHE_V1_DIR,
    )
    verify_v0_sensor_consistency(
        build_ids,
        cache_v0_dir=config.LSTM_FULL86_CACHE_V0_DIR,
        cache_sensor_dir=config.LSTM_FULL86_CACHE_SENSOR_DIR,
    )
    verify_dscnn_v0_consistency(
        build_ids,
        cache_v0_dir=config.LSTM_FULL86_CACHE_V0_DIR,
        cache_dscnn_dir=config.LSTM_FULL86_CACHE_DSCNN_DIR,
    )
    verify_cad_patch_v0_consistency(
        build_ids,
        cache_v0_dir=config.LSTM_FULL86_CACHE_V0_DIR,
        cache_cad_dir=config.LSTM_FULL86_CACHE_CAD_DIR,
    )
    verify_scan_patch_v0_consistency(
        build_ids,
        cache_v0_dir=config.LSTM_FULL86_CACHE_V0_DIR,
        cache_scan_dir=config.LSTM_FULL86_CACHE_SCAN_DIR,
    )

    print("\n[S2 train] loading raw → normalize → 5-fold train")
    raw = load_septet_dataset(
        features_npz=config.FEATURES_DIR / "all_features.npz",
        cache_v0_dir=config.LSTM_FULL86_CACHE_V0_DIR,
        cache_v1_dir=config.LSTM_FULL86_CACHE_V1_DIR,
        cache_sensor_dir=config.LSTM_FULL86_CACHE_SENSOR_DIR,
        cache_dscnn_dir=config.LSTM_FULL86_CACHE_DSCNN_DIR,
        cache_cad_dir=config.LSTM_FULL86_CACHE_CAD_DIR,
        cache_scan_dir=config.LSTM_FULL86_CACHE_SCAN_DIR,
        build_ids=build_ids,
    )
    dataset = build_normalized_dataset(raw)

    n_valid = len(dataset["features_static"])
    n_samples = len(np.unique(dataset["sample_ids"]))
    n_static = len(config.LSTM_FULL86_STATIC_IDX)                   # 2
    n_total = (
        n_static
        + config.LSTM_FULL86_D_EMBED_V0
        + config.LSTM_FULL86_D_EMBED_V1
        + config.LSTM_FULL86_N_SENSOR_FIELDS * config.LSTM_FULL86_D_PER_SENSOR_FIELD
        + config.LSTM_FULL86_D_EMBED_D
        + config.LSTM_FULL86_D_EMBED_C
        + config.LSTM_FULL86_D_EMBED_SC
    )                                                                # 86
    print(f"Dataset: {n_valid} SVs, {n_samples} unique samples")
    print(f"  static feat dim: {dataset['features_static'].shape[1]} (build_height + laser_module)")
    print(f"  MLP input dim:   "
          f"{n_static} (static) + "
          f"{config.LSTM_FULL86_D_EMBED_V0} (v0) + "
          f"{config.LSTM_FULL86_D_EMBED_V1} (v1) + "
          f"{config.LSTM_FULL86_N_SENSOR_FIELDS * config.LSTM_FULL86_D_PER_SENSOR_FIELD} (sensor) + "
          f"{config.LSTM_FULL86_D_EMBED_D} (dscnn) + "
          f"{config.LSTM_FULL86_D_EMBED_C} (cad) + "
          f"{config.LSTM_FULL86_D_EMBED_SC} (scan) = {n_total}")
    print(f"  stack_v0   shape: {dataset['stacks_v0'].shape}")
    print(f"  stack_v1   shape: {dataset['stacks_v1'].shape}")
    print(f"  sensors    shape: {dataset['sensors'].shape}")
    print(f"  dscnn      shape: {dataset['dscnn'].shape}")
    print(f"  cad_patch  shape: {dataset['cad_patch'].shape}")
    print(f"  scan_patch shape: {dataset['scan_patch'].shape}")
    print(f"  lengths range [{int(dataset['lengths'].min())}, "
          f"{int(dataset['lengths'].max())}], "
          f"median={int(np.median(dataset['lengths']))}")

    save_norm_params(
        dataset["norm_params"],
        config.LSTM_FULL86_FEATURES_DIR / "normalization.json",
    )

    train_all(
        dataset,
        output_dir=config.LSTM_FULL86_MODELS_DIR,
        device=device,
    )

    print("\n[S3 evaluate]")
    results = evaluate_all(
        dataset,
        models_dir=config.LSTM_FULL86_MODELS_DIR,
        device=device,
    )
    save_metrics(results, output_dir=config.LSTM_FULL86_RESULTS_DIR)
    plot_correlation(results, output_dir=config.LSTM_FULL86_RESULTS_DIR)
    plot_scatter_uts(results, output_dir=config.LSTM_FULL86_RESULTS_DIR)


def run_evaluate(device: str, build_ids: list[str]):
    raw = load_septet_dataset(
        features_npz=config.FEATURES_DIR / "all_features.npz",
        cache_v0_dir=config.LSTM_FULL86_CACHE_V0_DIR,
        cache_v1_dir=config.LSTM_FULL86_CACHE_V1_DIR,
        cache_sensor_dir=config.LSTM_FULL86_CACHE_SENSOR_DIR,
        cache_dscnn_dir=config.LSTM_FULL86_CACHE_DSCNN_DIR,
        cache_cad_dir=config.LSTM_FULL86_CACHE_CAD_DIR,
        cache_scan_dir=config.LSTM_FULL86_CACHE_SCAN_DIR,
        build_ids=build_ids,
    )
    dataset = build_normalized_dataset(raw)
    results = evaluate_all(
        dataset,
        models_dir=config.LSTM_FULL86_MODELS_DIR,
        device=device,
    )
    save_metrics(results, output_dir=config.LSTM_FULL86_RESULTS_DIR)
    plot_correlation(results, output_dir=config.LSTM_FULL86_RESULTS_DIR)
    plot_scatter_uts(results, output_dir=config.LSTM_FULL86_RESULTS_DIR)


def _save_experiment_meta():
    n_static = len(config.LSTM_FULL86_STATIC_IDX)
    n_total = (
        n_static
        + config.LSTM_FULL86_D_EMBED_V0 + config.LSTM_FULL86_D_EMBED_V1
        + config.LSTM_FULL86_N_SENSOR_FIELDS * config.LSTM_FULL86_D_PER_SENSOR_FIELD
        + config.LSTM_FULL86_D_EMBED_D
        + config.LSTM_FULL86_D_EMBED_C
        + config.LSTM_FULL86_D_EMBED_SC
    )
    dscnn_channel_names = [v[1] for v in config.DSCNN_FEATURE_MAP.values()]
    dscnn_class_ids = [v[0] for v in config.DSCNN_FEATURE_MAP.values()]
    meta = {
        "model_class": "VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_1DCNN_Sensor_4",
        "n_static_feats": n_static,
        "static_feat_idx": config.LSTM_FULL86_STATIC_IDX,
        "static_feat_names": ["build_height", "laser_module"],
        "n_total_feats": n_total,
        "branches": {
            "camera_v0": {
                "in_channels": 1,
                "d_cnn": config.LSTM_D_CNN,
                "d_hidden": config.LSTM_FULL86_D_HIDDEN_CAM,
                "d_embed": config.LSTM_FULL86_D_EMBED_V0,
            },
            "camera_v1": {
                "in_channels": 1,
                "d_cnn": config.LSTM_D_CNN,
                "d_hidden": config.LSTM_FULL86_D_HIDDEN_CAM,
                "d_embed": config.LSTM_FULL86_D_EMBED_V1,
            },
            "sensor": {
                "type": "per_field_1dcnn",
                "n_fields": config.LSTM_FULL86_N_SENSOR_FIELDS,
                "d_per_field": config.LSTM_FULL86_D_PER_SENSOR_FIELD,
                "hidden_ch": config.LSTM_FULL86_SENSOR_HIDDEN_CH,
                "kernel": config.LSTM_FULL86_SENSOR_KERNEL,
                "field_names": list(config.TEMPORAL_FEATURES),
            },
            "dscnn": {
                "type": "group_lstm",
                "n_channels": config.LSTM_FULL86_N_DSCNN_CH,
                "d_hidden": config.LSTM_FULL86_D_HIDDEN_D,
                "d_embed": config.LSTM_FULL86_D_EMBED_D,
                "channel_names": dscnn_channel_names,
                "class_ids": dscnn_class_ids,
            },
            "cad_patch": {
                "type": "spatial_cnn_lstm",
                "in_channels": config.LSTM_FULL86_N_CAD_CH,
                "patch_h": config.LSTM_FULL86_CAD_PATCH_H,
                "patch_w": config.LSTM_FULL86_CAD_PATCH_W,
                "d_cnn": config.LSTM_FULL86_D_CNN_C,
                "d_hidden": config.LSTM_FULL86_D_HIDDEN_C,
                "d_embed": config.LSTM_FULL86_D_EMBED_C,
                "channel_names": ["edge_proximity", "overhang_proximity"],
                "inversion_applied": config.LSTM_FULL86_CAD_INVERSION_APPLIED,
                "mask_applied": config.LSTM_FULL86_CAD_MASK_APPLIED,
                "edge_saturation_mm": config.DIST_EDGE_SATURATION_MM,
                "overhang_saturation_layers": config.DIST_OVERHANG_SATURATION_LAYERS,
            },
            "scan_patch": {
                "type": "spatial_cnn_lstm",
                "in_channels": config.LSTM_FULL86_N_SCAN_CH,
                "patch_h": config.LSTM_FULL86_SCAN_PATCH_H,
                "patch_w": config.LSTM_FULL86_SCAN_PATCH_W,
                "d_cnn": config.LSTM_FULL86_D_CNN_SC,
                "d_hidden": config.LSTM_FULL86_D_HIDDEN_SC,
                "d_embed": config.LSTM_FULL86_D_EMBED_SC,
                "channel_names": ["return_delay", "stripe_boundaries"],
                "inversion_applied": config.LSTM_FULL86_SCAN_INVERSION_APPLIED,
                "mask_applied": config.LSTM_FULL86_SCAN_MASK_APPLIED,
                "return_delay_saturation_s": 0.75,
            },
        },
        "mlp_hidden": list(config.LSTM_FULL86_MLP_HIDDEN),
        "T_max": config.LSTM_T_MAX,
        "train": {
            "lr": config.LSTM_LR,
            "batch_size": config.LSTM_BATCH_SIZE,
            "max_epochs": config.LSTM_MAX_EPOCHS,
            "early_stop_patience": config.LSTM_EARLY_STOP_PATIENCE,
            "grad_clip": config.LSTM_GRAD_CLIP,
            "weight_decay": config.LSTM_WEIGHT_DECAY,
            "dropout": config.DROPOUT_RATE,
        },
    }
    with open(config.LSTM_FULL86_EXPERIMENT_DIR / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-1DCNN-Sensor-4 Pipeline",
    )
    parser.add_argument(
        "--phase",
        choices=["cache_cad_patch", "cache_scan_patch", "train", "evaluate"],
        help="실행 단계",
    )
    parser.add_argument("--all", action="store_true",
                        help="cache_cad_patch → cache_scan_patch → train → evaluate 일괄")
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

    if args.all or args.phase == "cache_cad_patch":
        run_cache_cad_patch(args.builds)
    if args.all or args.phase == "cache_scan_patch":
        run_cache_scan_patch(args.builds)
    if args.all or args.phase == "train":
        run_train(device, args.builds)
    if (not args.all) and args.phase == "evaluate":
        run_evaluate(device, args.builds)


if __name__ == "__main__":
    main()
