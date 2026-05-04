"""VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-Sensor-1 실행 진입점.

전제 (모든 캐시 외부 재사용 — 신규 빌드 없음):
  - baseline `Sources/pipeline_outputs/features/all_features.npz` 가 이미 생성됨
  - visible/0 캐시 `experiments/vppm_lstm/cache/crop_stacks_B1.x.h5` 존재
  - visible/1 캐시 `experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.x.h5` 존재
  - sensor 캐시 `experiments/vppm_lstm_dual_img_4_sensor_7/cache/sensor_stacks_B1.x.h5` 존재
  - DSCNN 캐시 `experiments/vppm_lstm_dual_img_4_sensor_7_dscnn_8/cache/dscnn_stacks_B1.x.h5` 존재
  - cad_patch / scan_patch 캐시 (fullstack 디렉터리에서 재사용):
    `experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache/{cad_patch,scan_patch}_stacks_B1.x.h5`

Usage:
    # 학습 (모든 prop × 5 folds, 단일 GPU)
    python -m Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1.run --phase train

    # 학습 (특정 prop 만 — 4 GPU 병렬용. metrics_summary 는 만들지 않음)
    python -m Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1.run --phase train --prop YS

    # 평가만 (모든 prop 모델 로드 → metrics_summary / plots 생성)
    python -m Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1.run --phase evaluate

    # 전체 (train → evaluate; 캐시 빌드 단계 없음)
    python -m Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1.run --all

Smoke test:
    python -m Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1.run --all --quick
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
    verify_cad_patch_v0_consistency,
)
from Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.cache_scan_patch import (
    verify_scan_patch_v0_consistency,
)
from Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1.dataset import (
    build_normalized_dataset,
    load_septet_dataset,
)
from Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1.evaluate import evaluate_all
from Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1.train import train_all


def _ensure_dirs():
    for d in (
        config.LSTM_FULL59_EXPERIMENT_DIR,
        config.LSTM_FULL59_CACHE_DIR,
        config.LSTM_FULL59_MODELS_DIR,
        config.LSTM_FULL59_RESULTS_DIR,
        config.LSTM_FULL59_FEATURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def _verify_caches(build_ids: list[str]):
    """6 캐시 v0 정합성 일괄 검증 — 누락 시 명확한 FATAL 메시지 후 종료."""
    print("\n[verify caches] 6-way consistency check")
    verify_v0_v1_consistency(
        build_ids,
        cache_v0_dir=config.LSTM_FULL59_CACHE_V0_DIR,
        cache_v1_dir=config.LSTM_FULL59_CACHE_V1_DIR,
    )
    verify_v0_sensor_consistency(
        build_ids,
        cache_v0_dir=config.LSTM_FULL59_CACHE_V0_DIR,
        cache_sensor_dir=config.LSTM_FULL59_CACHE_SENSOR_DIR,
    )
    verify_dscnn_v0_consistency(
        build_ids,
        cache_v0_dir=config.LSTM_FULL59_CACHE_V0_DIR,
        cache_dscnn_dir=config.LSTM_FULL59_CACHE_DSCNN_DIR,
    )
    verify_cad_patch_v0_consistency(
        build_ids,
        cache_v0_dir=config.LSTM_FULL59_CACHE_V0_DIR,
        cache_cad_dir=config.LSTM_FULL59_CACHE_CAD_DIR,
    )
    verify_scan_patch_v0_consistency(
        build_ids,
        cache_v0_dir=config.LSTM_FULL59_CACHE_V0_DIR,
        cache_scan_dir=config.LSTM_FULL59_CACHE_SCAN_DIR,
    )


def _load_dataset(build_ids: list[str]) -> dict:
    raw = load_septet_dataset(
        features_npz=config.FEATURES_DIR / "all_features.npz",
        cache_v0_dir=config.LSTM_FULL59_CACHE_V0_DIR,
        cache_v1_dir=config.LSTM_FULL59_CACHE_V1_DIR,
        cache_sensor_dir=config.LSTM_FULL59_CACHE_SENSOR_DIR,
        cache_dscnn_dir=config.LSTM_FULL59_CACHE_DSCNN_DIR,
        cache_cad_dir=config.LSTM_FULL59_CACHE_CAD_DIR,
        cache_scan_dir=config.LSTM_FULL59_CACHE_SCAN_DIR,
        build_ids=build_ids,
    )
    return build_normalized_dataset(raw)


def _print_dataset_summary(dataset: dict):
    n_valid = len(dataset["features_static"])
    n_samples = len(np.unique(dataset["sample_ids"]))
    n_static = len(config.LSTM_FULL59_STATIC_IDX)
    n_total = (
        n_static
        + config.LSTM_FULL59_D_EMBED_V0 + config.LSTM_FULL59_D_EMBED_V1
        + config.LSTM_FULL59_D_EMBED_S
        + config.LSTM_FULL59_D_EMBED_D + config.LSTM_FULL59_D_EMBED_C
        + config.LSTM_FULL59_D_EMBED_SC
    )                                                                # 59
    print(f"Dataset: {n_valid} SVs, {n_samples} unique samples")
    print(f"  static feat dim: {dataset['features_static'].shape[1]} (build_height + laser_module)")
    print(f"  MLP input dim:   "
          f"{n_static} (static) + "
          f"{config.LSTM_FULL59_D_EMBED_V0} (v0) + "
          f"{config.LSTM_FULL59_D_EMBED_V1} (v1) + "
          f"{config.LSTM_FULL59_D_EMBED_S} (sensor★LSTM) + "
          f"{config.LSTM_FULL59_D_EMBED_D} (dscnn) + "
          f"{config.LSTM_FULL59_D_EMBED_C} (cad) + "
          f"{config.LSTM_FULL59_D_EMBED_SC} (scan) = {n_total}")
    print(f"  stack_v0   shape: {dataset['stacks_v0'].shape}")
    print(f"  stack_v1   shape: {dataset['stacks_v1'].shape}")
    print(f"  sensors    shape: {dataset['sensors'].shape}")
    print(f"  dscnn      shape: {dataset['dscnn'].shape}")
    print(f"  cad_patch  shape: {dataset['cad_patch'].shape}")
    print(f"  scan_patch shape: {dataset['scan_patch'].shape}")
    print(f"  lengths range [{int(dataset['lengths'].min())}, "
          f"{int(dataset['lengths'].max())}], "
          f"median={int(np.median(dataset['lengths']))}")


def run_train(device: str, build_ids: list[str], prop_filter: str | None = None):
    _verify_caches(build_ids)

    label = prop_filter if (prop_filter and prop_filter.lower() != "all") else "all-props"
    print(f"\n[train:{label}] loading raw → normalize → 5-fold train")
    dataset = _load_dataset(build_ids)
    _print_dataset_summary(dataset)

    # 4 GPU 병렬: prop_filter 가 박힌 컨테이너들이 동시에 같은 normalization.json 을
    # 쓰면 race. all-props 컨테이너 (보통 evaluate 1번 실행) 만 normalization.json 저장.
    if prop_filter is None or prop_filter.lower() == "all":
        save_norm_params(
            dataset["norm_params"],
            config.LSTM_FULL59_FEATURES_DIR / "normalization.json",
        )

    train_all(
        dataset,
        output_dir=config.LSTM_FULL59_MODELS_DIR,
        device=device,
        prop_filter=prop_filter,
    )


def run_evaluate(device: str, build_ids: list[str], prop_filter: str | None = None):
    dataset = _load_dataset(build_ids)
    # evaluate 단독 실행 시 normalization.json 도 (없으면) 저장.
    norm_path = config.LSTM_FULL59_FEATURES_DIR / "normalization.json"
    if not norm_path.exists():
        save_norm_params(dataset["norm_params"], norm_path)

    results = evaluate_all(
        dataset,
        models_dir=config.LSTM_FULL59_MODELS_DIR,
        device=device,
        prop_filter=prop_filter,
    )
    save_metrics(results, output_dir=config.LSTM_FULL59_RESULTS_DIR)
    plot_correlation(results, output_dir=config.LSTM_FULL59_RESULTS_DIR)
    plot_scatter_uts(results, output_dir=config.LSTM_FULL59_RESULTS_DIR)


def _save_experiment_meta():
    n_static = len(config.LSTM_FULL59_STATIC_IDX)
    n_total = (
        n_static
        + config.LSTM_FULL59_D_EMBED_V0 + config.LSTM_FULL59_D_EMBED_V1
        + config.LSTM_FULL59_D_EMBED_S
        + config.LSTM_FULL59_D_EMBED_D
        + config.LSTM_FULL59_D_EMBED_C
        + config.LSTM_FULL59_D_EMBED_SC
    )
    dscnn_channel_names = [v[1] for v in config.DSCNN_FEATURE_MAP.values()]
    dscnn_class_ids = [v[0] for v in config.DSCNN_FEATURE_MAP.values()]
    meta = {
        "model_class": "VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_Sensor_1",
        "n_static_feats": n_static,
        "static_feat_idx": config.LSTM_FULL59_STATIC_IDX,
        "static_feat_names": ["build_height", "laser_module"],
        "n_total_feats": n_total,
        "branches": {
            "camera_v0": {
                "in_channels": 1,
                "d_cnn": config.LSTM_D_CNN,
                "d_hidden": config.LSTM_FULL59_D_HIDDEN_CAM,
                "d_embed": config.LSTM_FULL59_D_EMBED_V0,
            },
            "camera_v1": {
                "in_channels": 1,
                "d_cnn": config.LSTM_D_CNN,
                "d_hidden": config.LSTM_FULL59_D_HIDDEN_CAM,
                "d_embed": config.LSTM_FULL59_D_EMBED_V1,
            },
            "sensor": {
                "type": "multi_ch_lstm",                                          # ★ fullstack 의 per_field_1dcnn 과 다름
                "n_channels": config.LSTM_FULL59_N_SENSOR_CH,
                "d_hidden": config.LSTM_FULL59_D_HIDDEN_S,
                "d_embed": config.LSTM_FULL59_D_EMBED_S,                          # 1
                "num_layers": config.LSTM_FULL59_NUM_LAYERS_S,
                "bidirectional": config.LSTM_FULL59_BIDIRECTIONAL_S,
                "field_names": list(config.TEMPORAL_FEATURES),
            },
            "dscnn": {
                "type": "group_lstm",
                "n_channels": config.LSTM_FULL59_N_DSCNN_CH,
                "d_hidden": config.LSTM_FULL59_D_HIDDEN_D,
                "d_embed": config.LSTM_FULL59_D_EMBED_D,
                "channel_names": dscnn_channel_names,
                "class_ids": dscnn_class_ids,
            },
            "cad_patch": {
                "type": "spatial_cnn_lstm",
                "in_channels": config.LSTM_FULL59_N_CAD_CH,
                "patch_h": config.LSTM_FULL59_CAD_PATCH_H,
                "patch_w": config.LSTM_FULL59_CAD_PATCH_W,
                "d_cnn": config.LSTM_FULL59_D_CNN_C,
                "d_hidden": config.LSTM_FULL59_D_HIDDEN_C,
                "d_embed": config.LSTM_FULL59_D_EMBED_C,
                "channel_names": ["edge_proximity", "overhang_proximity"],
                "inversion_applied": config.LSTM_FULL86_CAD_INVERSION_APPLIED,
                "mask_applied": config.LSTM_FULL86_CAD_MASK_APPLIED,
                "edge_saturation_mm": config.DIST_EDGE_SATURATION_MM,
                "overhang_saturation_layers": config.DIST_OVERHANG_SATURATION_LAYERS,
                "cache_source": "vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4",
            },
            "scan_patch": {
                "type": "spatial_cnn_lstm",
                "in_channels": config.LSTM_FULL59_N_SCAN_CH,
                "patch_h": config.LSTM_FULL59_SCAN_PATCH_H,
                "patch_w": config.LSTM_FULL59_SCAN_PATCH_W,
                "d_cnn": config.LSTM_FULL59_D_CNN_SC,
                "d_hidden": config.LSTM_FULL59_D_HIDDEN_SC,
                "d_embed": config.LSTM_FULL59_D_EMBED_SC,
                "channel_names": ["return_delay", "stripe_boundaries"],
                "inversion_applied": config.LSTM_FULL86_SCAN_INVERSION_APPLIED,
                "mask_applied": config.LSTM_FULL86_SCAN_MASK_APPLIED,
                "return_delay_saturation_s": 0.75,
                "cache_source": "vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4",
            },
        },
        "mlp_hidden": list(config.LSTM_FULL59_MLP_HIDDEN),
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
    with open(config.LSTM_FULL59_EXPERIMENT_DIR / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-Sensor-1 Pipeline",
    )
    parser.add_argument(
        "--phase",
        choices=["train", "evaluate"],
        help="실행 단계 (캐시 빌드 단계 없음 — 모두 외부 재사용)",
    )
    parser.add_argument("--all", action="store_true",
                        help="train → evaluate 일괄 (단일 GPU 풀 학습용; --prop 와 함께 쓰면 evaluate skip)")
    parser.add_argument("--prop", choices=["all", "YS", "UTS", "UE", "TE"], default="all",
                        help="학습/평가할 property (4-GPU 병렬: 컨테이너별 다른 prop 지정)")
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

    # all-props 컨테이너만 meta.json 저장 (4-GPU 병렬 시 race 회피)
    if args.prop == "all":
        _save_experiment_meta()

    if args.all:
        run_train(device, args.builds, prop_filter=args.prop)
        # prop 필터 있으면 evaluate skip (다른 prop 모델 없을 수 있어 metrics_summary 가 부분만 저장됨)
        if args.prop == "all":
            run_evaluate(device, args.builds, prop_filter=None)
        else:
            print(f"[all + --prop {args.prop}] evaluate skip — 별도 컨테이너에서 "
                  f"`--phase evaluate` 로 모든 prop 모델을 한 번에 평가하세요")
    elif args.phase == "train":
        run_train(device, args.builds, prop_filter=args.prop)
    elif args.phase == "evaluate":
        run_evaluate(device, args.builds,
                     prop_filter=args.prop if args.prop != "all" else None)


if __name__ == "__main__":
    main()
