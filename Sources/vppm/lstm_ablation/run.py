"""LSTM-Full-Stack Ablation 실행 진입점.

전제: base 풀-스택 모델 (`vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4`) 의
모든 캐시 (v0, v1, sensor, dscnn, cad_patch, scan_patch) 가 이미 빌드돼 있어야 한다.

Usage:
    # smoke (1 fold × 1 prop × max_epochs=5)
    python -m Sources.vppm.lstm_ablation.run --experiment E1 --quick
    python -m Sources.vppm.lstm_ablation.run --experiment E3 --quick

    # 풀런 (4 props × 5 folds × max_epochs=5000) — 사용자 실행
    python -m Sources.vppm.lstm_ablation.run --experiment E1
    python -m Sources.vppm.lstm_ablation.run --experiment E3

    # 평가만 (학습된 모델 .pt 가 이미 있을 때)
    python -m Sources.vppm.lstm_ablation.run --experiment E1 --phase evaluate
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
from Sources.vppm.lstm_ablation.dataset import (
    build_normalized_dataset,
    load_septet_dataset,
)
from Sources.vppm.lstm_ablation.evaluate import evaluate_all
from Sources.vppm.lstm_ablation.model import VPPM_LSTM_FullStack_Ablation
from Sources.vppm.lstm_ablation.train import train_all


# ───────────────────────────────────────────────────────────────────────
# 실험 정의 (시리즈 1: E1/E2 카메라 제거, 시리즈 2: E3-E7 단일 분기 isolation)
# ───────────────────────────────────────────────────────────────────────


EXPERIMENTS: dict[str, dict] = {
    # ── 시리즈 1: 카메라 분기 제거 ──────────────────────────────────────
    "E1": {
        "use_static": True,
        "use_v0": False,
        "use_v1": True,
        "use_sensor": True,
        "use_dscnn": True,
        "use_cad": True,
        "use_scan": True,
        "out_subdir": "E1_no_v0",
        "n_total_feats": 70,        # 2 + 0 + 16 + 28 + 8 + 8 + 8
        "removed": ["branch_v0"],
        "kept": ["feat_static", "branch_v1", "branch_sensor", "branch_dscnn", "branch_cad", "branch_scan"],
    },
    "E2": {
        "use_static": True,
        "use_v0": False,
        "use_v1": False,
        "use_sensor": True,
        "use_dscnn": True,
        "use_cad": True,
        "use_scan": True,
        "out_subdir": "E2_no_cameras",
        "n_total_feats": 54,        # 2 + 0 +  0 + 28 + 8 + 8 + 8
        "removed": ["branch_v0", "branch_v1"],
        "kept": ["feat_static", "branch_sensor", "branch_dscnn", "branch_cad", "branch_scan"],
    },
    # ── 시리즈 2: 단일 분기 isolation (use_static=False, 하나만 True) ──
    "E3": {
        "use_static": False,
        "use_v0": True,
        "use_v1": False,
        "use_sensor": False,
        "use_dscnn": False,
        "use_cad": False,
        "use_scan": False,
        "out_subdir": "E3_only_v0_img",
        "n_total_feats": 16,        # 0 + 16 + 0 + 0 + 0 + 0 + 0
        "removed": ["feat_static", "branch_v1", "branch_sensor", "branch_dscnn", "branch_cad", "branch_scan"],
        "kept": ["branch_v0"],
    },
    "E4": {
        "use_static": False,
        "use_v0": False,
        "use_v1": False,
        "use_sensor": False,
        "use_dscnn": True,
        "use_cad": False,
        "use_scan": False,
        "out_subdir": "E4_only_dscnn",
        "n_total_feats": 8,         # 0 + 0 + 0 + 0 + 8 + 0 + 0
        "removed": ["feat_static", "branch_v0", "branch_v1", "branch_sensor", "branch_cad", "branch_scan"],
        "kept": ["branch_dscnn"],
    },
    "E5": {
        "use_static": False,
        "use_v0": False,
        "use_v1": False,
        "use_sensor": False,
        "use_dscnn": False,
        "use_cad": True,
        "use_scan": False,
        "out_subdir": "E5_only_cad",
        "n_total_feats": 8,         # 0 + 0 + 0 + 0 + 0 + 8 + 0
        "removed": ["feat_static", "branch_v0", "branch_v1", "branch_sensor", "branch_dscnn", "branch_scan"],
        "kept": ["branch_cad"],
    },
    "E6": {
        "use_static": False,
        "use_v0": False,
        "use_v1": False,
        "use_sensor": False,
        "use_dscnn": False,
        "use_cad": False,
        "use_scan": True,
        "out_subdir": "E6_only_scan",
        "n_total_feats": 8,         # 0 + 0 + 0 + 0 + 0 + 0 + 8
        "removed": ["feat_static", "branch_v0", "branch_v1", "branch_sensor", "branch_dscnn", "branch_cad"],
        "kept": ["branch_scan"],
    },
    "E7": {
        "use_static": False,
        "use_v0": False,
        "use_v1": False,
        "use_sensor": True,
        "use_dscnn": False,
        "use_cad": False,
        "use_scan": False,
        "out_subdir": "E7_only_sensor",
        "n_total_feats": 28,        # 0 + 0 + 0 + 28 + 0 + 0 + 0
        "removed": ["feat_static", "branch_v0", "branch_v1", "branch_dscnn", "branch_cad", "branch_scan"],
        "kept": ["branch_sensor"],
    },
}


def _experiment_dirs(exp_id: str) -> dict[str, Path]:
    """실험 ID 별 산출물 디렉터리 반환."""
    cfg = EXPERIMENTS[exp_id]
    base = config.LSTM_ABLATION_EXPERIMENT_BASE_DIR / cfg["out_subdir"]
    return {
        "experiment": base,
        "models": base / "models",
        "results": base / "results",
        "features": base / "features",
    }


def _ensure_dirs(dirs: dict[str, Path]):
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)


def _model_factory(exp_id: str):
    """closure: experiment ID → model factory callable."""
    cfg = EXPERIMENTS[exp_id]

    def factory():
        return VPPM_LSTM_FullStack_Ablation(
            use_static=cfg["use_static"],
            use_v0=cfg["use_v0"],
            use_v1=cfg["use_v1"],
            use_sensor=cfg["use_sensor"],
            use_dscnn=cfg["use_dscnn"],
            use_cad=cfg["use_cad"],
            use_scan=cfg["use_scan"],
        )
    return factory


def _model_file_prefix(exp_id: str) -> str:
    return f"vppm_lstm_ablation_{exp_id}"


def _save_experiment_meta(exp_id: str, dirs: dict[str, Path]):
    cfg = EXPERIMENTS[exp_id]
    n_static = len(config.LSTM_FULL86_STATIC_IDX)
    dscnn_channel_names = [v[1] for v in config.DSCNN_FEATURE_MAP.values()]
    dscnn_class_ids = [v[0] for v in config.DSCNN_FEATURE_MAP.values()]
    meta = {
        "model_class": "VPPM_LSTM_FullStack_Ablation",
        "experiment_id": exp_id,
        "removed": cfg["removed"],
        "kept": cfg.get("kept", []),
        "use_static": cfg["use_static"],
        "use_v0": cfg["use_v0"],
        "use_v1": cfg["use_v1"],
        "use_sensor": cfg["use_sensor"],
        "use_dscnn": cfg["use_dscnn"],
        "use_cad": cfg["use_cad"],
        "use_scan": cfg["use_scan"],
        "base_model": "VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_1DCNN_Sensor_4",
        "n_static_feats": n_static,
        "static_feat_idx": config.LSTM_FULL86_STATIC_IDX,
        "static_feat_names": ["build_height", "laser_module"],
        "n_total_feats": cfg["n_total_feats"],
        "branches": {
            "camera_v0": (
                None
                if not cfg["use_v0"]
                else {
                    "in_channels": 1,
                    "d_cnn": config.LSTM_D_CNN,
                    "d_hidden": config.LSTM_FULL86_D_HIDDEN_CAM,
                    "d_embed": config.LSTM_FULL86_D_EMBED_V0,
                }
            ),
            "camera_v1": (
                None
                if not cfg["use_v1"]
                else {
                    "in_channels": 1,
                    "d_cnn": config.LSTM_D_CNN,
                    "d_hidden": config.LSTM_FULL86_D_HIDDEN_CAM,
                    "d_embed": config.LSTM_FULL86_D_EMBED_V1,
                }
            ),
            "sensor": (
                None
                if not cfg["use_sensor"]
                else {
                    "type": "per_field_1dcnn",
                    "n_fields": config.LSTM_FULL86_N_SENSOR_FIELDS,
                    "d_per_field": config.LSTM_FULL86_D_PER_SENSOR_FIELD,
                    "hidden_ch": config.LSTM_FULL86_SENSOR_HIDDEN_CH,
                    "kernel": config.LSTM_FULL86_SENSOR_KERNEL,
                    "field_names": list(config.TEMPORAL_FEATURES),
                }
            ),
            "dscnn": (
                None
                if not cfg["use_dscnn"]
                else {
                    "type": "group_lstm",
                    "n_channels": config.LSTM_FULL86_N_DSCNN_CH,
                    "d_hidden": config.LSTM_FULL86_D_HIDDEN_D,
                    "d_embed": config.LSTM_FULL86_D_EMBED_D,
                    "channel_names": dscnn_channel_names,
                    "class_ids": dscnn_class_ids,
                }
            ),
            "cad_patch": (
                None
                if not cfg["use_cad"]
                else {
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
                }
            ),
            "scan_patch": (
                None
                if not cfg["use_scan"]
                else {
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
                }
            ),
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
    with open(dirs["experiment"] / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ───────────────────────────────────────────────────────────────────────
# 학습 / 평가 phase
# ───────────────────────────────────────────────────────────────────────


def _load_dataset(build_ids: list[str]) -> dict:
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
    return build_normalized_dataset(raw)


def run_train(
    exp_id: str,
    device: str,
    build_ids: list[str],
    *,
    quick: bool,
):
    cfg = EXPERIMENTS[exp_id]
    dirs = _experiment_dirs(exp_id)

    if quick:
        properties = ["yield_strength"]
        n_folds = 1
        max_epochs = 5
        patience = 5
        print(f"[quick] properties=['YS'], n_folds=1, max_epochs={max_epochs}, patience={patience}")
    else:
        properties = None
        n_folds = None
        max_epochs = config.LSTM_MAX_EPOCHS
        patience = config.LSTM_EARLY_STOP_PATIENCE

    print(f"\n[S2 train] loading raw → normalize → train (exp={exp_id})")
    dataset = _load_dataset(build_ids)

    n_valid = len(dataset["features_static"])
    n_samples = len(np.unique(dataset["sample_ids"]))
    print(f"Dataset: {n_valid} SVs, {n_samples} unique samples")
    print(
        f"  use_static={cfg['use_static']}, use_v0={cfg['use_v0']}, use_v1={cfg['use_v1']}, "
        f"use_sensor={cfg['use_sensor']}, use_dscnn={cfg['use_dscnn']}, "
        f"use_cad={cfg['use_cad']}, use_scan={cfg['use_scan']}, "
        f"n_total_feats={cfg['n_total_feats']}"
    )

    save_norm_params(
        dataset["norm_params"],
        dirs["features"] / "normalization.json",
    )

    train_all(
        dataset,
        model_factory=_model_factory(exp_id),
        model_file_prefix=_model_file_prefix(exp_id),
        output_dir=dirs["models"],
        device=device,
        properties=properties,
        n_folds=n_folds,
        max_epochs=max_epochs,
        patience=patience,
    )

    print("\n[S3 evaluate]")
    results = evaluate_all(
        dataset,
        model_factory=_model_factory(exp_id),
        model_file_prefix=_model_file_prefix(exp_id),
        models_dir=dirs["models"],
        device=device,
        properties=properties,
        n_folds=n_folds,
    )
    save_metrics(results, output_dir=dirs["results"])
    if not quick:
        plot_correlation(results, output_dir=dirs["results"])
        plot_scatter_uts(results, output_dir=dirs["results"])


def run_evaluate(exp_id: str, device: str, build_ids: list[str]):
    dirs = _experiment_dirs(exp_id)
    dataset = _load_dataset(build_ids)
    results = evaluate_all(
        dataset,
        model_factory=_model_factory(exp_id),
        model_file_prefix=_model_file_prefix(exp_id),
        models_dir=dirs["models"],
        device=device,
    )
    save_metrics(results, output_dir=dirs["results"])
    plot_correlation(results, output_dir=dirs["results"])
    plot_scatter_uts(results, output_dir=dirs["results"])


# ───────────────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="VPPM-LSTM-Full-Stack Ablation Pipeline (E1-E7)",
    )
    parser.add_argument(
        "--experiment", "--exp", required=True,
        choices=list(EXPERIMENTS.keys()),
        help="실험 ID: E1 (no-v0) | E2 (no-cameras) | E3 (only-v0) | E4 (only-dscnn) | E5 (only-cad) | E6 (only-scan) | E7 (only-sensor)",
    )
    parser.add_argument(
        "--phase",
        choices=["train", "evaluate"],
        default="train",
        help="실행 단계 (default: train, S2+S3 일괄)",
    )
    parser.add_argument("--builds", nargs="+", default=list(config.BUILDS.keys()))
    parser.add_argument("--device", default=None, help="cpu | cuda (기본: 자동)")
    parser.add_argument(
        "--quick", action="store_true",
        help="smoke test: 1 fold × 1 property (YS) × max_epochs=5",
    )
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: {args.experiment} → {EXPERIMENTS[args.experiment]['out_subdir']}")

    dirs = _experiment_dirs(args.experiment)
    _ensure_dirs(dirs)
    _save_experiment_meta(args.experiment, dirs)

    if args.phase == "train":
        run_train(args.experiment, device, args.builds, quick=args.quick)
    elif args.phase == "evaluate":
        run_evaluate(args.experiment, device, args.builds)


if __name__ == "__main__":
    main()
