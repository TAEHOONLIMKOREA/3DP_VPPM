"""VPPM-1DCNN 실행 진입점.

사전조건:
  - baseline ``Sources/pipeline_outputs/features/all_features.npz`` 가 이미
    생성되어 있어야 한다 (검증 단계에서 z-평균 비교에 사용).

Usage::

    # Phase 1: layer 시퀀스 캐시 빌드 (5 빌드)
    python -m Sources.vppm.1dcnn.run --phase features

    # Phase 2: 학습 (4 props × 5 folds)
    python -m Sources.vppm.1dcnn.run --phase train

    # Phase 3: 평가
    python -m Sources.vppm.1dcnn.run --phase evaluate

    # 전체 (features → train → evaluate)
    python -m Sources.vppm.1dcnn.run --phase all

    # Smoke test
    python -m Sources.vppm.1dcnn.run --phase all --quick

Note: 본 디렉터리 이름이 숫자로 시작 (``1dcnn``) 하므로 일반 ``import`` 로는
접근할 수 없다 — ``importlib.import_module`` 또는 ``-m`` 실행 (``-m`` 은 점 분
리된 경로 그대로 받아들임) 으로만 호출 가능.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# 본 패키지의 형제 모듈은 디렉터리명이 숫자로 시작해 일반 import 불가 → importlib.
_pkg = "Sources.vppm.1dcnn"
exp_config = importlib.import_module(f"{_pkg}.config")
features_seq = importlib.import_module(f"{_pkg}.features_seq")
ds_mod = importlib.import_module(f"{_pkg}.dataset")
train_mod = importlib.import_module(f"{_pkg}.train")
evaluate_mod = importlib.import_module(f"{_pkg}.evaluate")

from Sources.vppm.baseline.evaluate import (
    plot_correlation,
    plot_scatter_uts,
    save_metrics,
)
from Sources.vppm.common import config as common_config
from Sources.vppm.common.dataset import save_norm_params


def _ensure_dirs() -> None:
    for d in (
        exp_config.EXPERIMENT_DIR,
        exp_config.FEATURES_DIR,
        exp_config.MODELS_DIR,
        exp_config.RESULTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def _save_experiment_meta() -> None:
    meta = {
        "n_channels": exp_config.N_CHANNELS,
        "seq_length": exp_config.SEQ_LENGTH,
        "kernel_size": exp_config.KERNEL_SIZE,
        "conv_layers": exp_config.CONV_LAYERS,
        "p1_indices": exp_config.P1_INDICES,
        "p2_indices": exp_config.P2_INDICES,
        "p3_indices": exp_config.P3_INDICES,
        "p4_indices": exp_config.P4_INDICES,
        "train": {
            "lr": common_config.LEARNING_RATE,
            "batch_size": common_config.BATCH_SIZE,
            "max_epochs": common_config.MAX_EPOCHS,
            "early_stop_patience": common_config.EARLY_STOP_PATIENCE,
            "grad_clip": train_mod.GRAD_CLIP,
            "n_folds": common_config.N_FOLDS,
            "random_seed": common_config.RANDOM_SEED,
        },
    }
    with open(exp_config.EXPERIMENT_DIR / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ----------------------------------------------------------------
# Phases
# ----------------------------------------------------------------
def run_features(build_ids: list[str], quick: bool = False) -> None:
    if quick:
        build_ids = build_ids[:1]
        print(f"[quick] features only for {build_ids}")
    print(f"\n[Phase 1: features_seq] builds={build_ids} → {exp_config.FEATURES_SEQ_NPZ}")
    features_seq.build_cache(build_ids, exp_config.FEATURES_SEQ_NPZ)

    # 가능하면 즉시 검증
    if exp_config.BASELINE_FEATURES_NPZ.exists():
        print("\n[Phase 1: validate] z-평균 vs baseline all_features.npz 비교")
        features_seq.validate_against_baseline(
            exp_config.FEATURES_SEQ_NPZ,
            exp_config.BASELINE_FEATURES_NPZ,
        )
    else:
        print(f"[validate] baseline 캐시 없음 ({exp_config.BASELINE_FEATURES_NPZ}) — 검증 skip")


def _load_dataset() -> dict:
    if not exp_config.FEATURES_SEQ_NPZ.exists():
        raise FileNotFoundError(
            f"FATAL: {exp_config.FEATURES_SEQ_NPZ} 가 없습니다. "
            "먼저 `--phase features` 를 실행하세요."
        )
    raw = ds_mod.load_features_seq(exp_config.FEATURES_SEQ_NPZ)
    dataset = ds_mod.build_normalized_dataset_seq(raw)
    n_valid = len(dataset["features"])
    n_samples = len(np.unique(dataset["sample_ids"]))
    print(f"\nDataset: {n_valid} SVs, {n_samples} unique samples")
    print(f"  feature shape: {dataset['features'].shape}")
    return dataset


def run_train(device: str, quick: bool = False) -> None:
    if quick:
        # baseline 의 학습 하이퍼는 common_config 에 있으므로 module 변수로 override
        common_config.MAX_EPOCHS = 5
        common_config.EARLY_STOP_PATIENCE = 3
        print("[quick] MAX_EPOCHS=5, patience=3")

    dataset = _load_dataset()
    save_norm_params(dataset["norm_params"], exp_config.NORMALIZATION_JSON)

    train_mod.train_all(dataset, output_dir=exp_config.MODELS_DIR, device=device)

    # 즉시 평가
    print("\n[Phase 3: evaluate]")
    results = evaluate_mod.evaluate_all(
        dataset, models_dir=exp_config.MODELS_DIR, device=device,
    )
    save_metrics(results, output_dir=exp_config.RESULTS_DIR)
    plot_correlation(results, output_dir=exp_config.RESULTS_DIR)
    plot_scatter_uts(results, output_dir=exp_config.RESULTS_DIR)
    evaluate_mod.save_per_build_rmse(results, output_dir=exp_config.RESULTS_DIR)


def run_evaluate(device: str) -> None:
    dataset = _load_dataset()
    results = evaluate_mod.evaluate_all(
        dataset, models_dir=exp_config.MODELS_DIR, device=device,
    )
    save_metrics(results, output_dir=exp_config.RESULTS_DIR)
    plot_correlation(results, output_dir=exp_config.RESULTS_DIR)
    plot_scatter_uts(results, output_dir=exp_config.RESULTS_DIR)
    evaluate_mod.save_per_build_rmse(results, output_dir=exp_config.RESULTS_DIR)


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="VPPM-1DCNN Pipeline")
    parser.add_argument(
        "--phase", choices=["features", "train", "evaluate", "all"], required=True,
    )
    parser.add_argument("--builds", nargs="+", default=list(common_config.BUILDS.keys()))
    parser.add_argument("--build", default=None,
                        help="단일 빌드만 features 추출 (예: B1.2)")
    parser.add_argument("--device", default=None, help="cpu | cuda (기본: 자동)")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: features=빌드 1개, train=epoch 5, patience=3")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _ensure_dirs()
    _save_experiment_meta()

    builds = [args.build] if args.build else args.builds

    if args.phase == "features":
        run_features(builds, quick=args.quick)
    elif args.phase == "train":
        run_train(device, quick=args.quick)
    elif args.phase == "evaluate":
        run_evaluate(device)
    elif args.phase == "all":
        run_features(builds, quick=args.quick)
        run_train(device, quick=args.quick)


if __name__ == "__main__":
    main()
