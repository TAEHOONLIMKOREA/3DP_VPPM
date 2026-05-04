"""[new_v2] 평가 진입점.

Phases:
  features  — 21-feat 추출 + valid SV 식별 + GT 로드 → features/features.npz
  cache     — 6 시퀀스 캐시 빌드 (visible/0, visible/1, sensor, dscnn, cad_patch, scan_patch)
  evaluate  — 학습된 5-fold ensemble inference + part-level mean + GT 비교

전제:
  - ORNL_Data/Co-Registered.../[new_v2] (Peregrine v2023-10)/2023-03-15 AMMTO Spatial Variation Baseline.hdf5 존재
  - 학습된 fullstack 모델/정규화 통계 존재:
      Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/
        models/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1_{YS,UTS,UE,TE}_fold{0..4}.pt
        features/normalization.json

산출물:
  Sources/pipeline_outputs/experiments/eval_new_v2_with_lstm_full59/
    features/features.npz                    — 21-feat + part_ids + GT
    cache/{crop_stacks,crop_stacks_v1,sensor_stacks,dscnn_stacks,cad_patch_stacks,scan_patch_stacks}_AMMTO_v2.h5
    results/per_part_predictions.csv
    results/per_sv_predictions.csv
    results/metrics_summary.json
    results/scatter_{YS,UTS,TE}.png

Usage:
  # 전체 (features → cache → evaluate)
  python -m Sources.vppm.eval_new_v2_with_lstm_full59.run --all

  # 단계별
  python -m Sources.vppm.eval_new_v2_with_lstm_full59.run --phase features
  python -m Sources.vppm.eval_new_v2_with_lstm_full59.run --phase cache
  python -m Sources.vppm.eval_new_v2_with_lstm_full59.run --phase evaluate

  # smoke test (--quick: feature 추출은 그대로, evaluate 만 처음 256 SV)
  python -m Sources.vppm.eval_new_v2_with_lstm_full59.run --phase evaluate --quick
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Sources.vppm.common import config
from Sources.vppm.eval_new_v2_with_lstm_full59 import cache_new_v2
from Sources.vppm.eval_new_v2_with_lstm_full59.evaluate_partlevel import run_evaluation
from Sources.vppm.eval_new_v2_with_lstm_full59.features_partbased import (
    extract_features_new_v2, save_features,
)
from Sources.vppm.eval_new_v2_with_lstm_full59.supervoxel_partbased import (
    find_valid_supervoxels_partbased,
)


def _ensure_dirs():
    for d in (
        config.NEW_V2_EVAL_EXPERIMENT_DIR,
        config.NEW_V2_EVAL_CACHE_DIR,
        config.NEW_V2_EVAL_FEATURES_DIR,
        config.NEW_V2_EVAL_RESULTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def _load_part_targets(hdf5_path: Path) -> dict:
    """parts/test_results/{yield_strength,ultimate_tensile_strength,total_elongation} 로드.

    parts 인덱스는 0..n_parts-1 그대로. NaN-aware (raw 0 은 NaN 으로 변환).
    """
    out = {}
    with h5py.File(hdf5_path, "r") as f:
        for key in ("yield_strength", "ultimate_tensile_strength", "total_elongation"):
            full = f"parts/test_results/{key}"
            if full not in f:
                continue
            arr = f[full][...].astype(np.float32)
            arr_clean = arr.copy()
            arr_clean[arr_clean <= 0.0] = np.nan
            out[key] = arr_clean
    return out


def run_features():
    print("\n[phase: features]")
    hdf5 = config.new_v2_hdf5_path()
    if not hdf5.exists():
        raise FileNotFoundError(f"new_v2 HDF5 없음: {hdf5}")

    out_path = config.NEW_V2_EVAL_FEATURES_DIR / "features.npz"
    if out_path.exists():
        print(f"  {out_path.name} already exists, skip")
        return

    features, valid, _ = extract_features_new_v2(str(hdf5))
    targets = _load_part_targets(hdf5)
    save_features(features, valid, targets, out_path)


def run_cache():
    print("\n[phase: cache] building 6 sequence caches for new_v2")
    cache_new_v2.build_all(config.NEW_V2_EVAL_CACHE_DIR)


def run_evaluate(device: str, quick: bool, batch_size: int):
    print("\n[phase: evaluate]")
    run_evaluation(
        cache_dir=config.NEW_V2_EVAL_CACHE_DIR,
        norm_path=config.NEW_V2_EVAL_TRAINED_NORM_PATH,
        models_dir=config.NEW_V2_EVAL_TRAINED_MODELS_DIR,
        results_dir=config.NEW_V2_EVAL_RESULTS_DIR,
        device=device,
        batch_size=batch_size,
        quick=quick,
    )


def main():
    parser = argparse.ArgumentParser(
        description="[new_v2] LSTM_FULL59 part-level 평가",
    )
    parser.add_argument(
        "--phase",
        choices=["features", "cache", "evaluate"],
        help="단일 phase 실행",
    )
    parser.add_argument("--all", action="store_true",
                        help="features → cache → evaluate 일괄 실행")
    parser.add_argument("--device", default=None, help="cpu | cuda")
    parser.add_argument("--quick", action="store_true",
                        help="evaluate 시 첫 256 SV 만 (smoke test)")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if not args.phase and not args.all:
        parser.error("--phase 또는 --all 중 하나는 필수")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _ensure_dirs()

    if args.all:
        run_features()
        run_cache()
        run_evaluate(device, args.quick, args.batch_size)
    elif args.phase == "features":
        run_features()
    elif args.phase == "cache":
        run_cache()
    elif args.phase == "evaluate":
        run_evaluate(device, args.quick, args.batch_size)


if __name__ == "__main__":
    main()
