"""
VPPM 재구현 전체 파이프라인 실행 진입점

Usage:
    # 전체 파이프라인 (단일 빌드 빠른 테스트)
    python -m Sources.vppm.run_pipeline --quick-test

    # 전체 빌드 파이프라인
    python -m Sources.vppm.run_pipeline --all

    # 개별 단계
    python -m Sources.vppm.run_pipeline --phase features --builds B1.2
    python -m Sources.vppm.run_pipeline --phase train
    python -m Sources.vppm.run_pipeline --phase evaluate
"""
import argparse
import sys
import json
import numpy as np
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Sources.vppm import config
from Sources.vppm.supervoxel import SuperVoxelGrid, find_valid_supervoxels
from Sources.vppm.origin.features import FeatureExtractor
from Sources.vppm.dataset import build_dataset, save_norm_params
from Sources.vppm.origin.train import train_all
from Sources.vppm.origin.evaluate import (
    evaluate_all, save_metrics, plot_correlation, plot_scatter_uts,
)


def extract_features_for_build(build_id: str):
    """단일 빌드의 피처 추출"""
    hdf5 = str(config.hdf5_path(build_id))
    print(f"\n{'='*60}")
    print(f"Extracting features for {build_id}")
    print(f"{'='*60}")

    # 그리드 생성
    grid = SuperVoxelGrid.from_hdf5(hdf5)
    s = grid.summary()
    print(f"Grid: {s['nx']}x{s['ny']}x{s['nz']} = {s['total_voxels']} total voxels")

    # 유효 슈퍼복셀 찾기
    print("Finding valid supervoxels...")
    valid = find_valid_supervoxels(grid, hdf5)
    n = len(valid["sample_ids"])
    n_unique = len(np.unique(valid["sample_ids"]))
    print(f"Valid supervoxels: {n} ({n_unique} unique samples)")

    if n == 0:
        print(f"  No valid supervoxels found for {build_id}, skipping")
        return None

    # 피처 추출
    print("Extracting 21 features...")
    extractor = FeatureExtractor(grid, hdf5)
    features = extractor.extract_features(valid)

    # 인장 시험 결과 로드
    import h5py
    targets = {}
    with h5py.File(hdf5, "r") as f:
        for prop in config.TARGET_PROPERTIES:
            key = f"samples/test_results/{prop}"
            if key in f:
                all_vals = f[key][...]
                # sample_id로 인덱싱하여 각 슈퍼복셀의 타겟 할당
                t = np.array([
                    all_vals[sid] if sid < len(all_vals) else np.nan
                    for sid in valid["sample_ids"]
                ])
                targets[prop] = t

    # 저장
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FEATURES_DIR / f"{build_id}_features.npz"
    np.savez_compressed(
        out_path,
        features=features,
        sample_ids=valid["sample_ids"],
        part_ids=valid["part_ids"],
        voxel_indices=valid["voxel_indices"],
        cad_ratio=valid["cad_ratio"],
        **{f"target_{k}": v for k, v in targets.items()},
    )
    print(f"Saved to {out_path}")
    return out_path


def merge_all_builds(build_ids: list):
    """모든 빌드의 피처를 합산"""
    all_features = []
    all_sample_ids = []
    all_targets = {p: [] for p in config.TARGET_PROPERTIES}
    all_build_labels = []

    # 샘플 ID 오프셋 (빌드 간 중복 방지)
    sample_offset = 0

    for build_id in build_ids:
        path = config.FEATURES_DIR / f"{build_id}_features.npz"
        if not path.exists():
            print(f"  {path} not found, skipping {build_id}")
            continue

        data = np.load(path)
        n = len(data["sample_ids"])
        all_features.append(data["features"])
        all_sample_ids.append(data["sample_ids"] + sample_offset)
        for prop in config.TARGET_PROPERTIES:
            key = f"target_{prop}"
            if key in data:
                all_targets[prop].append(data[key])
            else:
                all_targets[prop].append(np.full(n, np.nan))
        all_build_labels.append(np.full(n, list(config.BUILDS.keys()).index(build_id)))

        max_sid = data["sample_ids"].max()
        sample_offset += max_sid + 1
        print(f"  {build_id}: {n} supervoxels loaded")

    features = np.concatenate(all_features)
    sample_ids = np.concatenate(all_sample_ids)
    targets = {k: np.concatenate(v) for k, v in all_targets.items()}
    build_ids_arr = np.concatenate(all_build_labels)

    out_path = config.FEATURES_DIR / "all_features.npz"
    np.savez_compressed(
        out_path,
        features=features,
        sample_ids=sample_ids,
        build_ids=build_ids_arr,
        **{f"target_{k}": v for k, v in targets.items()},
    )
    print(f"\nMerged: {len(features)} total supervoxels → {out_path}")
    return features, sample_ids, targets, build_ids_arr


def run_train(n_feats: int = config.N_FEATURES):
    """학습 실행"""
    # 데이터 로드
    data_path = config.FEATURES_DIR / "all_features.npz"
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run feature extraction first.")
        return

    data = np.load(data_path)
    features = data["features"]
    sample_ids = data["sample_ids"]
    targets = {prop: data[f"target_{prop}"] for prop in config.TARGET_PROPERTIES
               if f"target_{prop}" in data}

    build_ids = data["build_ids"] if "build_ids" in data else None

    # 데이터셋 구축 (정규화)
    dataset = build_dataset(features, sample_ids, targets, build_ids)

    n_valid = len(dataset["features"])
    n_samples = len(np.unique(dataset["sample_ids"]))
    print(f"\nDataset: {n_valid} supervoxels, {n_samples} unique samples")
    print(f"Avg supervoxels/sample: {n_valid/n_samples:.1f}")

    # 정규화 파라미터 저장
    save_norm_params(dataset["norm_params"], config.FEATURES_DIR / "normalization.json")

    # 학습
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Device: {device}")
    train_all(dataset, n_feats=min(n_feats, features.shape[1]), device=device)

    # 즉시 평가
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)
    results = evaluate_all(dataset, n_feats=min(n_feats, features.shape[1]), device=device)
    save_metrics(results)
    plot_correlation(results)
    plot_scatter_uts(results)


def run_lstm_image_stack(builds, channels=None, patch_px=None, cache_dir=None):
    """Phase L1: 이미지 스택 캐시 생성."""
    from Sources.vppm.lstm.image_stack import build_stacks_cache, verify_alignment
    out = build_stacks_cache(
        builds=builds,
        channels_name=channels,
        patch_px=patch_px,
        cache_dir=cache_dir,
    )
    verify_alignment(out)
    return out


def run_lstm_train(device=None):
    """Phase L5: VPPM-LSTM 학습."""
    from Sources.vppm.lstm.train_lstm import train_vppm_lstm
    return train_vppm_lstm(device=device)


def run_lstm_eval(device=None):
    """Phase L6: 평가 + 임베딩 export."""
    from Sources.vppm.lstm.eval_lstm import evaluate_vppm_lstm, export_lstm_embeddings
    evaluate_vppm_lstm(device=device)
    try:
        export_lstm_embeddings(device=device)
    except Exception as e:
        print(f"[L6] embedding export skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="VPPM / VPPM-LSTM Pipeline")
    parser.add_argument(
        "--phase",
        choices=["features", "train", "evaluate",
                 "image-stack", "train-lstm", "eval-lstm", "all"],
        help="Run specific phase",
    )
    parser.add_argument("--builds", nargs="+", default=list(config.BUILDS.keys()),
                        help="Build IDs to process")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test with B1.2 only")
    parser.add_argument("--n-feats", type=int, default=config.N_FEATURES,
                        help="Number of features (for ablation)")

    # VPPM-LSTM 옵션
    parser.add_argument("--use-lstm", action="store_true",
                        help="Run VPPM-LSTM upgrade pipeline")
    parser.add_argument("--channels", default=None,
                        help="LSTM input channels: raw|raw_both|dscnn|raw+dscnn")
    parser.add_argument("--patch-px", type=int, default=None,
                        help="LSTM patch size in pixels")
    parser.add_argument("--cache-dir", default=None,
                        help="Image stack cache dir (default /tmp/image_stacks)")
    args = parser.parse_args()

    if args.quick_test:
        args.builds = ["B1.2"]
        args.all = True

    # ============================================================
    # VPPM-LSTM branch
    # ============================================================
    if args.use_lstm:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        phase = args.phase or "all"

        if phase in ("all", "image-stack"):
            run_lstm_image_stack(
                builds=args.builds,
                channels=args.channels,
                patch_px=args.patch_px,
                cache_dir=args.cache_dir,
            )
        if phase in ("all", "train-lstm"):
            run_lstm_train(device=device)
        if phase in ("all", "eval-lstm"):
            run_lstm_eval(device=device)
        return

    # ============================================================
    # Baseline VPPM branch (기존)
    # ============================================================
    if args.all:
        for bid in args.builds:
            extract_features_for_build(bid)
        merge_all_builds(args.builds)
        run_train(n_feats=args.n_feats)
        return

    if args.phase == "features":
        for bid in args.builds:
            extract_features_for_build(bid)
        merge_all_builds(args.builds)

    elif args.phase == "train":
        run_train(n_feats=args.n_feats)

    elif args.phase == "evaluate":
        data_path = config.FEATURES_DIR / "all_features.npz"
        data = np.load(data_path)
        features = data["features"]
        sample_ids = data["sample_ids"]
        targets = {p: data[f"target_{p}"] for p in config.TARGET_PROPERTIES
                   if f"target_{p}" in data}
        build_ids = data.get("build_ids")
        dataset = build_dataset(features, sample_ids, targets, build_ids)

        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        results = evaluate_all(dataset, n_feats=args.n_feats, device=device)
        save_metrics(results)
        plot_correlation(results)
        plot_scatter_uts(results)


if __name__ == "__main__":
    main()
