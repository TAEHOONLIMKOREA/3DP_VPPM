"""
Feature Ablation 실험 실행기 — PLAN.md 참고.

4개의 피처 그룹 (cad / dscnn / sensor / scan) 중 하나를 제거한 뒤
기존 VPPM 학습/평가 파이프라인을 동일 하이퍼파라미터로 돌린다.

산출물 구조 (baseline `pipeline_outputs/{models,results,features}` 레이아웃을 각 실험 폴더 내부에서 그대로 재사용):
    Sources/pipeline_outputs/ablation/<exp_id>_no_<group>/
        experiment_meta.json
        models/
            vppm_{YS,UTS,UE,TE}_fold{0-4}.pt
            training_log.json
        results/
            metrics_summary.json, metrics_raw.json
            predictions_{YS,UTS,UE,TE}.csv
            correlation_plots.png, scatter_plot_uts.png
        features/
            normalization.json

Usage:
    ./venv/bin/python -m Sources.vppm.ablation.run --experiment E1
    ./venv/bin/python -m Sources.vppm.ablation.run --all
    ./venv/bin/python -m Sources.vppm.ablation.run --experiment E3 --quick  # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Sources/vppm/ablation/run.py → 프로젝트 루트는 parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Sources.vppm.common import config
from Sources.vppm.common.dataset import build_dataset, save_norm_params
from Sources.vppm.origin.evaluate import (
    evaluate_all,
    plot_correlation,
    plot_scatter_uts,
    save_metrics,
)
from Sources.vppm.origin.train import train_all

ABLATION_DIR = config.OUTPUT_DIR / "ablation"

# 실험 정의: (exp_id, drop_group, 설명)
EXPERIMENTS = {
    "E1": ("dscnn",  "No-DSCNN — DSCNN 8피처 제거"),
    "E2": ("sensor", "No-Sensor — Temporal 센서 7피처 제거"),
    "E3": ("cad",    "No-CAD — CAD/좌표 3피처 제거"),
    "E4": ("scan",   "No-Scan — 스캔 3피처 제거 (placeholder 2개 포함)"),
}


def load_all_features() -> dict:
    """all_features.npz 로드."""
    path = config.FEATURES_DIR / "all_features.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 `python -m Sources.vppm.run_pipeline --phase features` 로 피처를 생성하세요."
        )
    data = np.load(path)
    return {
        "features": data["features"],
        "sample_ids": data["sample_ids"],
        "build_ids": data["build_ids"] if "build_ids" in data.files else None,
        "targets": {
            p: data[f"target_{p}"]
            for p in config.TARGET_PROPERTIES
            if f"target_{p}" in data.files
        },
    }


def drop_feature_group(features: np.ndarray, group: str) -> tuple[np.ndarray, list[int]]:
    """지정 그룹의 컬럼을 제거해 (N, 21-k) 배열과 남은 인덱스 리스트를 반환."""
    if group not in config.FEATURE_GROUPS:
        raise ValueError(f"알 수 없는 그룹 '{group}'. 가능: {list(config.FEATURE_GROUPS)}")
    drop_idx = set(config.FEATURE_GROUPS[group])
    keep_idx = [i for i in range(features.shape[1]) if i not in drop_idx]
    return features[:, keep_idx], keep_idx


def run_experiment(
    exp_id: str,
    drop_group: str,
    description: str,
    device: str,
    quick: bool = False,
) -> dict:
    """단일 ablation 실험 실행.

    experiment 디렉터리 구조 (baseline `pipeline_outputs/` 와 동일 레이아웃):
        <out_dir>/
          models/     # vppm_*.pt + training_log.json
          results/    # metrics_*.json, predictions_*.csv, plots
          features/   # normalization.json
          experiment_meta.json
    """
    out_dir = ABLATION_DIR / f"{exp_id}_no_{drop_group}"
    models_dir = out_dir / "models"
    results_dir = out_dir / "results"
    features_dir = out_dir / "features"
    for d in (out_dir, models_dir, results_dir, features_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# {exp_id}: {description}")
    print(f"# Drop group: {drop_group}  |  Output: {out_dir}")
    print(f"{'#'*70}")

    raw = load_all_features()
    features_full = raw["features"]
    features_ablated, keep_idx = drop_feature_group(features_full, drop_group)
    n_feats = features_ablated.shape[1]
    dropped_idx = sorted(set(range(features_full.shape[1])) - set(keep_idx))

    print(
        f"Features: {features_full.shape[1]} → {n_feats} "
        f"(dropped idx={dropped_idx}, kept idx={keep_idx[:4]}... total {len(keep_idx)})"
    )

    dataset = build_dataset(
        features_ablated, raw["sample_ids"], raw["targets"], raw["build_ids"]
    )
    n_valid = len(dataset["features"])
    n_samples = len(np.unique(dataset["sample_ids"]))
    print(f"Valid supervoxels: {n_valid} | unique samples: {n_samples}")

    save_norm_params(dataset["norm_params"], features_dir / "normalization.json")

    # 실험 메타 저장 (재현성)
    meta = {
        "exp_id": exp_id,
        "drop_group": drop_group,
        "description": description,
        "dropped_feature_indices": dropped_idx,
        "kept_feature_indices": keep_idx,
        "n_feats": n_feats,
        "n_valid_supervoxels": int(n_valid),
        "n_unique_samples": int(n_samples),
    }
    with open(out_dir / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # quick 모드: 에포크·폴드 수를 줄여 1-2분 내 검증용
    if quick:
        original_max = config.MAX_EPOCHS
        original_patience = config.EARLY_STOP_PATIENCE
        config.MAX_EPOCHS = 20
        config.EARLY_STOP_PATIENCE = 10
        print("[quick] MAX_EPOCHS=20, patience=10 로 smoke test")

    try:
        train_all(dataset, output_dir=models_dir, n_feats=n_feats, device=device)
        results = evaluate_all(dataset, models_dir=models_dir, n_feats=n_feats, device=device)
        save_metrics(results, output_dir=results_dir)
        plot_correlation(results, output_dir=results_dir)
        plot_scatter_uts(results, output_dir=results_dir)
    finally:
        if quick:
            config.MAX_EPOCHS = original_max
            config.EARLY_STOP_PATIENCE = original_patience

    summary = {
        config.TARGET_SHORT[p]: {
            "vppm_rmse_mean": r["vppm_rmse_mean"],
            "vppm_rmse_std": r["vppm_rmse_std"],
            "naive_rmse": r["naive_rmse"],
            "reduction_pct": r["reduction_pct"],
            "fold_rmses": r["fold_rmses"],
        }
        for p, r in results.items()
    }
    return {"meta": meta, "summary": summary}


def build_summary_from_disk() -> dict:
    """ABLATION_DIR 내 모든 E*_no_*/ 를 스캔해 {exp_id: {meta, summary}} dict 로 돌려준다.

    병렬 컨테이너 실행 후 호스트에서 `--rebuild-summary` 로 호출할 때 사용한다.
    각 실험 폴더에는 `experiment_meta.json` 과 `results/metrics_raw.json` 이 있어야 집계된다.
    """
    all_runs: dict = {}
    if not ABLATION_DIR.exists():
        return all_runs

    for exp_dir in sorted(ABLATION_DIR.glob("E*_no_*")):
        meta_path = exp_dir / "experiment_meta.json"
        metrics_path = exp_dir / "results" / "metrics_raw.json"
        if not (meta_path.exists() and metrics_path.exists()):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)
        exp_id = meta.get("exp_id") or exp_dir.name.split("_", 1)[0]
        all_runs[exp_id] = {"meta": meta, "summary": metrics}
    return all_runs


def write_summary_md(all_runs: dict[str, dict]) -> None:
    """ablation/summary.md 에 전 실험 결과를 표로 정리."""
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    path = ABLATION_DIR / "summary.md"

    # baseline 은 기존 pipeline_outputs/results/metrics_raw.json 에서 읽어옴
    baseline_path = config.RESULTS_DIR / "metrics_raw.json"
    baseline = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)

    lines = [
        "# VPPM Feature Ablation 결과 요약",
        "",
        "> 자동 생성 — [PLAN.md](../../vppm/ablation/PLAN.md) 와 같이 볼 것.",
        "",
        "## RMSE (원본 스케일, 5-Fold CV 평균 ± 표준편차)",
        "",
        "| 실험 | Drop | n_feats | YS (MPa) | UTS (MPa) | UE (%) | TE (%) |",
        "|:----:|:----:|:------:|:--------:|:---------:|:------:|:------:|",
    ]
    if baseline:
        def fmt(prop_short):
            r = baseline.get(prop_short)
            if r is None:
                return "—"
            return f"{r['vppm_rmse_mean']:.1f} ± {r['vppm_rmse_std']:.1f}"

        lines.append(
            f"| E0 Baseline | — | 21 | {fmt('YS')} | {fmt('UTS')} | {fmt('UE')} | {fmt('TE')} |"
        )

    for exp_id in sorted(all_runs.keys()):
        run = all_runs[exp_id]
        meta = run["meta"]
        s = run["summary"]

        def cell(short):
            if short not in s:
                return "—"
            return f"{s[short]['vppm_rmse_mean']:.1f} ± {s[short]['vppm_rmse_std']:.1f}"

        lines.append(
            f"| {exp_id} | {meta['drop_group']} | {meta['n_feats']} | "
            f"{cell('YS')} | {cell('UTS')} | {cell('UE')} | {cell('TE')} |"
        )

    lines += [
        "",
        "## ΔRMSE (E*i* − Baseline)",
        "",
        "| 실험 | ΔYS | ΔUTS | ΔUE | ΔTE |",
        "|:----:|:---:|:----:|:---:|:---:|",
    ]
    if baseline:
        for exp_id in sorted(all_runs.keys()):
            run = all_runs[exp_id]
            s = run["summary"]

            def delta(short):
                if short not in s or short not in baseline:
                    return "—"
                d = s[short]["vppm_rmse_mean"] - baseline[short]["vppm_rmse_mean"]
                sign = "+" if d > 0 else ""
                return f"{sign}{d:.2f}"

            lines.append(
                f"| {exp_id} | {delta('YS')} | {delta('UTS')} | {delta('UE')} | {delta('TE')} |"
            )

    lines += [
        "",
        "## 내재 측정오차 (Reference)",
        "",
        "| 속성 | 측정오차 |",
        "|:----:|:-------:|",
        "| YS | 16.6 MPa |",
        "| UTS | 15.6 MPa |",
        "| UE | 1.73 % |",
        "| TE | 2.92 % |",
        "",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary written to {path}")


def main():
    parser = argparse.ArgumentParser(description="VPPM Feature Ablation 실행")
    parser.add_argument(
        "--experiment", "-e",
        choices=list(EXPERIMENTS.keys()),
        help="단일 실험 실행 (E1~E4)",
    )
    parser.add_argument("--all", action="store_true", help="E1~E4 모두 순차 실행")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: epochs=20, patience=10")
    parser.add_argument("--device", default=None,
                        help="cpu | cuda (기본: 자동 감지)")
    parser.add_argument("--skip-summary", action="store_true",
                        help="summary.md 쓰지 않음 (병렬 컨테이너 race 방지용)")
    parser.add_argument("--rebuild-summary", action="store_true",
                        help="학습 없이 디스크의 모든 E*_no_*/ 를 스캔해 summary.md 만 재생성")
    args = parser.parse_args()

    if args.rebuild_summary:
        all_runs = build_summary_from_disk()
        if not all_runs:
            print(f"경고: {ABLATION_DIR} 아래에 완료된 실험 결과를 찾지 못했습니다.")
        else:
            print(f"Rebuilding summary from {len(all_runs)} experiments: {sorted(all_runs)}")
        write_summary_md(all_runs)
        return

    if not args.all and not args.experiment:
        parser.error("--experiment 또는 --all 중 하나는 필수입니다.")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_ids = list(EXPERIMENTS.keys()) if args.all else [args.experiment]
    all_runs = {}
    for exp_id in exp_ids:
        drop_group, desc = EXPERIMENTS[exp_id]
        all_runs[exp_id] = run_experiment(
            exp_id=exp_id,
            drop_group=drop_group,
            description=desc,
            device=device,
            quick=args.quick,
        )

    if args.skip_summary:
        print("\n[skip-summary] summary.md 쓰기 건너뜀 (상위 스크립트가 rebuild 할 예정)")
    else:
        write_summary_md(all_runs)

    print("\n" + "="*70)
    print("Ablation 완료")
    for exp_id, run in all_runs.items():
        print(f"\n[{exp_id}] {EXPERIMENTS[exp_id][1]}")
        for short, s in run["summary"].items():
            print(
                f"  {short}: RMSE {s['vppm_rmse_mean']:.2f} ± {s['vppm_rmse_std']:.2f} "
                f"(naive {s['naive_rmse']:.2f}, -{s['reduction_pct']:.0f}%)"
            )


if __name__ == "__main__":
    main()
