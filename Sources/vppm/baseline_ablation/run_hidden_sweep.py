"""
Hidden-dim Sweep 실행기 — PLAN_hidden_dim_sweep.md 참고.

baseline VPPM 의 `Linear(21 → hidden) → ReLU → Dropout → Linear(hidden → 1)` 구조에서
hidden_dim ∈ {1, 64, 256} 만 변화시켜 model capacity 의 RMSE 영향을 측정.

기준: E0 baseline (hidden=128) — `Sources/pipeline_outputs/experiments/vppm_baseline/`
재학습하지 않고 기존 결과를 비교 기준으로 사용.

Usage:
    ./venv/bin/python -m Sources.vppm.baseline_ablation.run_hidden_sweep --hidden 1
    ./venv/bin/python -m Sources.vppm.baseline_ablation.run_hidden_sweep --all
    ./venv/bin/python -m Sources.vppm.baseline_ablation.run_hidden_sweep --hidden 64 --quick
    ./venv/bin/python -m Sources.vppm.baseline_ablation.run_hidden_sweep --rebuild-summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Sources.vppm.common import config
from Sources.vppm.common.dataset import build_dataset, save_norm_params
from Sources.vppm.baseline.evaluate import (
    evaluate_all,
    plot_correlation,
    plot_scatter_uts,
    save_metrics,
)
from Sources.vppm.baseline.train import train_all

ABLATION_DIR = config.OUTPUT_DIR / "experiments" / "baseline_ablation"

HIDDEN_SWEEP = {
    "H1": (1,   "Bottleneck=1 — 1차원 capacity 하한"),
    "H2": (64,  "Hidden=64 — plateau 진입 후보"),
    "H3": (256, "Hidden=256 — 상위 capacity 검증"),
}


def load_all_features() -> dict:
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


def count_params(n_feats: int, hidden_dim: int) -> int:
    """VPPM 파라미터 수: (n_feats × h + h) + (h × 1 + 1)"""
    return (n_feats * hidden_dim + hidden_dim) + (hidden_dim * 1 + 1)


def run_experiment(
    exp_id: str,
    hidden_dim: int,
    description: str,
    device: str,
    quick: bool = False,
) -> dict:
    """단일 hidden-dim 실험 실행."""
    out_dir = ABLATION_DIR / f"{exp_id}_hidden_{hidden_dim}"
    models_dir = out_dir / "models"
    results_dir = out_dir / "results"
    features_dir = out_dir / "features"
    for d in (out_dir, models_dir, results_dir, features_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# {exp_id}: {description}")
    print(f"# hidden_dim={hidden_dim}  |  Output: {out_dir}")
    print(f"{'#'*70}")

    raw = load_all_features()
    features_full = raw["features"]
    n_feats = features_full.shape[1]
    n_params = count_params(n_feats, hidden_dim)

    print(f"Features: {n_feats} (전체 21 그대로)  |  Params: {n_params}")

    dataset = build_dataset(
        features_full, raw["sample_ids"], raw["targets"], raw["build_ids"]
    )
    n_valid = len(dataset["features"])
    n_samples = len(np.unique(dataset["sample_ids"]))
    print(f"Valid supervoxels: {n_valid} | unique samples: {n_samples}")

    save_norm_params(dataset["norm_params"], features_dir / "normalization.json")

    meta = {
        "exp_id": exp_id,
        "hidden_dim": hidden_dim,
        "description": description,
        "n_feats": n_feats,
        "n_params": n_params,
        "n_valid_supervoxels": int(n_valid),
        "n_unique_samples": int(n_samples),
    }
    with open(out_dir / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 동적 override (ablation/run.py 의 quick 모드와 동일 패턴)
    original_hidden = config.HIDDEN_DIM
    original_max = config.MAX_EPOCHS
    original_patience = config.EARLY_STOP_PATIENCE
    config.HIDDEN_DIM = hidden_dim
    if quick:
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
        config.HIDDEN_DIM = original_hidden
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
    all_runs: dict = {}
    if not ABLATION_DIR.exists():
        return all_runs
    for exp_dir in sorted(ABLATION_DIR.glob("H*_hidden_*")):
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
    """ablation/hidden_sweep_summary.md 에 sweep 결과 표 작성."""
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    path = ABLATION_DIR / "hidden_sweep_summary.md"

    baseline = None
    for candidate in (
        config.OUTPUT_DIR / "experiments" / "vppm_baseline" / "results" / "metrics_raw.json",
        config.RESULTS_DIR / "vppm_baseline" / "metrics_raw.json",
        config.RESULTS_DIR / "metrics_raw.json",
    ):
        if candidate.exists():
            with open(candidate) as f:
                baseline = json.load(f)
            break

    lines = [
        "# VPPM Hidden-Dim Sweep 결과 요약",
        "",
        "> 자동 생성 — [PLAN_hidden_dim_sweep.md](../../vppm/baseline_ablation/PLAN_hidden_dim_sweep.md) 와 같이 볼 것.",
        "",
        "## RMSE (원본 스케일, 5-Fold CV 평균 ± 표준편차)",
        "",
        "| ID | hidden | n_params | YS (MPa) | UTS (MPa) | UE (%) | TE (%) |",
        "|:--:|:------:|:--------:|:--------:|:---------:|:------:|:------:|",
    ]

    # 정렬: hidden_dim 오름차순
    sorted_ids = sorted(all_runs.keys(), key=lambda k: all_runs[k]["meta"]["hidden_dim"])
    rows_done = []
    baseline_inserted = False

    def fmt(rdict, short):
        r = rdict.get(short)
        if r is None:
            return "—"
        return f"{r['vppm_rmse_mean']:.2f} ± {r['vppm_rmse_std']:.2f}"

    for exp_id in sorted_ids:
        run = all_runs[exp_id]
        meta = run["meta"]
        s = run["summary"]
        h = meta["hidden_dim"]
        # baseline (128) 을 정렬 위치에 삽입
        if baseline and (not baseline_inserted) and h > 128:
            n_params_e0 = (21 * 128 + 128) + (128 + 1)
            lines.append(
                f"| **(E0)** | **128** | {n_params_e0} | "
                f"**{fmt(baseline,'YS')}** | **{fmt(baseline,'UTS')}** | "
                f"**{fmt(baseline,'UE')}** | **{fmt(baseline,'TE')}** |"
            )
            baseline_inserted = True
        lines.append(
            f"| {exp_id} | {h} | {meta['n_params']} | "
            f"{fmt(s,'YS')} | {fmt(s,'UTS')} | {fmt(s,'UE')} | {fmt(s,'TE')} |"
        )
        rows_done.append(exp_id)

    if baseline and not baseline_inserted:
        n_params_e0 = (21 * 128 + 128) + (128 + 1)
        lines.append(
            f"| **(E0)** | **128** | {n_params_e0} | "
            f"**{fmt(baseline,'YS')}** | **{fmt(baseline,'UTS')}** | "
            f"**{fmt(baseline,'UE')}** | **{fmt(baseline,'TE')}** |"
        )

    if baseline:
        lines += [
            "",
            "## ΔRMSE vs E0 (= hidden=128)",
            "",
            "| ID | hidden | ΔYS | ΔUTS | ΔUE | ΔTE |",
            "|:--:|:------:|:---:|:----:|:---:|:---:|",
        ]
        for exp_id in sorted_ids:
            run = all_runs[exp_id]
            s = run["summary"]
            h = run["meta"]["hidden_dim"]

            def delta(short):
                if short not in s or short not in baseline:
                    return "—"
                d = s[short]["vppm_rmse_mean"] - baseline[short]["vppm_rmse_mean"]
                sign = "+" if d > 0 else ""
                return f"{sign}{d:.2f}"

            lines.append(
                f"| {exp_id} | {h} | {delta('YS')} | {delta('UTS')} | {delta('UE')} | {delta('TE')} |"
            )

    lines += [
        "",
        "## 학습 안정성 (fold std)",
        "",
        "| ID | hidden | std(YS) | std(UTS) | std(UE) | std(TE) |",
        "|:--:|:------:|:-------:|:--------:|:-------:|:-------:|",
    ]
    if baseline:
        h_e0 = 128
        lines.append(
            f"| (E0) | {h_e0} | "
            f"{baseline.get('YS',{}).get('vppm_rmse_std','—'):.2f} | "
            f"{baseline.get('UTS',{}).get('vppm_rmse_std','—'):.2f} | "
            f"{baseline.get('UE',{}).get('vppm_rmse_std','—'):.2f} | "
            f"{baseline.get('TE',{}).get('vppm_rmse_std','—'):.2f} |"
        )
    for exp_id in sorted_ids:
        run = all_runs[exp_id]
        s = run["summary"]
        h = run["meta"]["hidden_dim"]
        def std(short):
            return f"{s[short]['vppm_rmse_std']:.2f}" if short in s else "—"
        lines.append(
            f"| {exp_id} | {h} | {std('YS')} | {std('UTS')} | {std('UE')} | {std('TE')} |"
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
    parser = argparse.ArgumentParser(description="VPPM Hidden-Dim Sweep 실행")
    parser.add_argument(
        "--hidden", type=int, choices=[1, 64, 256],
        help="단일 hidden_dim 실행 (1 / 64 / 256)",
    )
    parser.add_argument("--all", action="store_true", help="H1, H2, H3 모두 순차 실행")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: epochs=20, patience=10")
    parser.add_argument("--device", default=None,
                        help="cpu | cuda (기본: 자동 감지)")
    parser.add_argument("--rebuild-summary", action="store_true",
                        help="학습 없이 디스크의 H*_hidden_*/ 를 스캔해 hidden_sweep_summary.md 만 재생성")
    args = parser.parse_args()

    if args.rebuild_summary:
        all_runs = build_summary_from_disk()
        if not all_runs:
            print(f"경고: {ABLATION_DIR} 아래에 완료된 hidden sweep 결과를 찾지 못했습니다.")
        else:
            print(f"Rebuilding summary from {len(all_runs)} runs: {sorted(all_runs)}")
        write_summary_md(all_runs)
        return

    if not args.all and args.hidden is None:
        parser.error("--hidden 또는 --all 중 하나는 필수입니다.")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # hidden_dim → exp_id 역매핑
    hidden_to_id = {h: eid for eid, (h, _) in HIDDEN_SWEEP.items()}

    if args.all:
        exp_ids = list(HIDDEN_SWEEP.keys())
    else:
        exp_ids = [hidden_to_id[args.hidden]]

    all_runs = {}
    for exp_id in exp_ids:
        hidden_dim, desc = HIDDEN_SWEEP[exp_id]
        all_runs[exp_id] = run_experiment(
            exp_id=exp_id,
            hidden_dim=hidden_dim,
            description=desc,
            device=device,
            quick=args.quick,
        )

    # 디스크의 기존 H* 결과까지 합쳐 summary 갱신
    merged = build_summary_from_disk()
    merged.update(all_runs)
    write_summary_md(merged)

    print("\n" + "="*70)
    print("Hidden-dim sweep 완료")
    for exp_id, run in all_runs.items():
        h, desc = HIDDEN_SWEEP[exp_id]
        print(f"\n[{exp_id}] hidden={h} — {desc}")
        for short, s in run["summary"].items():
            print(
                f"  {short}: RMSE {s['vppm_rmse_mean']:.2f} ± {s['vppm_rmse_std']:.2f} "
                f"(naive {s['naive_rmse']:.2f}, -{s['reduction_pct']:.0f}%)"
            )


if __name__ == "__main__":
    main()
