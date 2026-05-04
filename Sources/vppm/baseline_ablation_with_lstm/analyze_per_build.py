"""E0 baseline vs (E*) ablation: 빌드별 잔차 분해.

기존 E2 전용 스크립트를 일반화. 임의의 ablation 실험(E*_no_<group>)과
baseline(21-feat) 을 비교해 빌드(B1.1~B1.5) 단위 RMSE 를 보고한다.

Usage:
    ./venv/bin/python -m Sources.vppm.baseline_ablation_with_lstm.analyze_per_build --experiment E2
    ./venv/bin/python -m Sources.vppm.baseline_ablation_with_lstm.analyze_per_build --experiment E13

산출: ablation/<exp_id>_no_<group>/per_build_analysis.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Sources.vppm.common import config
from Sources.vppm.common.dataset import build_dataset, create_cv_splits, denormalize
from Sources.vppm.common.model import VPPM

BUILD_LABELS = {0: "B1.1", 1: "B1.2", 2: "B1.3", 3: "B1.4", 4: "B1.5"}

# ablation/run.py 의 EXPERIMENTS 와 동기화해서 가져옴 (drop_group 만 필요)
from Sources.vppm.baseline_ablation_with_lstm.run import EXPERIMENTS


def load_raw() -> dict:
    path = config.FEATURES_DIR / "all_features.npz"
    data = np.load(path)
    return {
        "features": data["features"],
        "sample_ids": data["sample_ids"],
        "build_ids": data["build_ids"],
        "targets": {
            p: data[f"target_{p}"]
            for p in config.TARGET_PROPERTIES
            if f"target_{p}" in data.files
        },
    }


def collect_per_sample(
    dataset: dict, models_dir: Path, n_feats: int, device: str
) -> dict[str, list[dict]]:
    features = dataset["features"][:, :n_feats]
    sample_ids = dataset["sample_ids"]
    build_ids = dataset["build_ids"]
    splits = create_cv_splits(sample_ids)
    norm_params = dataset["norm_params"]

    records: dict[str, list[dict]] = {p: [] for p in config.TARGET_PROPERTIES}

    for prop in config.TARGET_PROPERTIES:
        short = config.TARGET_SHORT[prop]
        if prop not in dataset["targets_raw"]:
            continue

        targets_raw = dataset["targets_raw"][prop]
        t_min = norm_params["target_min"][prop]
        t_max = norm_params["target_max"][prop]

        for fold, (_train, val_mask) in enumerate(splits):
            model_path = models_dir / f"vppm_{short}_fold{fold}.pt"
            if not model_path.exists():
                print(f"  skip missing: {model_path}")
                continue
            model = VPPM(n_feats=n_feats)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.to(device).eval()

            x = torch.from_numpy(features[val_mask]).float().to(device)
            with torch.no_grad():
                pred_norm = model(x).cpu().numpy().flatten()
            pred_raw = denormalize(pred_norm, t_min, t_max)

            val_sids = sample_ids[val_mask]
            val_bids = build_ids[val_mask]
            val_true = targets_raw[val_mask]

            per_sample_preds: dict[int, list[float]] = {}
            per_sample_true: dict[int, float] = {}
            per_sample_bid: dict[int, int] = {}
            for i, sid in enumerate(val_sids):
                sid = int(sid)
                per_sample_preds.setdefault(sid, []).append(float(pred_raw[i]))
                per_sample_true[sid] = float(val_true[i])
                per_sample_bid[sid] = int(val_bids[i])

            for sid in sorted(per_sample_preds.keys()):
                records[prop].append({
                    "sample_id": sid,
                    "build_id": per_sample_bid[sid],
                    "y_true": per_sample_true[sid],
                    "y_pred": min(per_sample_preds[sid]),  # 보수적 집계
                    "fold": fold,
                })

    return records


def rmse(arr_true: np.ndarray, arr_pred: np.ndarray) -> float:
    if len(arr_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((arr_pred - arr_true) ** 2)))


def per_build_rmse(records: list[dict]) -> dict:
    by_build: dict[int, dict] = {}
    for bid in sorted({r["build_id"] for r in records}):
        sub = [r for r in records if r["build_id"] == bid]
        arr_t = np.array([r["y_true"] for r in sub])
        arr_p = np.array([r["y_pred"] for r in sub])
        per_fold = []
        for f in range(config.N_FOLDS):
            fsub = [r for r in sub if r["fold"] == f]
            if fsub:
                ft = np.array([r["y_true"] for r in fsub])
                fp = np.array([r["y_pred"] for r in fsub])
                per_fold.append(rmse(ft, fp))
        by_build[bid] = {
            "overall_rmse": rmse(arr_t, arr_p),
            "n_samples": len(sub),
            "per_fold_rmse": per_fold,
            "per_fold_std": float(np.std(per_fold)) if per_fold else float("nan"),
        }
    return by_build


def write_report(
    exp_id: str,
    drop_group: str,
    baseline: dict,
    ablated: dict,
    sample_counts: dict[int, int],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {exp_id} (No-{drop_group}) 빌드별 잔차 분해",
        "",
        f"> 목적: {exp_id} 의 성능 저하 원인 규명. 특정 빌드에서만 해당 그룹 의존이 크면",
        "> 해당 빌드의 ΔRMSE 가 outlier 로 드러난다.",
        "",
        "## 방법",
        "",
        f"- Baseline (21 feats) 와 {exp_id} ({drop_group} 제거) 의 5-Fold CV 예측을",
        "  샘플별 최소값으로 집계 후 빌드(B1.1~B1.5) 단위로 RMSE 를 분해.",
        "- 각 빌드의 overall RMSE 와 fold-간 std 를 기록.",
        "",
        "## 빌드별 샘플 수",
        "",
        "| Build | n_samples |",
        "|:-----:|----------:|",
    ]
    for bid in sorted(sample_counts):
        lines.append(f"| {BUILD_LABELS.get(bid, bid)} | {sample_counts[bid]} |")

    for prop in config.TARGET_PROPERTIES:
        short = config.TARGET_SHORT[prop]
        if prop not in baseline or prop not in ablated:
            continue
        lines += [
            "",
            f"## {short} — 빌드별 RMSE (원본 스케일)",
            "",
            f"| Build | Baseline RMSE | Baseline fold std | {exp_id} RMSE | {exp_id} fold std | ΔRMSE |",
            "|:-----:|--------------:|------------------:|--------------:|------------------:|------:|",
        ]
        for bid in sorted(baseline[prop]):
            b = baseline[prop][bid]
            e = ablated[prop].get(bid)
            if e is None:
                continue
            d = e["overall_rmse"] - b["overall_rmse"]
            sign = "+" if d > 0 else ""
            lines.append(
                f"| {BUILD_LABELS.get(bid, bid)} | "
                f"{b['overall_rmse']:.2f} | {b['per_fold_std']:.2f} | "
                f"{e['overall_rmse']:.2f} | {e['per_fold_std']:.2f} | "
                f"{sign}{d:.2f} |"
            )

        lines.append("")
        lines.append(f"**Fold 별 RMSE ({exp_id})**")
        lines.append("")
        lines.append("| Build | " + " | ".join(f"fold{i}" for i in range(config.N_FOLDS)) + " |")
        lines.append("|:-----:|" + "|".join(["------:"] * config.N_FOLDS) + "|")
        for bid in sorted(ablated[prop]):
            pf = ablated[prop][bid]["per_fold_rmse"]
            lines.append(
                f"| {BUILD_LABELS.get(bid, bid)} | "
                + " | ".join(f"{v:.2f}" if v is not None else "—" for v in pf) + " |"
            )

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n Report → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", required=True,
                        choices=list(EXPERIMENTS.keys()),
                        help="비교 대상 ablation 실험 ID (E1~E4, E13 등)")
    args = parser.parse_args()

    drop_group, _desc = EXPERIMENTS[args.experiment]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    raw = load_raw()

    # ----- Baseline (21 feats) -----
    print("\n[1/2] Baseline (21-feat) 평가…")
    ds_base = build_dataset(raw["features"], raw["sample_ids"], raw["targets"], raw["build_ids"])
    models_base = config.MODELS_DIR
    rec_base = collect_per_sample(ds_base, models_base, n_feats=21, device=device)

    # ----- Ablation 실험 -----
    exp_dir = config.OUTPUT_DIR / "experiments" / "baseline_ablation_with_lstm" / f"{args.experiment}_no_{drop_group}"
    models_abl = exp_dir / "models"
    print(f"\n[2/2] {args.experiment} ({drop_group} 제거) 평가…")
    drop_idx = set(config.FEATURE_GROUPS[drop_group])
    keep_idx = [i for i in range(raw["features"].shape[1]) if i not in drop_idx]
    feats_abl = raw["features"][:, keep_idx]
    ds_abl = build_dataset(feats_abl, raw["sample_ids"], raw["targets"], raw["build_ids"])
    rec_abl = collect_per_sample(ds_abl, models_abl, n_feats=len(keep_idx), device=device)

    baseline = {p: per_build_rmse(rec_base[p]) for p in rec_base}
    ablated = {p: per_build_rmse(rec_abl[p]) for p in rec_abl}

    first_prop = next(iter(baseline))
    sample_counts = {bid: baseline[first_prop][bid]["n_samples"] for bid in baseline[first_prop]}

    out_path = exp_dir / "per_build_analysis.md"
    write_report(args.experiment, drop_group, baseline, ablated, sample_counts, out_path)


if __name__ == "__main__":
    main()
