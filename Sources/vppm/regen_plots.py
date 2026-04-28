"""
correlation_plots.png 재생성 스크립트.

- origin: results/predictions_{YS,UTS,UE,TE}.csv 에서 읽어 재플롯

축 하한 고정(사용자 요청):
  YS=177, UTS=129, UE=0.1, TE=4.2
X/Y 축 동일 범위, 정사각형.
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AXIS_LOWER = {"YS": 177.0, "UTS": 129.0, "UE": 0.1, "TE": 4.2}
SHORTS = ["YS", "UTS", "UE", "TE"]
UNITS = {"YS": "MPa", "UTS": "MPa", "UE": "%", "TE": "%"}


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    trues, preds = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trues.append(float(row["ground_truth"]))
            preds.append(float(row["prediction"]))
    return np.array(trues), np.array(preds)


def plot_correlation(results_dir: Path, titles: dict[str, str] | None = None,
                     out_name: str = "correlation_plots.png") -> Path | None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    any_plotted = False
    for idx, short in enumerate(SHORTS):
        ax = axes[idx // 2, idx % 2]
        csv_path = results_dir / f"predictions_{short}.csv"
        if not csv_path.exists():
            ax.axis("off")
            continue
        trues, preds = _load_csv(csv_path)
        lo = AXIS_LOWER[short]
        hi = max(trues.max(), preds.max()) * 1.02
        lims = [lo, hi]

        ax.set_facecolor("black")
        ax.hist2d(trues, preds, bins=80, range=[lims, lims], cmap="hot",
                  cmin=1)
        ax.plot(lims, lims, "w--", alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Ground Truth ({UNITS[short]})")
        ax.set_ylabel(f"Predicted ({UNITS[short]})")
        title = short
        if titles and short in titles:
            title = f"{short}  {titles[short]}"
        ax.set_title(title)
        ax.set_aspect("equal")
        any_plotted = True
    if not any_plotted:
        plt.close()
        return None
    plt.tight_layout()
    out = results_dir / out_name
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved: {out}")
    return out


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent / "pipeline_outputs" / "results"
    plot_correlation(base)
