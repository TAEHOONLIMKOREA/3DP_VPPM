"""5개 빌드(B1.1~B1.5) × 7개 센서 = 35개 plot 시각화.

baseline 21 피처 중 11~17번 (논문 Table A4의 12~18) 에 해당하는 7개
센서 시계열을 빌드별로 PNG 1개에 묶어 저장한다.

출력: Sources/pipeline_outputs/figures/sensor_data/{B1.x}_temporal_sensors.png
"""
from __future__ import annotations

from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Sources"))

from vppm.common import config  # noqa: E402

OUTPUT_DIR = ROOT / "Sources/pipeline_outputs/figures/sensor_data"

SENSORS = config.TEMPORAL_FEATURES  # 7개

TITLES = {
    "layer_times": "Layer Print Time",
    "top_flow_rate": "Top Argon Flow Rate",
    "bottom_flow_rate": "Bottom Argon Flow Rate",
    "module_oxygen": "Module Oxygen (Chamber O$_2$)",
    "build_plate_temperature": "Build Plate Temperature",
    "bottom_flow_temperature": "Bottom Flow Temperature",
    "actual_ventilator_flow_rate": "Ventilator Flow Rate",
}

UNITS = {
    "layer_times": "s/layer",
    "top_flow_rate": "L/min",
    "bottom_flow_rate": "L/min",
    "module_oxygen": "ppm",
    "build_plate_temperature": "$^\\circ$C",
    "bottom_flow_temperature": "$^\\circ$C",
    "actual_ventilator_flow_rate": "L/min",
}

BUILD_DESC = {
    "B1.1": "Baseline conditions (503 samples)",
    "B1.2": "Process parameter sweep (2705 samples)",
    "B1.3": "Overhang geometry (813 samples)",
    "B1.4": "Spatter / gas flow variation (694 samples)",
    "B1.5": "Recoater damage / powder underfeed (1584 samples)",
}


def load_temporal(build_id: str) -> dict[str, np.ndarray]:
    path = config.hdf5_path(build_id)
    out = {}
    with h5py.File(path, "r") as f:
        for key in SENSORS:
            out[key] = f[f"temporal/{key}"][:]
    return out


def plot_build(build_id: str, data: dict[str, np.ndarray]) -> Path:
    n_layers = len(next(iter(data.values())))
    fig, axes = plt.subplots(7, 1, figsize=(13, 16), sharex=True)
    fig.suptitle(
        f"{build_id} — {BUILD_DESC.get(build_id, '')}    [n_layers = {n_layers}]",
        fontsize=14, fontweight="bold", y=0.995,
    )

    for ax, sensor in zip(axes, SENSORS):
        vals = np.asarray(data[sensor], dtype=float)
        x = np.arange(len(vals))
        ax.plot(x, vals, lw=0.6, color="tab:blue")

        finite = vals[np.isfinite(vals)]
        if finite.size:
            mean, std = finite.mean(), finite.std()
            vmin, vmax = finite.min(), finite.max()
            stats = f"mean={mean:.3g}  std={std:.3g}  min={vmin:.3g}  max={vmax:.3g}  n_nan={(~np.isfinite(vals)).sum()}"
            ax.text(
                0.995, 0.96, stats,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="0.7"),
            )

        ax.set_ylabel(f"{TITLES[sensor]}\n[{UNITS[sensor]}]", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.005)

    axes[-1].set_xlabel("Layer index")
    plt.tight_layout(rect=[0, 0, 1, 0.985])

    out_path = OUTPUT_DIR / f"{build_id}_temporal_sensors.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output → {OUTPUT_DIR}")
    for build_id in ["B1.1", "B1.2", "B1.3", "B1.4", "B1.5"]:
        print(f"  [{build_id}] loading temporal/* ...", flush=True)
        data = load_temporal(build_id)
        out = plot_build(build_id, data)
        n = len(next(iter(data.values())))
        print(f"    saved {out.name}  (n_layers={n})")
    print("Done.")


if __name__ == "__main__":
    main()
