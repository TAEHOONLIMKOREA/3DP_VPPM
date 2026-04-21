"""
stacks_all.h5 를 사람이 보기 쉽게 시각화하는 예제.

usage:
    ./venv/bin/python -m Sources.vppm.tools.view_stacks_example [row_idx]
"""
from __future__ import annotations
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STACKS = Path(__file__).resolve().parent.parent / "pipeline_outputs" \
    / "image_stacks" / "stacks_all.h5"


def main(row: int = 0):
    with h5py.File(STACKS, "r") as f:
        print(f"file: {STACKS}")
        print(f"attrs: {dict(f.attrs)}")
        print(f"stacks shape: {f['stacks'].shape}  dtype: {f['stacks'].dtype}")
        N, T, C, H, W = f["stacks"].shape
        print(f"N={N}, T={T}(layers), C={C}(channels), patch={H}x{W}")

        stack = f["stacks"][row]     # (T, C, H, W)
        mask = f["masks"][row]       # (T,)
        build = int(f["build_ids"][row])
        valid_layers = np.where(mask)[0]
        print(f"row={row}: build_id={build}, valid layers={len(valid_layers)}/{T}")

    # 채널 0 (raw 카메라) 의 시간축 스냅샷 10 장
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    picks = np.linspace(0, len(valid_layers) - 1, 10).astype(int)
    for ax, idx in zip(axes.flat, picks):
        t = valid_layers[idx]
        ax.imshow(stack[t, 0], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"layer {t}")
        ax.axis("off")
    fig.suptitle(f"row={row} build={build} — channel 0 (raw camera)")

    out = Path(__file__).resolve().parent.parent / "pipeline_outputs" \
        / f"stacks_row{row}_ch0.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"saved: {out}")


if __name__ == "__main__":
    row = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(row)
