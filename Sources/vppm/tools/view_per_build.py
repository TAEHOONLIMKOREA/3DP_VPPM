"""
per_build/stacks_B1.X.h5 에서 슈퍼복셀을 골라 이미지 저장.

usage:
    # 기본 (B1.2, row=0, 채널 0=raw camera, 레이어 10개 스냅샷)
    ./venv/bin/python -m Sources.vppm.tools.view_per_build

    # 특정 빌드/row/채널 지정
    ./venv/bin/python -m Sources.vppm.tools.view_per_build B1.5 123 0

    # 9개 채널 모두 한 레이어에서 비교
    ./venv/bin/python -m Sources.vppm.tools.view_per_build B1.2 0 all 35
"""
from __future__ import annotations
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common import config

CHANNEL_NAMES = [
    "raw camera", "Powder", "Printed", "Recoater Streaking", "Edge Swelling",
    "Debris", "Super-Elevation", "Soot", "Excessive Melting",
]

PER_BUILD = config.OUTPUT_DIR / "image_stacks" / "per_build"


def snapshot_layers(build: str, row: int, channel: int) -> Path:
    """한 슈퍼복셀의 70 레이어 중 10 장을 시간순으로."""
    src = PER_BUILD / f"stacks_{build}.h5"
    with h5py.File(src, "r") as f:
        stack = f["stacks"][row]  # (T, C, H, W)
        mask = f["masks"][row]
    valid = np.where(mask)[0]
    picks = np.linspace(0, len(valid) - 1, 10).astype(int)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    vmax = 255.0 if channel == 0 else None
    for ax, idx in zip(axes.flat, picks):
        t = valid[idx]
        ax.imshow(stack[t, channel], cmap="gray" if channel == 0 else "hot",
                  vmin=0, vmax=vmax)
        ax.set_title(f"layer {t}")
        ax.axis("off")
    fig.suptitle(f"{build} row={row} — C={channel} ({CHANNEL_NAMES[channel]})")
    out = config.OUTPUT_DIR / f"view_{build}_row{row}_C{channel}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def all_channels_one_layer(build: str, row: int, layer: int) -> Path:
    """한 레이어의 9 채널을 나란히."""
    src = PER_BUILD / f"stacks_{build}.h5"
    with h5py.File(src, "r") as f:
        slab = f["stacks"][row, layer]  # (C, H, W)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for c, ax in enumerate(axes.flat):
        vmax = 255.0 if c == 0 else None
        ax.imshow(slab[c], cmap="gray" if c == 0 else "hot",
                  vmin=0, vmax=vmax)
        ax.set_title(f"C{c}: {CHANNEL_NAMES[c]}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{build} row={row} layer={layer} — all 9 channels")
    out = config.OUTPUT_DIR / f"view_{build}_row{row}_layer{layer}_allC.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    build = args[0] if len(args) > 0 else "B1.2"
    row = int(args[1]) if len(args) > 1 else 0
    ch_arg = args[2] if len(args) > 2 else "0"

    if ch_arg == "all":
        layer = int(args[3]) if len(args) > 3 else 35
        out = all_channels_one_layer(build, row, layer)
    else:
        out = snapshot_layers(build, row, int(ch_arg))
    print(f"saved: {out}")
