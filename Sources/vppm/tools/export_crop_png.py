"""
슈퍼복셀 crop 이미지를 사람이 보기 쉬운 PNG 로 저장.

- small 모드: 기존 8x8 crop 을 upscale (기본 10배 = 80x80 → dpi 로 확대)
- big 모드:   HDF5 원본에서 주변 컨텍스트 포함해 큰 crop 재추출

usage:
    # 8x8 upscale: B1.2 row=0, 채널 0 (raw), 레이어 0..69 중 매 7번째
    ./venv/bin/python -m Sources.vppm.tools.export_crop_png small B1.2 0

    # 64x64 big crop from HDF5: B1.2 row=0
    ./venv/bin/python -m Sources.vppm.tools.export_crop_png big B1.2 0 64

결과: Sources/pipeline_outputs/crops_png/ 아래 PNG 저장.
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
from ..common.supervoxel import SuperVoxelGrid

OUT_DIR = config.OUTPUT_DIR / "crops_png"
PER_BUILD = config.OUTPUT_DIR / "image_stacks" / "per_build"


def _save_grid(imgs: list[np.ndarray], titles: list[str], suptitle: str,
               out: Path, cmap: str = "gray", vmax: float | None = 255):
    n = len(imgs)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.flat):
        if i < n:
            # bilinear interpolation → 작아도 부드럽게 보임
            ax.imshow(imgs[i], cmap=cmap, vmin=0, vmax=vmax,
                      interpolation="bilinear")
            ax.set_title(titles[i], fontsize=9)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def small_mode(build: str, row: int, channel: int = 0, n_snaps: int = 10):
    """8x8 crop → upscale PNG."""
    src = PER_BUILD / f"stacks_{build}.h5"
    with h5py.File(src, "r") as f:
        stack = f["stacks"][row]   # (T, C, 8, 8)
        mask = f["masks"][row]
    valid = np.where(mask)[0]
    picks = np.linspace(0, len(valid) - 1, n_snaps).astype(int)
    imgs = [stack[valid[i], channel] for i in picks]
    titles = [f"layer {valid[i]}" for i in picks]
    vmax = 255 if channel == 0 else 1.0
    cmap = "gray" if channel == 0 else "hot"

    out = OUT_DIR / f"small_{build}_row{row}_C{channel}.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_grid(
        imgs, titles,
        suptitle=f"{build} row={row} C{channel} — 8x8 upscaled (bilinear)",
        out=out, cmap=cmap, vmax=vmax,
    )
    print(f"saved: {out}")


def big_mode(build: str, row: int, patch_px: int = 64, n_snaps: int = 10):
    """HDF5 원본에서 patch_px × patch_px 재추출 (더 큰 컨텍스트)."""
    # voxel_indices 로드 (빌드 내 row index)
    # stacks_all.h5 에선 빌드 간 concat 되어 있으므로, 빌드별 npz 에서 가져옴
    npz_path = config.FEATURES_DIR / f"{build}_features.npz"
    npz = np.load(npz_path)
    voxels = npz["voxel_indices"]      # (N, 3) (ix, iy, iz)
    if row >= len(voxels):
        raise IndexError(f"{build} N={len(voxels)}, row={row}")
    ix, iy, iz = (int(v) for v in voxels[row])

    hdf5 = str(config.hdf5_path(build))
    grid = SuperVoxelGrid.from_hdf5(hdf5)
    l0, l1 = grid.get_layer_range(iz)
    r0, r1, c0, c1 = grid.get_pixel_range(ix, iy)
    cx = (r0 + r1) // 2
    cy = (c0 + c1) // 2
    half = patch_px // 2

    # 이미지 경계 처리
    R0 = max(0, cx - half)
    R1 = min(grid.image_h, cx + half)
    C0 = max(0, cy - half)
    C1 = min(grid.image_w, cy + half)

    with h5py.File(hdf5, "r") as f:
        block = f["slices/camera_data/visible/0"][l0:l1, R0:R1, C0:C1]
        block = block.astype(np.float32)

    # layer 스냅샷
    picks = np.linspace(0, block.shape[0] - 1, n_snaps).astype(int)
    imgs = [block[i] for i in picks]
    titles = [f"layer {iz*70 + i + l0 - l0}" for i in picks]  # 정확한 전역 레이어
    titles = [f"layer {l0 + i}" for i in picks]
    # supervoxel 경계를 빨간 박스로 overlay 하기 위해 박스 좌표 계산
    rel_r0 = r0 - R0
    rel_r1 = r1 - R0
    rel_c0 = c0 - C0
    rel_c1 = c1 - C0

    out = OUT_DIR / f"big_{build}_row{row}_p{patch_px}.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cols = 5
    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.flat):
        if i < len(imgs):
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=255)
            ax.plot(
                [rel_c0, rel_c1, rel_c1, rel_c0, rel_c0],
                [rel_r0, rel_r0, rel_r1, rel_r1, rel_r0],
                "r-", lw=1.2,
            )
            ax.set_title(titles[i], fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"{build} row={row} — {patch_px}x{patch_px} context "
        f"(red=supervoxel {ix},{iy},{iz})", fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"saved: {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0] if args else "small"
    build = args[1] if len(args) > 1 else "B1.2"
    row = int(args[2]) if len(args) > 2 else 0
    if mode == "small":
        channel = int(args[3]) if len(args) > 3 else 0
        small_mode(build, row, channel=channel)
    elif mode == "big":
        patch = int(args[3]) if len(args) > 3 else 64
        big_mode(build, row, patch_px=patch)
    else:
        print(f"unknown mode: {mode}")
        sys.exit(1)
