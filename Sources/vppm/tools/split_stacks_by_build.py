"""
stacks_all.h5 → 빌드별 stacks_B1.X.h5 분리 (H5Web 등에서 열기 쉽게).

usage:
    ./venv/bin/python -m Sources.vppm.tools.split_stacks_by_build
"""
from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np

from ..common import config

SRC = config.OUTPUT_DIR / "image_stacks" / "stacks_all.h5"
OUT_DIR = config.OUTPUT_DIR / "image_stacks" / "per_build"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builds = list(config.BUILDS.keys())

    with h5py.File(SRC, "r") as src:
        boundaries = src["build_boundaries"][...]
        attrs = dict(src.attrs)
        T = int(attrs["T"])
        C = int(attrs["n_channels"])
        patch = int(attrs["patch_px"])

        for i, bid in enumerate(builds):
            lo, hi = int(boundaries[i]), int(boundaries[i + 1])
            n = hi - lo
            out_path = OUT_DIR / f"stacks_{bid}.h5"
            print(f"[split] {bid}: rows {lo}:{hi}  (N={n}) → {out_path}")
            with h5py.File(out_path, "w") as dst:
                # H5Web 는 float16 미지원 → float32 로 변환
                # gzip 압축 + H5Web 친화적 chunk (한 슈퍼복셀 = 1 chunk)
                dst.create_dataset(
                    "stacks",
                    data=src["stacks"][lo:hi].astype(np.float32),
                    chunks=(1, T, C, patch, patch),
                    compression="gzip", compression_opts=4,
                )
                dst.create_dataset(
                    "masks",
                    data=src["masks"][lo:hi],
                    compression="gzip", compression_opts=4,
                )
                dst.create_dataset("row_offset", data=lo, dtype="int64")
                dst.create_dataset("n_rows", data=n, dtype="int64")
                for k, v in attrs.items():
                    dst.attrs[k] = v
                dst.attrs["build_id"] = bid
                dst.attrs["source"] = str(SRC)
            size_mb = out_path.stat().st_size / 1e6
            print(f"         done: {size_mb:.1f} MB")

    print(f"\nall done → {OUT_DIR}")


if __name__ == "__main__":
    main()
