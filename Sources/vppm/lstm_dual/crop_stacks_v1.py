"""
Phase D1 — visible/1 채널 SV별 가변 길이 크롭 시퀀스 캐시 빌드

기존 visible/0 캐시 (`experiments/vppm_lstm/cache/crop_stacks_B1.x.h5`) 와
**동일한 SV 순서·lengths** 를 가져야 학습 시 인덱싱 충돌이 없다.
"유효 레이어 = part_ids > 0 in SV xy" 규칙은 카메라 채널과 무관하므로 일치.

산출물: `experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.x.h5`
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..common import config
from ..lstm.crop_stacks import build_cache as _build_cache_generic


def build_v1_cache(build_ids: list[str] | None = None,
                   out_dir: Path = config.LSTM_DUAL_CACHE_DIR) -> list[Path]:
    """visible/1 채널 캐시 빌드 — `crop_stacks_v1_{bid}.h5`."""
    if build_ids is None:
        build_ids = list(config.BUILDS.keys())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return _build_cache_generic(
        build_ids=build_ids,
        out_dir=out_dir,
        channel=config.LSTM_DUAL_CAMERA_CHANNEL_V1,
        file_prefix="crop_stacks_v1",
    )


def verify_v0_v1_consistency(build_ids: list[str] | None = None,
                              cache_v0_dir: Path = config.LSTM_CACHE_DIR,
                              cache_v1_dir: Path = config.LSTM_DUAL_CACHE_DIR) -> None:
    """v0 캐시와 v1 캐시의 lengths/sv_indices/sample_ids 가 비트 단위 일치하는지 검증.

    불일치 시 `RuntimeError` 발생 — 캐시를 다시 빌드해야 함.
    """
    if build_ids is None:
        build_ids = list(config.BUILDS.keys())

    cache_v0_dir = Path(cache_v0_dir)
    cache_v1_dir = Path(cache_v1_dir)

    for bid in build_ids:
        v0 = cache_v0_dir / f"crop_stacks_{bid}.h5"
        v1 = cache_v1_dir / f"crop_stacks_v1_{bid}.h5"
        if not v0.exists():
            raise FileNotFoundError(f"v0 캐시 누락: {v0} — `python -m Sources.vppm.lstm.run --phase cache` 먼저 실행")
        if not v1.exists():
            raise FileNotFoundError(f"v1 캐시 누락: {v1} — `--phase cache_v1` 먼저 실행")

        with h5py.File(v0, "r") as f0, h5py.File(v1, "r") as f1:
            len0 = f0["lengths"][...]
            len1 = f1["lengths"][...]
            sv0 = f0["sv_indices"][...]
            sv1 = f1["sv_indices"][...]
            sid0 = f0["sample_ids"][...]
            sid1 = f1["sample_ids"][...]

        if len0.shape != len1.shape:
            raise RuntimeError(f"{bid}: lengths shape 불일치 v0={len0.shape} v1={len1.shape}")
        if not np.array_equal(len0, len1):
            n_diff = int((len0 != len1).sum())
            raise RuntimeError(f"{bid}: lengths {n_diff}/{len0.size} 개가 다름 — v0/v1 캐시가 다른 SV 정의로 빌드됨")
        if not np.array_equal(sv0, sv1):
            raise RuntimeError(f"{bid}: sv_indices 불일치 — v0/v1 캐시가 다른 SV 순서로 빌드됨")
        if not np.array_equal(sid0, sid1):
            raise RuntimeError(f"{bid}: sample_ids 불일치")
        print(f"  [verify] {bid}: OK (N={len0.size}, T_sv median={int(np.median(len0))})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--builds", nargs="+", default=list(config.BUILDS.keys()))
    p.add_argument("--out-dir", type=Path, default=config.LSTM_DUAL_CACHE_DIR)
    p.add_argument("--verify", action="store_true",
                   help="기존 v0 캐시와 일치 여부만 검증 (빌드 안 함)")
    args = p.parse_args()

    if args.verify:
        verify_v0_v1_consistency(args.builds)
    else:
        build_v1_cache(args.builds, args.out_dir)
        verify_v0_v1_consistency(args.builds, cache_v1_dir=args.out_dir)
