"""distance_from_overhang(#2) 단일 피처 통계 산출.

새로 작성한 vertical-column 알고리즘으로 5개 빌드 전체에 대해 SV-level
distance_from_overhang 값을 추출하고 통계(mean/std/min/max/percentiles + 히스토그램)
를 출력한다. Full feature extraction 대비 G1(DSCNN seg)/G2(temporal)/G4(scan rasterize)
를 건너뛰므로 빌드당 수 분 수준.

산출물:
    Sources/pipeline_outputs/figures/stats_overhang/
        per_build/{build_id}.npy        — 각 빌드의 SV-level dist_oh 배열
        all_builds.npy                  — 5개 빌드 concat
        stats.json                      — mean/std/min/max/percentiles
        histogram.png                   — 분포 히스토그램

사용:
    python -m Sources.vppm.tools.stats_overhang [--builds B1.1 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Sources.vppm.common import config
from Sources.vppm.common.supervoxel import SuperVoxelGrid, find_valid_supervoxels


def compute_overhang_per_sv(build_id: str, skip_gaussian: bool = True) -> np.ndarray:
    """단일 빌드의 모든 valid SV 에 대해 distance_from_overhang(#2) 만 계산.

    구현은 Sources/vppm/baseline/features.py 의 _extract_cad_features_block 에서
    #2 부분만 떼어낸 것과 동일. 통계 산출 목적의 빠른 경로를 위해 두 최적화를 적용:
      1) part_ids HDF5 를 z-block(70 layer) 단위로 일괄 read — 청크 해독 오버헤드 ↓
      2) `skip_gaussian=True` (default) 면 σ≈3.76 px 가우시안 블러 생략. SV 가
         CAD 픽셀에서만 평균을 잡고 dist_oh 가 같은 part 내부에서 거의 균일하므로
         가우시안 effect 가 SV-mean 에 미치는 영향은 미미 (경계에서 약간의 smear).
         정확히 baseline features 와 매칭하려면 `skip_gaussian=False` 로 호출.
    """
    hdf5 = str(config.hdf5_path(build_id))
    grid = SuperVoxelGrid.from_hdf5(hdf5)
    valid = find_valid_supervoxels(grid, hdf5)
    indices = valid["voxel_indices"]            # (N, 3)
    n_voxels = len(indices)
    if n_voxels == 0:
        print(f"  {build_id}: no valid SVs, skip")
        return np.array([], dtype=np.float32)

    # 빌드 전체에 걸쳐 carry-over 되는 상태
    H, W = grid.image_h, grid.image_w
    prev_cad_layer = None
    last_overhang_layer = np.full((H, W), -np.inf, dtype=np.float32)
    sat_layers = float(config.DIST_OVERHANG_SATURATION_LAYERS)
    sigma_px = config.GAUSSIAN_STD_PIXELS

    # SV-level 누적
    accum = np.zeros(n_voxels, dtype=np.float64)
    counts = np.zeros(n_voxels, dtype=np.float64)

    # iz 별 SV 인덱스 매핑
    iz_to_svs: dict[int, list[int]] = {}
    for i, (_, _, iz) in enumerate(indices):
        iz_to_svs.setdefault(int(iz), []).append(i)

    with h5py.File(hdf5, "r") as f:
        part_ids_ds = f["slices/part_ids"]

        for iz in tqdm(range(grid.nz), desc=f"{build_id} z-blocks"):
            l0, l1 = grid.get_layer_range(iz)
            sv_ids = iz_to_svs.get(iz, [])

            # z-block 단위 일괄 read — 청크 해독 오버헤드 ↓
            block_part = part_ids_ds[l0:l1]                # (Tb, H, W) uint32

            for li, layer in enumerate(range(l0, l1)):
                cad_mask = block_part[li] > 0

                if prev_cad_layer is not None:
                    overhang = cad_mask & (~prev_cad_layer)
                    if overhang.any():
                        last_overhang_layer[overhang] = float(layer)

                dist_oh_layers = float(layer) - last_overhang_layer
                dist_oh_layers = np.minimum(dist_oh_layers, sat_layers).astype(np.float32)
                if skip_gaussian:
                    dist_oh_smooth = dist_oh_layers
                else:
                    dist_oh_smooth = gaussian_filter(dist_oh_layers, sigma=sigma_px)

                prev_cad_layer = cad_mask.copy()

                # 이 z-block 안에서만 SV 평균 누적
                for sv_i in sv_ids:
                    ix, iy = indices[sv_i, 0], indices[sv_i, 1]
                    r0, r1, c0, c1 = grid.get_pixel_range(int(ix), int(iy))
                    patch_cad = cad_mask[r0:r1, c0:c1]
                    n_cad = int(patch_cad.sum())
                    if n_cad > 0:
                        accum[sv_i] += float(dist_oh_smooth[r0:r1, c0:c1][patch_cad].mean()) * n_cad
                        counts[sv_i] += n_cad

    out = np.full(n_voxels, np.nan, dtype=np.float32)
    ok = counts > 0
    out[ok] = (accum[ok] / counts[ok]).astype(np.float32)
    return out


def summarize(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    pcts = [1, 5, 25, 50, 75, 95, 99]
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "percentiles": {f"p{p}": float(np.percentile(finite, p)) for p in pcts},
        "frac_at_sat": float((finite >= float(config.DIST_OVERHANG_SATURATION_LAYERS) - 1e-3).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", nargs="+", default=list(config.BUILDS.keys()))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=config.OUTPUT_DIR / "figures" / "stats_overhang",
    )
    ap.add_argument("--no-plot", action="store_true", help="히스토그램 PNG 생성 생략")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    (out_dir / "per_build").mkdir(parents=True, exist_ok=True)

    all_vals: list[np.ndarray] = []
    per_build_stats: dict[str, dict] = {}
    for bid in args.builds:
        print(f"\n[{bid}] computing distance_from_overhang ...")
        vals = compute_overhang_per_sv(bid)
        np.save(out_dir / "per_build" / f"{bid}.npy", vals)
        per_build_stats[bid] = summarize(vals)
        print(f"  {bid}: " + json.dumps(per_build_stats[bid], indent=None))
        all_vals.append(vals)

    if all_vals:
        merged = np.concatenate(all_vals)
        np.save(out_dir / "all_builds.npy", merged)
        overall = summarize(merged)
    else:
        overall = {"count": 0}

    stats = {
        "feature": "distance_from_overhang",
        "implementation": "vertical-column (Scime et al. 2023 Appendix D Table A2)",
        "saturation_layers": config.DIST_OVERHANG_SATURATION_LAYERS,
        "per_build": per_build_stats,
        "overall": overall,
    }
    with open(out_dir / "stats.json", "w") as fp:
        json.dump(stats, fp, indent=2)
    print("\n=== overall ===")
    print(json.dumps(overall, indent=2))
    print(f"\nstats → {out_dir / 'stats.json'}")

    if not args.no_plot and all_vals:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        merged_finite = merged[np.isfinite(merged)]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(merged_finite, bins=72, range=(0, 72), edgecolor="black", linewidth=0.3)
        ax.axvline(
            config.DIST_OVERHANG_SATURATION_LAYERS,
            color="red",
            linestyle="--",
            label=f"sat = {config.DIST_OVERHANG_SATURATION_LAYERS}",
        )
        ax.set_xlabel("distance_from_overhang (layers)")
        ax.set_ylabel("# super-voxels")
        ax.set_title(
            f"distance_from_overhang (vertical-column) — N={merged_finite.size}\n"
            f"mean={overall['mean']:.2f}  std={overall['std']:.2f}  "
            f"frac@sat={overall['frac_at_sat']*100:.1f}%"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "histogram.png", dpi=120)
        print(f"plot  → {out_dir / 'histogram.png'}")


if __name__ == "__main__":
    main()
