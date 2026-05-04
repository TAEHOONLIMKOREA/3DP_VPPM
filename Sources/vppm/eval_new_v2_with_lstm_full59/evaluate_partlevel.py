"""[new_v2] 빌드에 학습된 LSTM_FULL59 모델 → part-level 예측 + GT 비교.

핵심 흐름:
  1. 21-feat (`features.npz`) + 6 시퀀스 캐시 로드
  2. **학습된 normalization.json** (LSTM_FULL59_FEATURES_DIR / "normalization.json") 로
     static / sensor / dscnn / cad / scan 정규화 적용 — new_v2 통계로 재계산하지 않음
  3. 4 properties × 5 folds = 20 모델 ensemble inference (SV 단위 예측)
  4. SV 예측을 denormalize → part_id 단위 mean
  5. GT (parts/test_results/{yield_strength,ultimate_tensile_strength,total_elongation})
     와 비교 → RMSE / MAE / scatter / CSV
  6. UE 는 GT 부재로 prediction CSV 만 저장

산출물 → NEW_V2_EVAL_RESULTS_DIR/
  per_part_predictions.csv   — part_id 별 모든 prop 의 (mean_pred, std_over_folds, n_sv, gt)
  per_sv_predictions.csv     — SV 별 raw prediction (디버깅)
  metrics_summary.json       — RMSE/MAE/n (YS/UTS/TE)
  scatter_{prop}.png         — pred vs GT scatter (3 prop)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..common import config
from ..common.dataset import denormalize, normalize
from ..lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1.model import (
    VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_Sensor_1,
)


# ───────────────────────────────────────────────────────────────────────
# Loading
# ───────────────────────────────────────────────────────────────────────


def load_norm_params(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_caches(cache_dir: Path) -> dict:
    """new_v2 6 캐시 + 21-feat npz 로드. 캐시 6 종 lengths/sv_indices 일치 가정 (verify 후)."""
    bid = config.NEW_V2_BUILD_ID
    cache_dir = Path(cache_dir)
    feat_npz = config.NEW_V2_EVAL_FEATURES_DIR / "features.npz"
    if not feat_npz.exists():
        raise FileNotFoundError(f"features.npz 없음: {feat_npz}")

    feat = np.load(feat_npz)
    features21 = feat["features"].astype(np.float32)
    part_ids_sv = feat["part_ids"].astype(np.int32)
    cad_ratio = feat["cad_ratio"].astype(np.float32)
    voxel_indices = feat["voxel_indices"].astype(np.int32)

    cache_paths = {
        "v0":         (cache_dir / f"crop_stacks_{bid}.h5",       "stacks"),
        "v1":         (cache_dir / f"crop_stacks_v1_{bid}.h5",    "stacks"),
        "sensor":     (cache_dir / f"sensor_stacks_{bid}.h5",     "sensors"),
        "dscnn":      (cache_dir / f"dscnn_stacks_{bid}.h5",      "dscnn"),
        "cad_patch":  (cache_dir / f"cad_patch_stacks_{bid}.h5",  "cad_patch"),
        "scan_patch": (cache_dir / f"scan_patch_stacks_{bid}.h5", "scan_patch"),
    }
    out: dict = {}
    ref_lengths: np.ndarray | None = None
    ref_sv_idx: np.ndarray | None = None
    for tag, (p, ds_key) in cache_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{tag} 캐시 누락: {p}")
        with h5py.File(p, "r") as f:
            arr = f[ds_key][...]
            ln = f["lengths"][...]
            sv = f["sv_indices"][...]
        out[tag] = arr
        if ref_lengths is None:
            ref_lengths = ln; ref_sv_idx = sv
        else:
            if not np.array_equal(ref_lengths, ln):
                raise RuntimeError(f"{tag}: lengths mismatch vs v0")
            if not np.array_equal(ref_sv_idx, sv):
                raise RuntimeError(f"{tag}: sv_indices mismatch vs v0")

    # features.npz 와 캐시의 SV 수 + sv_indices 일치 검증
    if len(features21) != len(ref_lengths):
        raise RuntimeError(
            f"features.npz N={len(features21)} ≠ cache N={len(ref_lengths)} — "
            "features.npz 와 캐시가 같은 valid_voxels 로 빌드돼야 합니다"
        )
    if not np.array_equal(voxel_indices, ref_sv_idx):
        raise RuntimeError(
            "features.npz voxel_indices ≠ cache sv_indices — "
            "두 산출물이 같은 valid_voxels 로 만들어졌는지 확인"
        )

    out["features21"] = features21
    out["part_ids_sv"] = part_ids_sv
    out["cad_ratio"] = cad_ratio
    out["voxel_indices"] = voxel_indices
    out["lengths"] = ref_lengths.astype(np.int64)
    return out


# ───────────────────────────────────────────────────────────────────────
# Normalization (학습된 통계 사용)
# ───────────────────────────────────────────────────────────────────────


def apply_train_normalization(raw: dict, norm: dict) -> dict:
    """학습 시 저장된 norm_params 그대로 [-1, 1] 정규화. new_v2 분포 무관."""
    f21 = raw["features21"]                                         # (N, 21)
    sensors = raw["sensor"].astype(np.float32)                      # (N, T, 7)
    dscnn = raw["dscnn"].astype(np.float32)                         # (N, T, 8)
    cad_patch = raw["cad_patch"].astype(np.float32)                 # (N, T, 2, 8, 8)
    scan_patch = raw["scan_patch"].astype(np.float32)               # (N, T, 2, 8, 8)

    # static 2-feat (build_height + laser_module)
    static_idx = np.array(norm["static_idx"], dtype=np.int64)
    s_min = np.array(norm["static_min"], dtype=np.float32)
    s_max = np.array(norm["static_max"], dtype=np.float32)
    feats_static = f21[:, static_idx]
    feats_static_norm = normalize(feats_static, s_min, s_max).astype(np.float32)

    # sensor / dscnn — per-channel min-max
    sensor_min = np.array(norm["sensor_min"], dtype=np.float32)
    sensor_max = np.array(norm["sensor_max"], dtype=np.float32)
    sensors_norm = normalize(sensors, sensor_min, sensor_max).astype(np.float32)

    dscnn_min = np.array(norm["dscnn_min"], dtype=np.float32)
    dscnn_max = np.array(norm["dscnn_max"], dtype=np.float32)
    dscnn_norm = normalize(dscnn, dscnn_min, dscnn_max).astype(np.float32)

    # cad_patch / scan_patch — per-channel (broadcast 1,1,C,1,1)
    cad_min = np.array(norm["cad_min"], dtype=np.float32)
    cad_max = np.array(norm["cad_max"], dtype=np.float32)
    cad_norm = (
        2 * (cad_patch - cad_min[None, None, :, None, None])
        / (cad_max[None, None, :, None, None] - cad_min[None, None, :, None, None] + 1e-8)
        - 1
    ).astype(np.float32)

    scan_min = np.array(norm["scan_min"], dtype=np.float32)
    scan_max = np.array(norm["scan_max"], dtype=np.float32)
    scan_norm = (
        2 * (scan_patch - scan_min[None, None, :, None, None])
        / (scan_max[None, None, :, None, None] - scan_min[None, None, :, None, None] + 1e-8)
        - 1
    ).astype(np.float32)

    return {
        "features_static": feats_static_norm,
        "stacks_v0": raw["v0"],                   # float16 그대로 (model 안에서 .float())
        "stacks_v1": raw["v1"],
        "sensors": sensors_norm,
        "dscnn": dscnn_norm,
        "cad_patch": cad_norm,
        "scan_patch": scan_norm,
        "lengths": raw["lengths"],
        "part_ids_sv": raw["part_ids_sv"],
        "voxel_indices": raw["voxel_indices"],
    }


# ───────────────────────────────────────────────────────────────────────
# Inference
# ───────────────────────────────────────────────────────────────────────


def _infer_one_model(
    model: VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_Sensor_1,
    ds: dict, device: str, batch_size: int = 256,
) -> np.ndarray:
    """단일 모델로 N SVs 정규화 입력에 대한 prediction (정규화 공간) 반환."""
    model.eval()
    N = len(ds["features_static"])
    preds_norm = np.empty(N, dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            i1 = min(i0 + batch_size, N)
            fs = torch.from_numpy(ds["features_static"][i0:i1]).float().to(device)
            s0 = torch.from_numpy(ds["stacks_v0"][i0:i1]).float().to(device)
            s1 = torch.from_numpy(ds["stacks_v1"][i0:i1]).float().to(device)
            sn = torch.from_numpy(ds["sensors"][i0:i1]).float().to(device)
            dn = torch.from_numpy(ds["dscnn"][i0:i1]).float().to(device)
            cad = torch.from_numpy(ds["cad_patch"][i0:i1]).float().to(device)
            scan = torch.from_numpy(ds["scan_patch"][i0:i1]).float().to(device)
            lg = torch.from_numpy(ds["lengths"][i0:i1].astype(np.int64))
            out = model(fs, s0, s1, sn, dn, cad, scan, lg).cpu().numpy().flatten()
            preds_norm[i0:i1] = out
    return preds_norm


def predict_all_props(
    ds_norm: dict, norm: dict,
    models_dir: Path, model_prefix: str,
    device: str, props: Iterable[str] = config.TARGET_PROPERTIES,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """4 prop × 5 fold 앙상블. 반환: prop → (N_SV,) raw prediction (denormalized).

    fold 평균 후 denormalize.
    """
    out: dict[str, np.ndarray] = {}
    for prop in props:
        short = config.TARGET_SHORT[prop]
        fold_preds = []
        for fold in range(config.N_FOLDS):
            mp = Path(models_dir) / f"{model_prefix}_{short}_fold{fold}.pt"
            if not mp.exists():
                print(f"  [{short}] fold{fold} 모델 없음 (skip): {mp}")
                continue
            model = VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_Sensor_1()
            model.load_state_dict(torch.load(mp, weights_only=True, map_location=device))
            model.to(device)
            preds_norm = _infer_one_model(model, ds_norm, device, batch_size=batch_size)
            fold_preds.append(preds_norm)
            print(f"  [{short}] fold{fold} done — pred range "
                  f"[{preds_norm.min():.3f}, {preds_norm.max():.3f}]")
        if not fold_preds:
            print(f"  [{short}] no models found, skip prop")
            continue
        ens = np.stack(fold_preds, axis=0).mean(axis=0)              # ensemble mean (정규화 공간)
        t_min = norm["target_min"][prop]
        t_max = norm["target_max"][prop]
        pred_raw = denormalize(ens, t_min, t_max)
        out[prop] = pred_raw.astype(np.float32)
        # also keep per-fold (정규화공간 → denorm) for std analysis
        out[f"{prop}_per_fold"] = np.stack(
            [denormalize(p, t_min, t_max) for p in fold_preds], axis=0,
        ).astype(np.float32)
    return out


# ───────────────────────────────────────────────────────────────────────
# Part-level aggregation
# ───────────────────────────────────────────────────────────────────────


def aggregate_per_part(
    sv_predictions: dict[str, np.ndarray],
    part_ids_sv: np.ndarray,
    cad_ratio: np.ndarray | None = None,
    weight_by_overlap: bool = True,
) -> dict[int, dict]:
    """SV 예측을 part_id 별로 집계 (mean / std).

    Returns:
        {part_id: {"n_sv": int, "<prop>_mean": float, "<prop>_std": float, ...}}
    """
    unique_parts = np.unique(part_ids_sv)
    if 0 in unique_parts:
        unique_parts = unique_parts[unique_parts != 0]

    out: dict[int, dict] = {}
    for pid in unique_parts:
        mask = (part_ids_sv == pid)
        n_sv = int(mask.sum())
        rec = {"n_sv": n_sv}
        w = None
        if weight_by_overlap and cad_ratio is not None:
            wm = cad_ratio[mask]
            if wm.sum() > 0:
                w = wm / wm.sum()
        for key, pred in sv_predictions.items():
            if key.endswith("_per_fold"):
                continue
            sv_pred_part = pred[mask]
            if w is not None:
                rec[f"{key}_mean"] = float((sv_pred_part * w).sum())
            else:
                rec[f"{key}_mean"] = float(sv_pred_part.mean())
            rec[f"{key}_std"] = float(sv_pred_part.std())
        out[int(pid)] = rec
    return out


# ───────────────────────────────────────────────────────────────────────
# GT loading + comparison
# ───────────────────────────────────────────────────────────────────────


def load_gt(hdf5_path: Path) -> dict[str, np.ndarray]:
    """parts/test_results 로부터 part-id 인덱스의 GT 배열 (NaN-aware) 반환.

    반환: { 'yield_strength': arr(parts,), 'ultimate_tensile_strength': ..., 'total_elongation': ...}
    parts 인덱스 배열 (0..n_parts-1) 와 같은 길이.
    """
    out: dict[str, np.ndarray] = {}
    with h5py.File(hdf5_path, "r") as f:
        if "parts/test_results" not in f:
            raise KeyError("parts/test_results 그룹이 없음")
        for key in ("yield_strength", "ultimate_tensile_strength", "total_elongation"):
            full = f"parts/test_results/{key}"
            if full not in f:
                print(f"  WARN: {full} 없음 — skip")
                continue
            arr = f[full][...].astype(np.float32)
            # NaN/0 둘 다 미측정 처리: 0 → NaN
            arr_clean = arr.copy()
            arr_clean[arr_clean <= 0.0] = np.nan
            out[key] = arr_clean
    return out


def compute_metrics(
    per_part: dict[int, dict],
    gt: dict[str, np.ndarray],
    props_to_compare: tuple[str, ...] = (
        "yield_strength", "ultimate_tensile_strength", "total_elongation",
    ),
) -> dict[str, dict]:
    """GT 가 있는 part_id 만 골라 RMSE / MAE 계산."""
    metrics: dict[str, dict] = {}
    for prop in props_to_compare:
        if prop not in gt:
            continue
        gt_arr = gt[prop]
        pids = []
        preds = []
        trues = []
        for pid, rec in per_part.items():
            if pid >= len(gt_arr):
                continue
            v = gt_arr[pid]
            if np.isnan(v):
                continue
            key = f"{prop}_mean"
            if key not in rec:
                continue
            pids.append(pid)
            preds.append(rec[key])
            trues.append(float(v))
        if not pids:
            continue
        preds_arr = np.array(preds, dtype=np.float64)
        trues_arr = np.array(trues, dtype=np.float64)
        diff = preds_arr - trues_arr
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        # naive baseline = predict mean of GT
        naive = float(np.sqrt(np.mean((trues_arr.mean() - trues_arr) ** 2)))
        metrics[prop] = {
            "rmse": rmse,
            "mae": mae,
            "naive_rmse": naive,
            "n_parts": len(pids),
            "part_ids": [int(x) for x in pids],
            "predictions": preds_arr.tolist(),
            "ground_truths": trues_arr.tolist(),
            "gt_min": float(trues_arr.min()),
            "gt_max": float(trues_arr.max()),
            "gt_mean": float(trues_arr.mean()),
            "gt_std": float(trues_arr.std()),
        }
    return metrics


# ───────────────────────────────────────────────────────────────────────
# Plotting + saving
# ───────────────────────────────────────────────────────────────────────


def plot_scatters(metrics: dict[str, dict], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    units = {
        "yield_strength": "MPa",
        "ultimate_tensile_strength": "MPa",
        "total_elongation": "%",
    }
    for prop, m in metrics.items():
        if "predictions" not in m:
            continue
        preds = np.array(m["predictions"])
        trues = np.array(m["ground_truths"])
        unit = units.get(prop, "")
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        lo = float(min(preds.min(), trues.min())) - 5
        hi = float(max(preds.max(), trues.max())) + 5
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y=x")
        ax.scatter(trues, preds, s=40, alpha=0.7, edgecolor="black")
        ax.set_xlabel(f"GT {prop} [{unit}]")
        ax.set_ylabel(f"Pred {prop} [{unit}]")
        ax.set_title(
            f"{prop} (new_v2 part-level)\n"
            f"RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} "
            f"naïve={m['naive_rmse']:.2f} n={m['n_parts']}"
        )
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path = out_dir / f"scatter_{config.TARGET_SHORT[prop]}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  saved {out_path}")


def save_predictions_csv(
    per_part: dict[int, dict],
    gt: dict[str, np.ndarray],
    out_path: Path,
    sv_part_ids: np.ndarray,
    sv_predictions: dict[str, np.ndarray],
    cad_ratio: np.ndarray,
) -> None:
    """per_part_predictions.csv + per_sv_predictions.csv 저장."""
    out_path = Path(out_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # per_part_predictions.csv
    cols = ["part_id", "n_sv"]
    pred_props = []
    for prop in config.TARGET_PROPERTIES:
        key = f"{prop}_mean"
        if any(key in r for r in per_part.values()):
            pred_props.append(prop)
            cols.append(f"pred_{config.TARGET_SHORT[prop]}_mean")
            cols.append(f"pred_{config.TARGET_SHORT[prop]}_std")
    for prop in ("yield_strength", "ultimate_tensile_strength", "total_elongation"):
        cols.append(f"gt_{config.TARGET_SHORT[prop]}")
    cols.append("gt_available")
    lines = [",".join(cols)]
    for pid in sorted(per_part.keys()):
        rec = per_part[pid]
        row = [str(pid), str(rec["n_sv"])]
        for prop in pred_props:
            row.append(f"{rec.get(f'{prop}_mean', np.nan):.4f}")
            row.append(f"{rec.get(f'{prop}_std', np.nan):.4f}")
        gt_avail = []
        for prop in ("yield_strength", "ultimate_tensile_strength", "total_elongation"):
            arr = gt.get(prop)
            if arr is None or pid >= len(arr) or np.isnan(arr[pid]):
                row.append("")
                gt_avail.append(0)
            else:
                row.append(f"{float(arr[pid]):.4f}")
                gt_avail.append(1)
        row.append(str(int(any(gt_avail))))
        lines.append(",".join(row))
    out_path.write_text("\n".join(lines))
    print(f"  saved {out_path}")

    # per_sv_predictions.csv (디버깅 — 너무 크면 skip 가능)
    sv_csv = out_dir / "per_sv_predictions.csv"
    sv_cols = ["sv_idx", "part_id", "cad_ratio"]
    for prop in pred_props:
        sv_cols.append(f"pred_{config.TARGET_SHORT[prop]}")
    sv_lines = [",".join(sv_cols)]
    for i in range(len(sv_part_ids)):
        row = [str(i), str(int(sv_part_ids[i])), f"{float(cad_ratio[i]):.4f}"]
        for prop in pred_props:
            row.append(f"{float(sv_predictions[prop][i]):.4f}")
        sv_lines.append(",".join(row))
    sv_csv.write_text("\n".join(sv_lines))
    print(f"  saved {sv_csv}  ({sv_csv.stat().st_size / 1024:.1f} KB)")


# ───────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ───────────────────────────────────────────────────────────────────────


def run_evaluation(
    cache_dir: Path | None = None,
    norm_path: Path | None = None,
    models_dir: Path | None = None,
    results_dir: Path | None = None,
    device: str = "cpu",
    batch_size: int = 256,
    quick: bool = False,
) -> dict:
    cache_dir = Path(cache_dir or config.NEW_V2_EVAL_CACHE_DIR)
    norm_path = Path(norm_path or config.NEW_V2_EVAL_TRAINED_NORM_PATH)
    models_dir = Path(models_dir or config.NEW_V2_EVAL_TRAINED_MODELS_DIR)
    results_dir = Path(results_dir or config.NEW_V2_EVAL_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not norm_path.exists():
        raise FileNotFoundError(
            f"학습된 정규화 통계 없음: {norm_path}\n"
            "  → 학습된 fullstack 실험 (vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1) "
            "의 features/normalization.json 가 필요합니다."
        )
    print(f"[eval] norm stats:    {norm_path}")
    print(f"[eval] models dir:    {models_dir}")
    print(f"[eval] cache dir:     {cache_dir}")
    print(f"[eval] results dir:   {results_dir}")
    print(f"[eval] device:        {device}")

    norm = load_norm_params(norm_path)
    raw = load_caches(cache_dir)
    print(f"[eval] N_SV = {len(raw['features21'])}, "
          f"unique parts = {len(np.unique(raw['part_ids_sv']))}")

    if quick:
        # 첫 256 SV 만 평가 (빠른 sanity check)
        N_q = min(256, len(raw["features21"]))
        print(f"[quick] 첫 {N_q} SV 만 평가")
        for k in ("features21", "v0", "v1", "sensor", "dscnn", "cad_patch",
                  "scan_patch", "part_ids_sv", "cad_ratio", "voxel_indices",
                  "lengths"):
            raw[k] = raw[k][:N_q]

    ds_norm = apply_train_normalization(raw, norm)
    print(f"[eval] feats_static range "
          f"[{ds_norm['features_static'].min():.3f}, {ds_norm['features_static'].max():.3f}]")

    sv_preds = predict_all_props(
        ds_norm, norm,
        models_dir=models_dir,
        model_prefix=config.NEW_V2_EVAL_MODEL_FILE_PREFIX,
        device=device,
        props=config.TARGET_PROPERTIES,
        batch_size=batch_size,
    )

    if not sv_preds:
        raise RuntimeError("어떤 모델도 로드되지 않았습니다 — models_dir 확인")

    per_part = aggregate_per_part(
        sv_preds, raw["part_ids_sv"], cad_ratio=raw["cad_ratio"],
        weight_by_overlap=True,
    )
    print(f"[eval] aggregated to {len(per_part)} parts")

    gt = load_gt(config.new_v2_hdf5_path())
    metrics = compute_metrics(per_part, gt)
    print("\n[metrics]")
    for prop, m in metrics.items():
        print(f"  {prop}: RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} "
              f"naïve={m['naive_rmse']:.2f} n={m['n_parts']}")

    # save
    metrics_path = results_dir / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  saved {metrics_path}")

    save_predictions_csv(
        per_part, gt,
        out_path=results_dir / "per_part_predictions.csv",
        sv_part_ids=raw["part_ids_sv"],
        sv_predictions={p: sv_preds[p] for p in config.TARGET_PROPERTIES if p in sv_preds},
        cad_ratio=raw["cad_ratio"],
    )

    plot_scatters(metrics, results_dir)

    return {"metrics": metrics, "per_part": per_part, "sv_preds": sv_preds}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_evaluation(device=device, batch_size=args.batch_size, quick=args.quick)
