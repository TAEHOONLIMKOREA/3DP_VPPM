"""Phase S2 — VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-1DCNN-Sensor-4 데이터셋.

7-입력 결합:
  feats_static: (N, 2)              — baseline 21-feat 의 idx [2, 18] (build_height, laser_module)
  stacks_v0:    (N, 70, 8, 8)        — visible/0 카메라 패치, float16 (lstm 캐시 재사용)
  stacks_v1:    (N, 70, 8, 8)        — visible/1 카메라 패치, float16 (lstm_dual 캐시 재사용)
  sensors:      (N, 70, 7)           — temporal 7-ch raw (lstm_dual_img_4_sensor_7 캐시 재사용)
  dscnn:        (N, 70, 8)           — DSCNN 8-ch raw (lstm_dual_img_4_sensor_7_dscnn_8 캐시 재사용)
  cad_patch:    (N, 70, 2, 8, 8)     — edge_proximity + overhang_proximity, inverted+masked, float16
  scan_patch:   (N, 70, 2, 8, 8)     — return_delay + stripe_boundaries, raw + NaN→0, float16
  lengths:      (N,) int64           — 모든 시퀀스 입력 공통

여섯 캐시의 lengths/sv_indices/sample_ids 가 비트 단위 일치한다고 가정 (verify_*_v0_consistency 로 검증 후).

설계: PLAN.md §5
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import config
from ..common.dataset import normalize


# ───────────────────────────────────────────────────────────────────────
# 캐시 로딩
# ───────────────────────────────────────────────────────────────────────


def _load_cache(cache_path: Path, ds_key: str) -> dict:
    """단일 캐시 파일 로드. ds_key 외에 lengths/sv_indices/sample_ids 표준 키도 포함."""
    with h5py.File(cache_path, "r") as f:
        return {
            ds_key: f[ds_key][...],
            "lengths": f["lengths"][...],
            "sv_indices": f["sv_indices"][...],
            "sample_ids": f["sample_ids"][...],
        }


def load_septet_dataset(
    features_npz: Path = config.FEATURES_DIR / "all_features.npz",
    cache_v0_dir: Path = config.LSTM_FULL86_CACHE_V0_DIR,
    cache_v1_dir: Path = config.LSTM_FULL86_CACHE_V1_DIR,
    cache_sensor_dir: Path = config.LSTM_FULL86_CACHE_SENSOR_DIR,
    cache_dscnn_dir: Path = config.LSTM_FULL86_CACHE_DSCNN_DIR,
    cache_cad_dir: Path = config.LSTM_FULL86_CACHE_CAD_DIR,
    cache_scan_dir: Path = config.LSTM_FULL86_CACHE_SCAN_DIR,
    build_ids: list[str] | None = None,
) -> dict:
    """7-입력 데이터셋 로드.

    반환:
        features:    (N, 21) float32   — baseline 21-feat raw (정규화 전, static 추출 전)
        sample_ids:  (N,) int32
        build_ids:   (N,) int32
        targets:     {prop: (N,) float32}
        stacks_v0:   (N, 70, 8, 8) float16
        stacks_v1:   (N, 70, 8, 8) float16
        sensors:     (N, 70, 7) float32 raw
        dscnn:       (N, 70, 8) float32 raw
        cad_patch:   (N, 70, 2, 8, 8) float16  — inverted+masked
        scan_patch:  (N, 70, 2, 8, 8) float16  — raw
        lengths:     (N,) int64                — 모든 시퀀스 입력 공통
    """
    if build_ids is None:
        build_ids = list(config.BUILDS.keys())

    feat_data = np.load(features_npz)
    features = feat_data["features"]
    sample_ids = feat_data["sample_ids"]
    build_idx_arr = feat_data["build_ids"]
    targets = {
        p: feat_data[f"target_{p}"]
        for p in config.TARGET_PROPERTIES
        if f"target_{p}" in feat_data.files
    }
    N = len(features)

    cache_v0_paths = [Path(cache_v0_dir) / f"crop_stacks_{bid}.h5" for bid in build_ids]
    cache_v1_paths = [Path(cache_v1_dir) / f"crop_stacks_v1_{bid}.h5" for bid in build_ids]
    cache_sn_paths = [Path(cache_sensor_dir) / f"sensor_stacks_{bid}.h5" for bid in build_ids]
    cache_dn_paths = [Path(cache_dscnn_dir) / f"dscnn_stacks_{bid}.h5" for bid in build_ids]
    cache_cad_paths = [Path(cache_cad_dir) / f"cad_patch_stacks_{bid}.h5" for bid in build_ids]
    cache_scan_paths = [Path(cache_scan_dir) / f"scan_patch_stacks_{bid}.h5" for bid in build_ids]

    for paths, hint in (
        (cache_v0_paths, "`python -m Sources.vppm.lstm.run --phase cache`"),
        (cache_v1_paths, "`python -m Sources.vppm.lstm_dual.run --phase cache_v1`"),
        (cache_sn_paths, "`python -m Sources.vppm.lstm_dual_img_4_sensor_7.run --phase cache_sensor`"),
        (cache_dn_paths, "`python -m Sources.vppm.lstm_dual_img_4_sensor_7_dscnn_8.run --phase cache_dscnn`"),
        (cache_cad_paths, "`python -m Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.cache_cad_patch`"),
        (cache_scan_paths, "`python -m Sources.vppm.lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.cache_scan_patch`"),
    ):
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"캐시 누락: {p} — {hint} 먼저 실행")

    v0_list, v1_list, sn_list, dn_list, cad_list, scan_list = [], [], [], [], [], []
    lengths_list, cache_count_per_build = [], []
    for v0p, v1p, snp, dnp, cap, scp in zip(
        cache_v0_paths, cache_v1_paths, cache_sn_paths, cache_dn_paths,
        cache_cad_paths, cache_scan_paths,
    ):
        c0 = _load_cache(v0p, "stacks")
        c1 = _load_cache(v1p, "stacks")
        cs = _load_cache(snp, "sensors")
        cd = _load_cache(dnp, "dscnn")
        cc = _load_cache(cap, "cad_patch")
        ck = _load_cache(scp, "scan_patch")

        # 6-way lengths/sv_indices/sample_ids 일치 검증
        for tag, ref, other in (
            ("v1", c0["lengths"], c1["lengths"]),
            ("sensor", c0["lengths"], cs["lengths"]),
            ("dscnn", c0["lengths"], cd["lengths"]),
            ("cad", c0["lengths"], cc["lengths"]),
            ("scan", c0["lengths"], ck["lengths"]),
        ):
            if not np.array_equal(ref, other):
                raise RuntimeError(
                    f"lengths 불일치 (v0 vs {tag}) at build cache pair "
                    f"{v0p.name}: 캐시 재빌드 필요"
                )
        for tag, ref, other in (
            ("v1", c0["sv_indices"], c1["sv_indices"]),
            ("sensor", c0["sv_indices"], cs["sv_indices"]),
            ("dscnn", c0["sv_indices"], cd["sv_indices"]),
            ("cad", c0["sv_indices"], cc["sv_indices"]),
            ("scan", c0["sv_indices"], ck["sv_indices"]),
        ):
            if not np.array_equal(ref, other):
                raise RuntimeError(f"sv_indices 불일치 (v0 vs {tag}) at {v0p.name}")
        for tag, ref, other in (
            ("v1", c0["sample_ids"], c1["sample_ids"]),
            ("sensor", c0["sample_ids"], cs["sample_ids"]),
            ("dscnn", c0["sample_ids"], cd["sample_ids"]),
            ("cad", c0["sample_ids"], cc["sample_ids"]),
            ("scan", c0["sample_ids"], ck["sample_ids"]),
        ):
            if not np.array_equal(ref, other):
                raise RuntimeError(f"sample_ids 불일치 (v0 vs {tag}) at {v0p.name}")

        v0_list.append(c0["stacks"])
        v1_list.append(c1["stacks"])
        sn_list.append(cs["sensors"])
        dn_list.append(cd["dscnn"])
        cad_list.append(cc["cad_patch"])
        scan_list.append(ck["scan_patch"])
        lengths_list.append(c0["lengths"])
        cache_count_per_build.append(len(c0["lengths"]))

    cache_stacks_v0 = np.concatenate(v0_list, axis=0)
    cache_stacks_v1 = np.concatenate(v1_list, axis=0)
    cache_sensors = np.concatenate(sn_list, axis=0)
    cache_dscnn = np.concatenate(dn_list, axis=0)
    cache_cad = np.concatenate(cad_list, axis=0)
    cache_scan = np.concatenate(scan_list, axis=0)
    cache_lengths = np.concatenate(lengths_list, axis=0)

    # features.npz 와 캐시의 빌드별 SV 수 일치 확인
    build_id_to_idx = {bid: i for i, bid in enumerate(config.BUILDS.keys())}
    feat_count_per_build = []
    for bid in build_ids:
        bi = build_id_to_idx[bid]
        feat_count_per_build.append(int((build_idx_arr == bi).sum()))

    if feat_count_per_build != cache_count_per_build:
        raise ValueError(
            "features.npz 와 캐시의 빌드별 SV 수가 다릅니다.\n"
            f"  features: {feat_count_per_build}\n"
            f"  cache:    {cache_count_per_build}"
        )

    # features.npz 가 build_ids 순으로 정렬돼있는지 확인 → 아니면 정렬
    sorted_order = []
    for bid in build_ids:
        bi = build_id_to_idx[bid]
        sorted_order.extend(np.where(build_idx_arr == bi)[0].tolist())
    sorted_order = np.array(sorted_order, dtype=np.int64)

    if not np.array_equal(sorted_order, np.arange(N)):
        features = features[sorted_order]
        sample_ids = sample_ids[sorted_order]
        build_idx_arr = build_idx_arr[sorted_order]
        targets = {k: v[sorted_order] for k, v in targets.items()}

    if (len(cache_stacks_v0) != N or len(cache_stacks_v1) != N
            or len(cache_sensors) != N or len(cache_dscnn) != N
            or len(cache_cad) != N or len(cache_scan) != N):
        raise ValueError(
            f"매칭 실패: features N={N}, "
            f"v0={len(cache_stacks_v0)}, v1={len(cache_stacks_v1)}, "
            f"sensor={len(cache_sensors)}, dscnn={len(cache_dscnn)}, "
            f"cad={len(cache_cad)}, scan={len(cache_scan)}"
        )

    return {
        "features": features.astype(np.float32),
        "sample_ids": sample_ids.astype(np.int32),
        "build_ids": build_idx_arr.astype(np.int32),
        "targets": {k: v.astype(np.float32) for k, v in targets.items()},
        "stacks_v0": cache_stacks_v0,
        "stacks_v1": cache_stacks_v1,
        "sensors": cache_sensors.astype(np.float32),
        "dscnn": cache_dscnn.astype(np.float32),
        "cad_patch": cache_cad,           # float16 보존
        "scan_patch": cache_scan,         # float16 보존
        "lengths": cache_lengths.astype(np.int64),
    }


# ───────────────────────────────────────────────────────────────────────
# 정규화
# ───────────────────────────────────────────────────────────────────────


def _per_channel_min_max_seq(seq: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, T_max, C) 시퀀스 → 채널별 (C,) min/max. 패딩 (lengths 이후) 제외."""
    N, _, C = seq.shape
    mins, maxs = np.zeros(C, dtype=np.float32), np.zeros(C, dtype=np.float32)
    for c in range(C):
        vals = []
        for i in range(N):
            T_sv = int(lengths[i])
            if T_sv > 0:
                vals.append(seq[i, :T_sv, c])
        flat = np.concatenate(vals) if vals else np.array([0.0, 1.0], dtype=np.float32)
        mins[c] = float(flat.min())
        maxs[c] = float(flat.max())
    return mins, maxs


def _per_channel_min_max_patch(
    patch: np.ndarray, lengths: np.ndarray, exclude_zero: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """(N, T_max, C, H, W) 패치 시퀀스 → 채널별 (C,) min/max. 패딩 제외.

    exclude_zero=True: 0 픽셀 (cad mask 곱 결과의 분말 영역) 도 통계에서 제외 → 의미 픽셀 분포로 정규화.
    """
    N, _, C, _, _ = patch.shape
    mins, maxs = np.zeros(C, dtype=np.float32), np.zeros(C, dtype=np.float32)
    for c in range(C):
        vals = []
        for i in range(N):
            T_sv = int(lengths[i])
            if T_sv > 0:
                arr = patch[i, :T_sv, c].astype(np.float32).reshape(-1)
                if exclude_zero:
                    arr = arr[arr != 0.0]
                if arr.size > 0:
                    vals.append(arr)
        if vals:
            flat = np.concatenate(vals)
            mins[c] = float(flat.min())
            maxs[c] = float(flat.max())
        else:
            mins[c], maxs[c] = 0.0, 1.0
    return mins, maxs


def build_normalized_dataset(raw: dict) -> dict:
    """static 2-feat (build_height, laser_module) + 6 시퀀스 입력 정규화.

    - static: 21-feat 에서 idx [2, 18] 추출 → min-max [-1, 1]
    - sensors / dscnn:        per-channel min-max [-1, 1] (패딩 0 제외)
    - cad_patch:              per-channel min-max [-1, 1] (패딩 + **0 픽셀 (분말)** 제외)
    - scan_patch:             per-channel min-max [-1, 1] (패딩 0 제외; raw 0 = no melt 도 통계 포함)
    - 카메라 stacks_v0/v1:    raw 보존 (CNN 안에서 처리)
    - 타겟:                   per-property min-max [-1, 1]

    cad_patch 의 0 픽셀 제외는 PLAN §11.6 의 결정 — 분말 영역 0 이 통계에 들어가면 분포 기형.
    """
    feats21 = raw["features"]
    sids = raw["sample_ids"]
    bids = raw["build_ids"]
    sv0 = raw["stacks_v0"]
    sv1 = raw["stacks_v1"]
    sensors = raw["sensors"]
    dscnn = raw["dscnn"]
    cad_patch = raw["cad_patch"]
    scan_patch = raw["scan_patch"]
    lengths = raw["lengths"]
    tgts = raw["targets"]

    # NaN / failed sample 마스크
    uts = tgts.get("ultimate_tensile_strength", np.zeros(len(feats21)))
    valid = ~np.isnan(uts) & (uts >= 50.0)
    for prop in config.TARGET_PROPERTIES:
        if prop in tgts:
            valid &= ~np.isnan(tgts[prop])
    valid &= ~np.isnan(feats21).any(axis=1)
    valid &= (lengths > 0)

    feats21 = feats21[valid]
    sids = sids[valid]
    bids = bids[valid]
    sv0 = sv0[valid]
    sv1 = sv1[valid]
    sensors = sensors[valid]
    dscnn = dscnn[valid]
    cad_patch = cad_patch[valid]
    scan_patch = scan_patch[valid]
    lengths = lengths[valid]
    tgts = {k: v[valid] for k, v in tgts.items()}

    # === Static 2-feat (build_height + laser_module) — 21-feat 에서 idx [2, 18] 추출 ===
    static_idx = np.array(config.LSTM_FULL86_STATIC_IDX, dtype=np.int64)        # [2, 18]
    feats_static = feats21[:, static_idx]
    s_min = feats_static.min(axis=0)
    s_max = feats_static.max(axis=0)
    feats_static_norm = normalize(feats_static, s_min, s_max).astype(np.float32)

    # === Sensor (N, T, 7) per-channel min-max (패딩 0 제외) ===
    sensor_min, sensor_max = _per_channel_min_max_seq(sensors, lengths)
    sensors_norm = normalize(sensors, sensor_min, sensor_max).astype(np.float32)

    # === DSCNN (N, T, 8) per-channel min-max (패딩 0 제외) ===
    dscnn_min, dscnn_max = _per_channel_min_max_seq(dscnn, lengths)
    dscnn_norm = normalize(dscnn, dscnn_min, dscnn_max).astype(np.float32)

    # === CAD patch (N, T, 2, 8, 8) per-channel min-max (패딩 + 0 픽셀 제외) ===
    cad_min, cad_max = _per_channel_min_max_patch(cad_patch, lengths, exclude_zero=True)
    # normalize 는 (..., C) 마지막 축 broadcast 가정 → patch reshape 후 적용
    cad_patch_f32 = cad_patch.astype(np.float32)
    # broadcast: (1, 1, C, 1, 1)
    cad_patch_norm = (
        2 * (cad_patch_f32 - cad_min[None, None, :, None, None])
        / (cad_max[None, None, :, None, None] - cad_min[None, None, :, None, None] + 1e-8)
        - 1
    ).astype(np.float32)

    # === Scan patch (N, T, 2, 8, 8) per-channel min-max (패딩 0 제외; raw 0 포함) ===
    scan_min, scan_max = _per_channel_min_max_patch(scan_patch, lengths, exclude_zero=False)
    scan_patch_f32 = scan_patch.astype(np.float32)
    scan_patch_norm = (
        2 * (scan_patch_f32 - scan_min[None, None, :, None, None])
        / (scan_max[None, None, :, None, None] - scan_min[None, None, :, None, None] + 1e-8)
        - 1
    ).astype(np.float32)

    # === 타겟 [-1, 1] 정규화 ===
    tgt_norm = {}
    t_min, t_max = {}, {}
    for prop in config.TARGET_PROPERTIES:
        if prop in tgts:
            t_min[prop] = float(tgts[prop].min())
            t_max[prop] = float(tgts[prop].max())
            tgt_norm[prop] = normalize(tgts[prop], t_min[prop], t_max[prop]).astype(np.float32)

    norm_params = {
        "static_min": s_min.tolist(),
        "static_max": s_max.tolist(),
        "static_idx": static_idx.tolist(),
        "sensor_min": sensor_min.tolist(),
        "sensor_max": sensor_max.tolist(),
        "dscnn_min": dscnn_min.tolist(),
        "dscnn_max": dscnn_max.tolist(),
        "cad_min": cad_min.tolist(),
        "cad_max": cad_max.tolist(),
        "cad_exclude_zero_in_stats": True,
        "scan_min": scan_min.tolist(),
        "scan_max": scan_max.tolist(),
        "scan_exclude_zero_in_stats": False,
        "target_min": t_min,
        "target_max": t_max,
    }

    return {
        "features_static": feats_static_norm,
        "features_static_raw": feats_static,
        "sample_ids": sids,
        "build_ids": bids,
        "stacks_v0": sv0,
        "stacks_v1": sv1,
        "sensors": sensors_norm,
        "sensors_raw": sensors,
        "dscnn": dscnn_norm,
        "dscnn_raw": dscnn,
        "cad_patch": cad_patch_norm,
        "cad_patch_raw": cad_patch,
        "scan_patch": scan_patch_norm,
        "scan_patch_raw": scan_patch,
        "lengths": lengths,
        "targets": tgt_norm,
        "targets_raw": tgts,
        "norm_params": norm_params,
    }


# ───────────────────────────────────────────────────────────────────────
# Dataset / collate
# ───────────────────────────────────────────────────────────────────────


class VPPMLSTMSeptetDataset(Dataset):
    """7-입력 한 SV 반환: (feats_static, stack_v0, stack_v1, sensor, dscnn, cad, scan, length, target)."""

    def __init__(
        self,
        features_static: np.ndarray,        # (N, 2) float32 normalized
        stacks_v0: np.ndarray,              # (N, 70, 8, 8) float16
        stacks_v1: np.ndarray,              # (N, 70, 8, 8) float16
        sensors: np.ndarray,                # (N, 70, 7) float32 normalized
        dscnn: np.ndarray,                  # (N, 70, 8) float32 normalized
        cad_patch: np.ndarray,              # (N, 70, 2, 8, 8) float32 normalized
        scan_patch: np.ndarray,             # (N, 70, 2, 8, 8) float32 normalized
        lengths: np.ndarray,                # (N,) int64
        targets: np.ndarray,                # (N,) float32 normalized
    ):
        self.features_static = torch.from_numpy(features_static).float()
        self.stacks_v0 = torch.from_numpy(stacks_v0)            # float16 → 학습 시 float32 캐스팅
        self.stacks_v1 = torch.from_numpy(stacks_v1)
        self.sensors = torch.from_numpy(sensors).float()
        self.dscnn = torch.from_numpy(dscnn).float()
        self.cad_patch = torch.from_numpy(cad_patch).float()
        self.scan_patch = torch.from_numpy(scan_patch).float()
        self.lengths = torch.from_numpy(lengths.astype(np.int64))
        self.targets = torch.from_numpy(targets).float().unsqueeze(1)

    def __len__(self):
        return len(self.features_static)

    def __getitem__(self, i):
        return (
            self.features_static[i],
            self.stacks_v0[i],
            self.stacks_v1[i],
            self.sensors[i],
            self.dscnn[i],
            self.cad_patch[i],
            self.scan_patch[i],
            self.lengths[i],
            self.targets[i],
        )


def collate_fn(batch):
    feats_static = torch.stack([b[0] for b in batch], dim=0)
    sv0 = torch.stack([b[1] for b in batch], dim=0).float()        # float16 → float32
    sv1 = torch.stack([b[2] for b in batch], dim=0).float()
    sensors = torch.stack([b[3] for b in batch], dim=0)
    dscnn = torch.stack([b[4] for b in batch], dim=0)
    cad_patch = torch.stack([b[5] for b in batch], dim=0)
    scan_patch = torch.stack([b[6] for b in batch], dim=0)
    lengths = torch.stack([b[7] for b in batch], dim=0)
    targets = torch.stack([b[8] for b in batch], dim=0)
    return feats_static, sv0, sv1, sensors, dscnn, cad_patch, scan_patch, lengths, targets
