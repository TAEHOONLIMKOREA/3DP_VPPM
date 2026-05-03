"""Phase S2 — VPPM-LSTM-Dual-Img-4-Sensor-7-DSCNN-8 데이터셋.

baseline `all_features.npz` (21 feat) 에서 G1(DSCNN 8) + G2(센서 7) 를 모두 제거한 6 feat
+ visible/0 / visible/1 카메라 시퀀스 + 7-channel sensor 시퀀스 + 8-channel DSCNN 시퀀스 결합.

네 캐시(v0/v1/sensor/dscnn) 의 lengths / sv_indices / sample_ids 가 비트 단위 일치한다고
가정 (cache_dscnn.verify_dscnn_v0_consistency / cache_sensor.verify_v0_consistency /
lstm_dual.crop_stacks_v1.verify_v0_v1_consistency 로 검증).

sensor 캐시는 sensor_7 디렉터리(`config.LSTM_DUAL_IMG_4_SENSOR_7_CACHE_DIR`) 에서 재사용.
dscnn 캐시는 본 실험 디렉터리(`config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_DIR`) 에서 로드.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import config
from ..common.dataset import normalize


def _load_cache(cache_path: Path, ds_key: str) -> dict:
    with h5py.File(cache_path, "r") as f:
        return {
            ds_key: f[ds_key][...],
            "lengths": f["lengths"][...],
            "sv_indices": f["sv_indices"][...],
            "sample_ids": f["sample_ids"][...],
        }


def load_pent_dataset(
    features_npz: Path = config.FEATURES_DIR / "all_features.npz",
    cache_v0_dir: Path = config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_V0_DIR,
    cache_v1_dir: Path = config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_V1_DIR,
    cache_sensor_dir: Path = config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_SENSOR_DIR,
    cache_dscnn_dir: Path = config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_DIR,
    build_ids: list[str] | None = None,
) -> dict:
    """baseline features.npz + v0/v1/sensor/dscnn 4 캐시 결합.

    반환 dict 의 features 는 21-feat raw (G1+G2 제거 전). build_normalized_dataset 에서 6-feat 로 자른다.

    반환:
        features:    (N, 21) float32   — baseline 21 피처 (raw, 정규화 전, G1+G2 제거 전)
        sample_ids:  (N,) int32
        build_ids:   (N,) int32
        targets:     {prop: (N,) float32}
        stacks_v0:   (N, 70, 8, 8) float16   — visible/0 padded 시퀀스
        stacks_v1:   (N, 70, 8, 8) float16   — visible/1 padded 시퀀스
        sensors:     (N, 70, 7) float32       — sensor padded 시퀀스 (raw)
        dscnn:       (N, 70, 8) float32       — DSCNN padded 시퀀스 (raw)
        lengths:     (N,) int64               — 네 캐시 공통
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

    for p, hint in (
        (cache_v0_paths, "`python -m Sources.vppm.lstm.run --phase cache`"),
        (cache_v1_paths, "`python -m Sources.vppm.lstm_dual.run --phase cache_v1`"),
        (cache_sn_paths, "`python -m Sources.vppm.lstm_dual_img_4_sensor_7.run --phase cache_sensor`"),
        (cache_dn_paths, "`python -m Sources.vppm.lstm_dual_img_4_sensor_7_dscnn_8.run --phase cache_dscnn`"),
    ):
        for f in p:
            if not f.exists():
                raise FileNotFoundError(f"캐시 누락: {f} — {hint} 먼저 실행")

    v0_stacks_list, v1_stacks_list, sensor_list, dscnn_list, lengths_list = [], [], [], [], []
    cache_sv_count_per_build = []
    for v0p, v1p, snp, dnp in zip(cache_v0_paths, cache_v1_paths, cache_sn_paths, cache_dn_paths):
        c0 = _load_cache(v0p, "stacks")
        c1 = _load_cache(v1p, "stacks")
        cs = _load_cache(snp, "sensors")
        cd = _load_cache(dnp, "dscnn")

        # 빌드 내부에서 v0/v1/sensor/dscnn lengths 일치 검증
        if not (np.array_equal(c0["lengths"], c1["lengths"])
                and np.array_equal(c0["lengths"], cs["lengths"])
                and np.array_equal(c0["lengths"], cd["lengths"])):
            raise RuntimeError(
                f"v0/v1/sensor/dscnn lengths 불일치 — "
                f"{v0p.name} / {v1p.name} / {snp.name} / {dnp.name}.\n"
                "캐시를 다시 빌드하세요 "
                "(lstm_dual_img_4_sensor_7_dscnn_8.cache_dscnn.verify_dscnn_v0_consistency)"
            )
        if not (np.array_equal(c0["sv_indices"], c1["sv_indices"])
                and np.array_equal(c0["sv_indices"], cs["sv_indices"])
                and np.array_equal(c0["sv_indices"], cd["sv_indices"])):
            raise RuntimeError(
                f"sv_indices 불일치 — {v0p.name} / {v1p.name} / {snp.name} / {dnp.name}"
            )
        if not (np.array_equal(c0["sample_ids"], c1["sample_ids"])
                and np.array_equal(c0["sample_ids"], cs["sample_ids"])
                and np.array_equal(c0["sample_ids"], cd["sample_ids"])):
            raise RuntimeError(
                f"sample_ids 불일치 — {v0p.name} / {v1p.name} / {snp.name} / {dnp.name}"
            )

        v0_stacks_list.append(c0["stacks"])
        v1_stacks_list.append(c1["stacks"])
        sensor_list.append(cs["sensors"])
        dscnn_list.append(cd["dscnn"])
        lengths_list.append(c0["lengths"])
        cache_sv_count_per_build.append(len(c0["lengths"]))

    cache_stacks_v0 = np.concatenate(v0_stacks_list, axis=0)
    cache_stacks_v1 = np.concatenate(v1_stacks_list, axis=0)
    cache_sensors = np.concatenate(sensor_list, axis=0)
    cache_dscnn = np.concatenate(dscnn_list, axis=0)
    cache_lengths = np.concatenate(lengths_list, axis=0)

    # features.npz 와 캐시의 빌드별 SV 수 일치 검증
    build_id_to_idx = {bid: i for i, bid in enumerate(config.BUILDS.keys())}
    feat_count_per_build = []
    for bid in build_ids:
        bi = build_id_to_idx[bid]
        feat_count_per_build.append(int((build_idx_arr == bi).sum()))

    if feat_count_per_build != cache_sv_count_per_build:
        raise ValueError(
            "features.npz 와 캐시의 빌드별 SV 수가 다릅니다.\n"
            f"  features: {feat_count_per_build}\n"
            f"  cache:    {cache_sv_count_per_build}"
        )

    # features.npz 가 build_ids 순으로 정렬돼있는지 확인 → 아니면 정렬 후 매칭
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

    if (len(cache_stacks_v0) != N
            or len(cache_stacks_v1) != N
            or len(cache_sensors) != N
            or len(cache_dscnn) != N):
        raise ValueError(
            f"매칭 실패: features N={N}, "
            f"v0 N={len(cache_stacks_v0)}, v1 N={len(cache_stacks_v1)}, "
            f"sensor N={len(cache_sensors)}, dscnn N={len(cache_dscnn)}"
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
        "lengths": cache_lengths.astype(np.int64),
    }


def build_normalized_dataset(raw: dict) -> dict:
    """G1(DSCNN) + G2(센서) 둘 다 제거 → 6-feat normalize.
    sensor / dscnn 각각 per-channel min-max [-1, 1] 정규화.

    sensor / dscnn min/max 는 패딩 영역(=0) 을 제외한 실제 시퀀스 값들로부터 계산.
    """
    feats21 = raw["features"]
    sids = raw["sample_ids"]
    bids = raw["build_ids"]
    sv0 = raw["stacks_v0"]
    sv1 = raw["stacks_v1"]
    sensors = raw["sensors"]                                    # (N, T_max, 7)
    dscnn = raw["dscnn"]                                        # (N, T_max, 8)
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
    lengths = lengths[valid]
    tgts = {k: v[valid] for k, v in tgts.items()}

    # G1(DSCNN) + G2(센서) 인덱스를 21-feat 에서 제거 → 6-feat (CAD 3 + Scan 3)
    sensor_idx = config.FEATURE_GROUPS["sensor"]                 # [11..17]
    dscnn_idx = config.FEATURE_GROUPS["dscnn"]                   # [3..10]
    remove_idx = set(sensor_idx) | set(dscnn_idx)
    keep_idx = np.array(
        [i for i in range(config.N_FEATURES) if i not in remove_idx],
        dtype=np.int64,
    )
    # keep_idx == [0, 1, 2, 18, 19, 20]
    feats6 = feats21[:, keep_idx]

    # 6-feat min-max [-1, 1] 정규화
    f_min = feats6.min(axis=0)
    f_max = feats6.max(axis=0)
    feats6_norm = normalize(feats6, f_min, f_max).astype(np.float32)

    # sensor per-channel min-max [-1, 1] (패딩 0 제외)
    n_sn = sensors.shape[2]
    sensor_mins, sensor_maxs = [], []
    for c in range(n_sn):
        vals = []
        for i in range(len(sensors)):
            T_sv = int(lengths[i])
            if T_sv > 0:
                vals.append(sensors[i, :T_sv, c])
        flat = np.concatenate(vals) if vals else np.array([0.0, 1.0], dtype=np.float32)
        sensor_mins.append(float(flat.min()))
        sensor_maxs.append(float(flat.max()))
    sensor_min = np.array(sensor_mins, dtype=np.float32)
    sensor_max = np.array(sensor_maxs, dtype=np.float32)
    sensors_norm = normalize(sensors, sensor_min, sensor_max).astype(np.float32)

    # dscnn per-channel min-max [-1, 1] (패딩 0 제외)
    n_dn = dscnn.shape[2]
    dscnn_mins, dscnn_maxs = [], []
    for c in range(n_dn):
        vals = []
        for i in range(len(dscnn)):
            T_sv = int(lengths[i])
            if T_sv > 0:
                vals.append(dscnn[i, :T_sv, c])
        flat = np.concatenate(vals) if vals else np.array([0.0, 1.0], dtype=np.float32)
        dscnn_mins.append(float(flat.min()))
        dscnn_maxs.append(float(flat.max()))
    dscnn_min = np.array(dscnn_mins, dtype=np.float32)
    dscnn_max = np.array(dscnn_maxs, dtype=np.float32)
    dscnn_norm = normalize(dscnn, dscnn_min, dscnn_max).astype(np.float32)
    # 패딩 영역도 [-1,1] 어딘가로 매핑되지만, LSTM 은 lengths 로 packed 되어 패딩 영역을 보지 않음.

    # 타겟 [-1, 1] 정규화
    tgt_norm = {}
    t_min, t_max = {}, {}
    for prop in config.TARGET_PROPERTIES:
        if prop in tgts:
            t_min[prop] = float(tgts[prop].min())
            t_max[prop] = float(tgts[prop].max())
            tgt_norm[prop] = normalize(tgts[prop], t_min[prop], t_max[prop]).astype(np.float32)

    norm_params = {
        "feature_min": f_min.tolist(),
        "feature_max": f_max.tolist(),
        "feature_keep_idx": keep_idx.tolist(),
        "sensor_min": sensor_min.tolist(),
        "sensor_max": sensor_max.tolist(),
        "dscnn_min": dscnn_min.tolist(),
        "dscnn_max": dscnn_max.tolist(),
        "target_min": t_min,
        "target_max": t_max,
    }

    return {
        "features": feats6_norm,
        "features_raw": feats6,
        "sample_ids": sids,
        "build_ids": bids,
        "stacks_v0": sv0,
        "stacks_v1": sv1,
        "sensors": sensors_norm,
        "sensors_raw": sensors,
        "dscnn": dscnn_norm,
        "dscnn_raw": dscnn,
        "lengths": lengths,
        "targets": tgt_norm,
        "targets_raw": tgts,
        "norm_params": norm_params,
    }


class VPPMLSTMPentDataset(Dataset):
    """(features6, stack_v0, stack_v1, sensor_seq, dscnn_seq, length, target) 한 쌍 반환."""

    def __init__(self, features: np.ndarray,
                 stacks_v0: np.ndarray, stacks_v1: np.ndarray,
                 sensors: np.ndarray, dscnn: np.ndarray,
                 lengths: np.ndarray, targets: np.ndarray):
        # features:  (N, 6) float32 normalized
        # stacks_v*: (N, 70, 8, 8) float16 padded
        # sensors:   (N, 70, 7) float32 normalized padded
        # dscnn:     (N, 70, 8) float32 normalized padded
        # lengths:   (N,) int64
        # targets:   (N,) float32 normalized
        self.features = torch.from_numpy(features).float()
        self.stacks_v0 = torch.from_numpy(stacks_v0)             # float16 → 학습 시 float32 캐스팅
        self.stacks_v1 = torch.from_numpy(stacks_v1)
        self.sensors = torch.from_numpy(sensors).float()
        self.dscnn = torch.from_numpy(dscnn).float()
        self.lengths = torch.from_numpy(lengths.astype(np.int64))
        self.targets = torch.from_numpy(targets).float().unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return (self.features[i], self.stacks_v0[i], self.stacks_v1[i],
                self.sensors[i], self.dscnn[i], self.lengths[i], self.targets[i])


def collate_fn(batch):
    feats = torch.stack([b[0] for b in batch], dim=0)
    sv0 = torch.stack([b[1] for b in batch], dim=0).float()       # float16 → float32
    sv1 = torch.stack([b[2] for b in batch], dim=0).float()
    sn = torch.stack([b[3] for b in batch], dim=0)
    dn = torch.stack([b[4] for b in batch], dim=0)
    lengths = torch.stack([b[5] for b in batch], dim=0)
    targets = torch.stack([b[6] for b in batch], dim=0)
    return feats, sv0, sv1, sn, dn, lengths, targets
