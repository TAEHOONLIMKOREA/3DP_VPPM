"""[new_v2] 단일 빌드용 6 시퀀스 캐시 빌더.

기존 6 개 캐시 빌더 (`Sources/vppm/{lstm,lstm_dual,lstm_dual_img_4_sensor_7,
lstm_dual_img_4_sensor_7_dscnn_8,lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4}/cache_*.py`)
와 1:1 동등한 로직을 사용하되:

  - hdf5_path  → config.new_v2_hdf5_path()
  - build_id   → config.NEW_V2_BUILD_ID  (캐시 파일명 접미사)
  - valid SV   → supervoxel_partbased.find_valid_supervoxels_partbased
                 (sample_ids 무시, part_ids overlap 만 본다)

산출물 6 종 (NEW_V2_EVAL_CACHE_DIR):
  crop_stacks_{B}.h5         (visible/0)
  crop_stacks_v1_{B}.h5      (visible/1)
  sensor_stacks_{B}.h5       (7-ch temporal)
  dscnn_stacks_{B}.h5        (8-ch DSCNN)
  cad_patch_stacks_{B}.h5    (2-ch edge_proximity / overhang_proximity)
  scan_patch_stacks_{B}.h5   (2-ch return_delay / stripe_boundaries)

여기서 {B} = NEW_V2_BUILD_ID.

valid_layer_rule (모든 캐시 공통): part_ids > 0 in SV xy region.
캐시 6 종의 `lengths / sv_indices / sample_ids` 는 비트 단위 일치 (같은 valid_voxels 사용).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from tqdm import tqdm

from ..baseline.scan_features import (
    build_melt_time_map,
    compute_return_delay_map,
    compute_stripe_boundaries_map,
)
from ..common import config
from ..common.supervoxel import SuperVoxelGrid
from .supervoxel_partbased import find_valid_supervoxels_partbased


# ───────────────────────────────────────────────────────────────────────
# 공통 helpers
# ───────────────────────────────────────────────────────────────────────


def _setup(out_dir: Path) -> tuple[str, SuperVoxelGrid, dict]:
    """new_v2 hdf5 path + grid + valid_voxels 일괄 준비. 6 빌더가 공유."""
    hdf5 = str(config.new_v2_hdf5_path())
    grid = SuperVoxelGrid.from_hdf5(hdf5)
    valid = find_valid_supervoxels_partbased(grid, hdf5)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[setup] hdf5={Path(hdf5).name}, grid nx={grid.nx} ny={grid.ny} nz={grid.nz}, "
        f"valid SVs={len(valid['voxel_indices'])}"
    )
    return hdf5, grid, valid


def _iz_to_svs(indices: np.ndarray) -> dict[int, list[int]]:
    iz_map: dict[int, list[int]] = {}
    for i, (_, _, iz) in enumerate(indices):
        iz_map.setdefault(int(iz), []).append(i)
    return iz_map


# ───────────────────────────────────────────────────────────────────────
# 1) 카메라 visible/{channel} — crop_stacks
# ───────────────────────────────────────────────────────────────────────


def build_camera_cache(out_dir: Path, channel: int, file_prefix: str) -> Path:
    bid = config.NEW_V2_BUILD_ID
    out_path = Path(out_dir) / f"{file_prefix}_{bid}.h5"
    if out_path.exists():
        print(f"[cache] {out_path.name} already exists, skip")
        return out_path

    hdf5, grid, valid = _setup(out_dir)
    indices = valid["voxel_indices"]
    N = len(indices)
    if N == 0:
        print(f"  {bid}: no valid SVs, skip")
        return out_path

    T_max = config.LSTM_T_MAX
    H = config.LSTM_CROP_H
    W = config.LSTM_CROP_W
    cam_key = f"slices/camera_data/visible/{channel}"

    iz_map = _iz_to_svs(indices)
    print(f"  {bid}: N={N} SVs, {len(iz_map)} z-blocks, channel={channel}")

    with h5py.File(out_path, "w") as out, h5py.File(hdf5, "r") as f:
        cam = f[cam_key]
        part_ds = f["slices/part_ids"]
        stacks = out.create_dataset(
            "stacks", shape=(N, T_max, H, W), dtype="float16",
            compression="gzip", compression_opts=4, chunks=(1, T_max, H, W),
        )
        lengths = out.create_dataset("lengths", shape=(N,), dtype="int16")
        out.create_dataset("sv_indices", data=indices.astype(np.int32))
        out.create_dataset("sample_ids", data=valid["sample_ids"].astype(np.int32))
        out.create_dataset("part_ids", data=valid["part_ids"].astype(np.int32))

        for iz in tqdm(sorted(iz_map.keys()), desc=f"{bid} z-blocks (cam{channel})"):
            l0, l1 = grid.get_layer_range(iz)
            block_cam = cam[l0:l1].astype(np.float16)
            # baseline 카메라 dtype 은 uint8 (255 normalize), new_v2 는 float32 (이미 0..1 또는 0..255?).
            # 일관성: max>1.5 일 때 /255 적용 (baseline), 아니면 raw 보존.
            if float(block_cam.max()) > 1.5:
                block_cam = block_cam / np.float16(255.0)
            block_part = part_ds[l0:l1]
            Tb = block_cam.shape[0]

            for sv_i in iz_map[iz]:
                ix, iy, _ = indices[sv_i]
                r0, r1, c0, c1 = grid.get_pixel_range(int(ix), int(iy))
                cam_crop = block_cam[:, r0:r1, c0:c1]
                part_crop = block_part[:, r0:r1, c0:c1]

                valid_mask = (part_crop > 0).reshape(Tb, -1).any(axis=1)
                if not valid_mask.any():
                    valid_mask[Tb // 2] = True

                seq = cam_crop[valid_mask]
                T_sv = seq.shape[0]
                if seq.shape[1] < H or seq.shape[2] < W:
                    padded = np.zeros((T_sv, H, W), dtype=np.float16)
                    padded[:, : seq.shape[1], : seq.shape[2]] = seq
                    seq = padded
                elif seq.shape[1] > H or seq.shape[2] > W:
                    seq = seq[:, :H, :W]
                T_sv_clip = min(T_sv, T_max)
                stacks[sv_i, :T_sv_clip] = seq[:T_sv_clip]
                lengths[sv_i] = T_sv_clip

        out.attrs["T_max"] = T_max
        out.attrs["H"] = H
        out.attrs["W"] = W
        out.attrs["camera_channel"] = channel
        out.attrs["build_id"] = bid
        out.attrs["n_sv"] = N
        out.attrs["valid_layer_rule"] = "part_ids>0 in SV xy region"

    print(f"  saved {out_path}  ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path


def build_v0(out_dir: Path | None = None) -> Path:
    return build_camera_cache(out_dir or config.NEW_V2_EVAL_CACHE_DIR,
                              channel=0, file_prefix="crop_stacks")


def build_v1(out_dir: Path | None = None) -> Path:
    return build_camera_cache(out_dir or config.NEW_V2_EVAL_CACHE_DIR,
                              channel=1, file_prefix="crop_stacks_v1")


# ───────────────────────────────────────────────────────────────────────
# 2) Sensor — sensor_stacks
# ───────────────────────────────────────────────────────────────────────


def build_sensor(out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or config.NEW_V2_EVAL_CACHE_DIR)
    bid = config.NEW_V2_BUILD_ID
    out_path = out_dir / f"sensor_stacks_{bid}.h5"
    if out_path.exists():
        print(f"[cache] {out_path.name} already exists, skip")
        return out_path

    hdf5, grid, valid = _setup(out_dir)
    indices = valid["voxel_indices"]
    N = len(indices)
    if N == 0:
        return out_path

    T_max = config.LSTM_T_MAX
    n_ch = len(config.TEMPORAL_FEATURES)
    iz_map = _iz_to_svs(indices)

    with h5py.File(out_path, "w") as out, h5py.File(hdf5, "r") as f:
        temporal_arrays = []
        for key in config.TEMPORAL_FEATURES:
            full_key = f"temporal/{key}"
            if full_key not in f:
                raise KeyError(f"new_v2 HDF5 에 {full_key} 없음")
            temporal_arrays.append(f[full_key][...].astype(np.float32))
        lengths_per_ch = [len(t) for t in temporal_arrays]
        if len(set(lengths_per_ch)) != 1:
            raise RuntimeError(f"temporal 채널 길이 불일치: {lengths_per_ch}")
        temporals = np.stack(temporal_arrays, axis=0)        # (7, num_layers)

        part_ds = f["slices/part_ids"]
        sensors = out.create_dataset(
            "sensors", shape=(N, T_max, n_ch), dtype="float32",
            compression="gzip", compression_opts=4, chunks=(1, T_max, n_ch),
        )
        lengths = out.create_dataset("lengths", shape=(N,), dtype="int16")
        out.create_dataset("sv_indices", data=indices.astype(np.int32))
        out.create_dataset("sample_ids", data=valid["sample_ids"].astype(np.int32))
        out.create_dataset("part_ids", data=valid["part_ids"].astype(np.int32))

        for iz in tqdm(sorted(iz_map.keys()), desc=f"{bid} z-blocks (sensor)"):
            l0, l1 = grid.get_layer_range(iz)
            block_part = part_ds[l0:l1]
            block_temp = temporals[:, l0:l1].T                # (Tb, 7)
            Tb = block_part.shape[0]
            for sv_i in iz_map[iz]:
                ix, iy, _ = indices[sv_i]
                r0, r1, c0, c1 = grid.get_pixel_range(int(ix), int(iy))
                part_crop = block_part[:, r0:r1, c0:c1]
                valid_mask = (part_crop > 0).reshape(Tb, -1).any(axis=1)
                if not valid_mask.any():
                    valid_mask[Tb // 2] = True
                seq = block_temp[valid_mask]
                T_sv = seq.shape[0]
                T_sv_clip = min(T_sv, T_max)
                sensors[sv_i, :T_sv_clip] = seq[:T_sv_clip]
                lengths[sv_i] = T_sv_clip

        out.attrs["T_max"] = T_max
        out.attrs["n_channels"] = n_ch
        out.attrs["channel_names"] = np.array(config.TEMPORAL_FEATURES, dtype="S")
        out.attrs["build_id"] = bid
        out.attrs["n_sv"] = N
        out.attrs["valid_layer_rule"] = "part_ids>0 in SV xy region"

    print(f"  saved {out_path}  ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path


# ───────────────────────────────────────────────────────────────────────
# 3) DSCNN — dscnn_stacks (8 paper class)
# ───────────────────────────────────────────────────────────────────────


def build_dscnn(out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or config.NEW_V2_EVAL_CACHE_DIR)
    bid = config.NEW_V2_BUILD_ID
    out_path = out_dir / f"dscnn_stacks_{bid}.h5"
    if out_path.exists():
        print(f"[cache] {out_path.name} already exists, skip")
        return out_path

    hdf5, grid, valid = _setup(out_dir)
    indices = valid["voxel_indices"]
    N = len(indices)
    if N == 0:
        return out_path

    T_max = config.LSTM_T_MAX
    n_ch = len(config.DSCNN_FEATURE_MAP)
    sigma_px = config.GAUSSIAN_STD_PIXELS
    dscnn_class_ids = [v[0] for v in config.DSCNN_FEATURE_MAP.values()]
    dscnn_channel_names = [v[1] for v in config.DSCNN_FEATURE_MAP.values()]
    iz_map = _iz_to_svs(indices)

    with h5py.File(out_path, "w") as out, h5py.File(hdf5, "r") as f:
        part_ds = f["slices/part_ids"]
        seg_dsets = []
        for cls_id in dscnn_class_ids:
            key = f"slices/segmentation_results/{cls_id}"
            seg_dsets.append(f[key] if key in f else None)
        missing = [
            cls for cls, ds in zip(dscnn_class_ids, seg_dsets) if ds is None
        ]
        if missing:
            print(f"  WARN: DSCNN 채널 부재 (cls_ids={missing}) — 0 으로 채움")

        dscnn = out.create_dataset(
            "dscnn", shape=(N, T_max, n_ch), dtype="float32",
            compression="gzip", compression_opts=4, chunks=(1, T_max, n_ch),
        )
        lengths = out.create_dataset("lengths", shape=(N,), dtype="int16")
        out.create_dataset("sv_indices", data=indices.astype(np.int32))
        out.create_dataset("sample_ids", data=valid["sample_ids"].astype(np.int32))
        out.create_dataset("part_ids", data=valid["part_ids"].astype(np.int32))

        for iz in tqdm(sorted(iz_map.keys()), desc=f"{bid} z-blocks (dscnn)"):
            l0, l1 = grid.get_layer_range(iz)
            block_part = part_ds[l0:l1]
            Tb = block_part.shape[0]

            sv_list = iz_map[iz]
            sv_meta: list[dict] = []
            for sv_i in sv_list:
                ix, iy, _ = indices[sv_i]
                r0, r1, c0, c1 = grid.get_pixel_range(int(ix), int(iy))
                part_crop = block_part[:, r0:r1, c0:c1]
                valid_mask = (part_crop > 0).reshape(Tb, -1).any(axis=1)
                if not valid_mask.any():
                    valid_mask[Tb // 2] = True
                T_sv = int(valid_mask.sum())
                T_sv_clip = min(T_sv, T_max)
                active_offsets = np.flatnonzero(valid_mask)
                if len(active_offsets) > T_sv_clip:
                    active_offsets = active_offsets[:T_sv_clip]
                offset_to_ti = {int(off): ti for ti, off in enumerate(active_offsets)}
                seq = np.zeros((T_sv_clip, n_ch), dtype=np.float32)
                sv_meta.append({
                    "sv_i": sv_i, "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                    "valid_mask": valid_mask, "T_sv_clip": T_sv_clip,
                    "offset_to_ti": offset_to_ti, "seq": seq,
                })

            for off in range(Tb):
                active = [m for m in sv_meta if m["valid_mask"][off]]
                if not active:
                    continue
                layer = l0 + off
                part_layer = block_part[off]
                seg_smoothed = []
                for ds in seg_dsets:
                    if ds is None:
                        seg_smoothed.append(None); continue
                    seg = ds[layer].astype(np.float32)
                    seg = gaussian_filter(seg, sigma=sigma_px)
                    seg_smoothed.append(seg)
                for m in active:
                    ti = m["offset_to_ti"].get(int(off))
                    if ti is None:
                        continue
                    r0, r1, c0, c1 = m["r0"], m["r1"], m["c0"], m["c1"]
                    patch_cad = part_layer[r0:r1, c0:c1] > 0
                    if int(patch_cad.sum()) == 0:
                        continue
                    for ci, seg in enumerate(seg_smoothed):
                        if seg is None:
                            continue
                        m["seq"][ti, ci] = seg[r0:r1, c0:c1][patch_cad].mean()

            for m in sv_meta:
                T_sv_clip = m["T_sv_clip"]
                if T_sv_clip > 0:
                    dscnn[m["sv_i"], :T_sv_clip] = m["seq"]
                lengths[m["sv_i"]] = T_sv_clip

        out.attrs["T_max"] = T_max
        out.attrs["n_channels"] = n_ch
        out.attrs["channel_names"] = np.array(dscnn_channel_names, dtype="S")
        out.attrs["dscnn_class_ids"] = np.array(dscnn_class_ids, dtype=np.int32)
        out.attrs["build_id"] = bid
        out.attrs["n_sv"] = N
        out.attrs["sigma_px"] = float(sigma_px)
        out.attrs["valid_layer_rule"] = "part_ids>0 in SV xy region"

    print(f"  saved {out_path}  ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path


# ───────────────────────────────────────────────────────────────────────
# 4) CAD patch — cad_patch_stacks
# ───────────────────────────────────────────────────────────────────────


def build_cad_patch(out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or config.NEW_V2_EVAL_CACHE_DIR)
    bid = config.NEW_V2_BUILD_ID
    out_path = out_dir / f"cad_patch_stacks_{bid}.h5"
    if out_path.exists():
        print(f"[cache] {out_path.name} already exists, skip")
        return out_path

    hdf5, grid, valid = _setup(out_dir)
    indices = valid["voxel_indices"]
    N = len(indices)
    if N == 0:
        return out_path

    T_max = config.LSTM_T_MAX
    H_patch = config.LSTM_FULL86_CAD_PATCH_H
    W_patch = config.LSTM_FULL86_CAD_PATCH_W
    n_ch = config.LSTM_FULL86_N_CAD_CH
    sigma_px = config.GAUSSIAN_STD_PIXELS
    EDGE_SAT = float(config.DIST_EDGE_SATURATION_MM)
    OH_SAT = float(config.DIST_OVERHANG_SATURATION_LAYERS)
    iz_map = _iz_to_svs(indices)

    with h5py.File(out_path, "w") as out, h5py.File(hdf5, "r") as f:
        part_ds = f["slices/part_ids"]
        H_img = grid.image_h
        W_img = grid.image_w

        cad_patch = out.create_dataset(
            "cad_patch", shape=(N, T_max, n_ch, H_patch, W_patch), dtype="float16",
            compression="gzip", compression_opts=4,
            chunks=(1, T_max, n_ch, H_patch, W_patch),
        )
        lengths = out.create_dataset("lengths", shape=(N,), dtype="int16")
        out.create_dataset("sv_indices", data=indices.astype(np.int32))
        out.create_dataset("sample_ids", data=valid["sample_ids"].astype(np.int32))
        out.create_dataset("part_ids", data=valid["part_ids"].astype(np.int32))

        last_overhang = np.full((H_img, W_img), -np.inf, dtype=np.float32)
        prev_cad: np.ndarray | None = None

        sorted_izs = sorted(iz_map.keys())
        for iz in tqdm(sorted_izs, desc=f"{bid} z-blocks (cad_patch)"):
            l0, l1 = grid.get_layer_range(iz)
            block_part = part_ds[l0:l1]
            Tb = block_part.shape[0]

            sv_list = iz_map[iz]
            sv_meta: list[dict] = []
            for sv_i in sv_list:
                ix, iy, _ = indices[sv_i]
                r0, r1, c0, c1 = grid.get_pixel_range(int(ix), int(iy))
                part_crop = block_part[:, r0:r1, c0:c1]
                valid_mask = (part_crop > 0).reshape(Tb, -1).any(axis=1)
                if not valid_mask.any():
                    valid_mask[Tb // 2] = True
                T_sv = int(valid_mask.sum())
                T_sv_clip = min(T_sv, T_max)
                active_offsets = np.flatnonzero(valid_mask)
                if len(active_offsets) > T_sv_clip:
                    active_offsets = active_offsets[:T_sv_clip]
                offset_to_ti = {int(off): ti for ti, off in enumerate(active_offsets)}
                seq = np.zeros((T_sv_clip, n_ch, H_patch, W_patch), dtype=np.float16)
                sv_meta.append({
                    "sv_i": sv_i, "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                    "valid_mask": valid_mask, "T_sv_clip": T_sv_clip,
                    "offset_to_ti": offset_to_ti, "seq": seq,
                })

            for off in range(Tb):
                layer = l0 + off
                cad_mask_layer = (block_part[off] > 0)
                if cad_mask_layer.any():
                    dist_edge = distance_transform_edt(cad_mask_layer) * config.PIXEL_SIZE_MM
                    dist_edge = np.minimum(dist_edge, EDGE_SAT).astype(np.float32)
                    dist_edge = gaussian_filter(dist_edge, sigma=sigma_px)
                else:
                    dist_edge = np.zeros((H_img, W_img), dtype=np.float32)

                if prev_cad is not None:
                    overhang = cad_mask_layer & (~prev_cad)
                    if overhang.any():
                        last_overhang[overhang] = float(layer)
                dist_oh = (float(layer) - last_overhang)
                dist_oh = np.minimum(dist_oh, OH_SAT).astype(np.float32)
                dist_oh = gaussian_filter(dist_oh, sigma=sigma_px)
                prev_cad = cad_mask_layer.copy()

                cad_mask_f32 = cad_mask_layer.astype(np.float32)
                edge_prox = (EDGE_SAT - dist_edge) * cad_mask_f32
                oh_prox   = (OH_SAT   - dist_oh)   * cad_mask_f32

                active = [m for m in sv_meta if m["valid_mask"][off]]
                for m in active:
                    ti = m["offset_to_ti"].get(int(off))
                    if ti is None:
                        continue
                    r0, r1, c0, c1 = m["r0"], m["r1"], m["c0"], m["c1"]
                    edge_patch = edge_prox[r0:r1, c0:c1]
                    oh_patch   = oh_prox  [r0:r1, c0:c1]
                    h_actual, w_actual = edge_patch.shape
                    h_use = min(h_actual, H_patch)
                    w_use = min(w_actual, W_patch)
                    m["seq"][ti, 0, :h_use, :w_use] = edge_patch[:h_use, :w_use]
                    m["seq"][ti, 1, :h_use, :w_use] = oh_patch  [:h_use, :w_use]

            for m in sv_meta:
                T_sv_clip = m["T_sv_clip"]
                if T_sv_clip > 0:
                    cad_patch[m["sv_i"], :T_sv_clip] = m["seq"]
                lengths[m["sv_i"]] = T_sv_clip

        out.attrs["T_max"] = T_max
        out.attrs["n_channels"] = n_ch
        out.attrs["patch_h"] = H_patch
        out.attrs["patch_w"] = W_patch
        out.attrs["channel_names"] = np.array(["edge_proximity", "overhang_proximity"], dtype="S")
        out.attrs["channel_units"] = np.array(["mm", "layers"], dtype="S")
        out.attrs["edge_saturation_mm"] = EDGE_SAT
        out.attrs["overhang_saturation_layers"] = OH_SAT
        out.attrs["inversion_applied"] = True
        out.attrs["mask_applied"] = True
        out.attrs["sigma_px"] = float(sigma_px)
        out.attrs["build_id"] = bid
        out.attrs["n_sv"] = N
        out.attrs["valid_layer_rule"] = "part_ids>0 in SV xy region"

    print(f"  saved {out_path}  ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path


# ───────────────────────────────────────────────────────────────────────
# 5) Scan patch — scan_patch_stacks
# ───────────────────────────────────────────────────────────────────────


def build_scan_patch(out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or config.NEW_V2_EVAL_CACHE_DIR)
    bid = config.NEW_V2_BUILD_ID
    out_path = out_dir / f"scan_patch_stacks_{bid}.h5"
    if out_path.exists():
        print(f"[cache] {out_path.name} already exists, skip")
        return out_path

    hdf5, grid, valid = _setup(out_dir)
    indices = valid["voxel_indices"]
    N = len(indices)
    if N == 0:
        return out_path

    T_max = config.LSTM_T_MAX
    H_patch = config.LSTM_FULL86_SCAN_PATCH_H
    W_patch = config.LSTM_FULL86_SCAN_PATCH_W
    n_ch = config.LSTM_FULL86_N_SCAN_CH
    sat_s = 0.75
    pixel_size_mm = config.PIXEL_SIZE_MM
    kernel_px = max(1, int(round(1.0 / pixel_size_mm)))
    iz_map = _iz_to_svs(indices)

    with h5py.File(out_path, "w") as out, h5py.File(hdf5, "r") as f:
        part_ds = f["slices/part_ids"]
        H_img = grid.image_h
        W_img = grid.image_w
        img_shape = (H_img, W_img)

        scan_patch = out.create_dataset(
            "scan_patch", shape=(N, T_max, n_ch, H_patch, W_patch), dtype="float16",
            compression="gzip", compression_opts=4,
            chunks=(1, T_max, n_ch, H_patch, W_patch),
        )
        lengths = out.create_dataset("lengths", shape=(N,), dtype="int16")
        out.create_dataset("sv_indices", data=indices.astype(np.int32))
        out.create_dataset("sample_ids", data=valid["sample_ids"].astype(np.int32))
        out.create_dataset("part_ids", data=valid["part_ids"].astype(np.int32))

        sorted_izs = sorted(iz_map.keys())
        for iz in tqdm(sorted_izs, desc=f"{bid} z-blocks (scan_patch)"):
            l0, l1 = grid.get_layer_range(iz)
            block_part = part_ds[l0:l1]
            Tb = block_part.shape[0]

            sv_list = iz_map[iz]
            sv_meta: list[dict] = []
            for sv_i in sv_list:
                ix, iy, _ = indices[sv_i]
                r0, r1, c0, c1 = grid.get_pixel_range(int(ix), int(iy))
                part_crop = block_part[:, r0:r1, c0:c1]
                valid_mask = (part_crop > 0).reshape(Tb, -1).any(axis=1)
                if not valid_mask.any():
                    valid_mask[Tb // 2] = True
                T_sv = int(valid_mask.sum())
                T_sv_clip = min(T_sv, T_max)
                active_offsets = np.flatnonzero(valid_mask)
                if len(active_offsets) > T_sv_clip:
                    active_offsets = active_offsets[:T_sv_clip]
                offset_to_ti = {int(off): ti for ti, off in enumerate(active_offsets)}
                seq = np.zeros((T_sv_clip, n_ch, H_patch, W_patch), dtype=np.float16)
                sv_meta.append({
                    "sv_i": sv_i, "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                    "valid_mask": valid_mask, "T_sv_clip": T_sv_clip,
                    "offset_to_ti": offset_to_ti, "seq": seq,
                })

            for off in range(Tb):
                layer = l0 + off
                active = [m for m in sv_meta if m["valid_mask"][off]]
                if not active:
                    continue
                scan_key = f"scans/{layer}"
                if scan_key not in f:
                    continue
                scans = f[scan_key][...]
                if scans.size == 0:
                    continue
                mt_map = build_melt_time_map(scans, img_shape, pixel_size_mm)
                rd_map = compute_return_delay_map(mt_map, kernel_px=kernel_px, sat_s=sat_s)
                sb_map = compute_stripe_boundaries_map(mt_map)
                rd_map = np.nan_to_num(rd_map, nan=0.0, copy=False).astype(np.float32)

                for m in active:
                    ti = m["offset_to_ti"].get(int(off))
                    if ti is None:
                        continue
                    r0, r1, c0, c1 = m["r0"], m["r1"], m["c0"], m["c1"]
                    rd_patch = rd_map[r0:r1, c0:c1]
                    sb_patch = sb_map[r0:r1, c0:c1]
                    h_actual, w_actual = rd_patch.shape
                    h_use = min(h_actual, H_patch)
                    w_use = min(w_actual, W_patch)
                    m["seq"][ti, 0, :h_use, :w_use] = rd_patch[:h_use, :w_use]
                    m["seq"][ti, 1, :h_use, :w_use] = sb_patch[:h_use, :w_use]

            for m in sv_meta:
                T_sv_clip = m["T_sv_clip"]
                if T_sv_clip > 0:
                    scan_patch[m["sv_i"], :T_sv_clip] = m["seq"]
                lengths[m["sv_i"]] = T_sv_clip

        out.attrs["T_max"] = T_max
        out.attrs["n_channels"] = n_ch
        out.attrs["patch_h"] = H_patch
        out.attrs["patch_w"] = W_patch
        out.attrs["channel_names"] = np.array(["return_delay", "stripe_boundaries"], dtype="S")
        out.attrs["channel_units"] = np.array(["s", "a.u."], dtype="S")
        out.attrs["return_delay_saturation_s"] = sat_s
        out.attrs["kernel_px"] = kernel_px
        out.attrs["inversion_applied"] = False
        out.attrs["mask_applied"] = False
        out.attrs["build_id"] = bid
        out.attrs["n_sv"] = N
        out.attrs["valid_layer_rule"] = "part_ids>0 in SV xy region"

    print(f"  saved {out_path}  ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path


# ───────────────────────────────────────────────────────────────────────
# Top-level: 6 캐시 일괄 빌드 + verify
# ───────────────────────────────────────────────────────────────────────


def build_all(out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir or config.NEW_V2_EVAL_CACHE_DIR)
    print("\n[cache_new_v2] building 6 sequence caches for new_v2")
    paths = {
        "v0":         build_v0(out_dir),
        "v1":         build_v1(out_dir),
        "sensor":     build_sensor(out_dir),
        "dscnn":      build_dscnn(out_dir),
        "cad_patch":  build_cad_patch(out_dir),
        "scan_patch": build_scan_patch(out_dir),
    }
    verify_consistency(out_dir)
    return paths


def verify_consistency(out_dir: Path | None = None) -> None:
    """6 캐시의 lengths / sv_indices / sample_ids 비트 일치 검증."""
    out_dir = Path(out_dir or config.NEW_V2_EVAL_CACHE_DIR)
    bid = config.NEW_V2_BUILD_ID
    cache_files = {
        "v0":         (out_dir / f"crop_stacks_{bid}.h5",         "stacks"),
        "v1":         (out_dir / f"crop_stacks_v1_{bid}.h5",      "stacks"),
        "sensor":     (out_dir / f"sensor_stacks_{bid}.h5",       "sensors"),
        "dscnn":      (out_dir / f"dscnn_stacks_{bid}.h5",        "dscnn"),
        "cad_patch":  (out_dir / f"cad_patch_stacks_{bid}.h5",    "cad_patch"),
        "scan_patch": (out_dir / f"scan_patch_stacks_{bid}.h5",   "scan_patch"),
    }
    for tag, (p, _) in cache_files.items():
        if not p.exists():
            raise FileNotFoundError(f"{tag} 캐시 누락: {p}")

    v0_path = cache_files["v0"][0]
    with h5py.File(v0_path, "r") as f0:
        ref_lengths = f0["lengths"][...]
        ref_sv = f0["sv_indices"][...]
        ref_sid = f0["sample_ids"][...]

    for tag, (p, _) in cache_files.items():
        with h5py.File(p, "r") as f:
            ln = f["lengths"][...]
            sv = f["sv_indices"][...]
            sid = f["sample_ids"][...]
        if not np.array_equal(ref_lengths, ln):
            raise RuntimeError(f"{tag}: lengths mismatch vs v0")
        if not np.array_equal(ref_sv, sv):
            raise RuntimeError(f"{tag}: sv_indices mismatch vs v0")
        if not np.array_equal(ref_sid, sid):
            raise RuntimeError(f"{tag}: sample_ids mismatch vs v0")
        print(f"  [verify] {tag}: OK (N={len(ln)}, T_sv median={int(np.median(ln))})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=config.NEW_V2_EVAL_CACHE_DIR)
    p.add_argument("--only", choices=["v0", "v1", "sensor", "dscnn", "cad", "scan", "all"],
                   default="all")
    p.add_argument("--verify", action="store_true",
                   help="6 캐시 일치 검증만 (빌드 안 함)")
    args = p.parse_args()
    if args.verify:
        verify_consistency(args.out_dir)
    elif args.only == "all":
        build_all(args.out_dir)
    elif args.only == "v0":
        build_v0(args.out_dir)
    elif args.only == "v1":
        build_v1(args.out_dir)
    elif args.only == "sensor":
        build_sensor(args.out_dir)
    elif args.only == "dscnn":
        build_dscnn(args.out_dir)
    elif args.only == "cad":
        build_cad_patch(args.out_dir)
    elif args.only == "scan":
        build_scan_patch(args.out_dir)
