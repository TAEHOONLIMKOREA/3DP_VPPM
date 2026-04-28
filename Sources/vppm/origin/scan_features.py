"""스캔 경로 기반 피처 (G4) 계산.

논문 Section 2.10 / implementation_spec.md §피처 20-21.

3 단계 파이프라인:
    1) build_melt_time_map        — HDF5 scans/{layer} → 1842×1842 melt-time 맵
    2) compute_return_delay_map   — 1mm 커널 max - min  (재용융 시간차)
    3) compute_stripe_boundaries_map — Sobel RMS         (스트라이프 경계 밀도)

성능 메모:
    - melt-time 맵 1842×1842 float32 = 13 MB. 영구 캐시하지 않고 레이어별 즉시 계산 + 폐기.
    - 스캔 세그먼트의 ~89% 가 sub-pixel (양 끝점이 같은 픽셀). 이 fast-path 는 `np.minimum.at`
      한 번으로 처리. 다중-픽셀 세그먼트는 벡터화 보간 (np.repeat + linspace 합산).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, sobel


# ── 1) Rasterization ───────────────────────────────────────────


def build_melt_time_map(
    scans_layer: np.ndarray,
    img_shape: tuple[int, int],
    pixel_size_mm: float,
) -> np.ndarray:
    """레이어 스캔 데이터를 melt-time 맵으로 래스터화.

    Args:
        scans_layer: (M, 5) — (x_start, x_end, y_start, y_end, time). 좌표 mm, 시간 s.
        img_shape: (H, W) 출력 맵 크기 (보통 1842×1842).
        pixel_size_mm: 픽셀당 mm.

    Returns:
        mt: (H, W) float32 — 각 픽셀의 최초 용융 시각 (s). 미용융 픽셀은 NaN.
    """
    H, W = img_shape
    if scans_layer is None or len(scans_layer) == 0:
        return np.full((H, W), np.nan, dtype=np.float32)

    # 픽셀 좌표로 변환 (반올림).
    # 좌표계 주의: scan 데이터는 build-plate 의 Cartesian (Y 위 방향) 이지만,
    # image 는 top-down (row=0 이 위). Y 를 flip 해서 image 좌표로 매핑.
    #   col = x / pixel_size       (X 축 방향 동일)
    #   row = (H-1) - y / pixel_size  (Y 축 반전)
    c0 = np.round(scans_layer[:, 0] / pixel_size_mm).astype(np.int32)
    c1 = np.round(scans_layer[:, 1] / pixel_size_mm).astype(np.int32)
    r0 = (H - 1) - np.round(scans_layer[:, 2] / pixel_size_mm).astype(np.int32)
    r1 = (H - 1) - np.round(scans_layer[:, 3] / pixel_size_mm).astype(np.int32)
    t = scans_layer[:, 4].astype(np.float32)

    # 양 끝점 모두 영역 밖이면 폐기 (한쪽이라도 안에 있으면 부분 라인 그림)
    in0 = (c0 >= 0) & (c0 < W) & (r0 >= 0) & (r0 < H)
    in1 = (c1 >= 0) & (c1 < W) & (r1 >= 0) & (r1 < H)
    keep = in0 | in1
    if not keep.any():
        return np.full((H, W), np.nan, dtype=np.float32)
    c0, c1, r0, r1, t = c0[keep], c1[keep], r0[keep], r1[keep], t[keep]

    # 픽셀 단위 세그먼트 길이
    length = np.maximum(np.abs(c1 - c0), np.abs(r1 - r0))  # 0 = same pixel
    same = length == 0
    multi = ~same

    rs_parts: list[np.ndarray] = []
    cs_parts: list[np.ndarray] = []
    ts_parts: list[np.ndarray] = []

    # ── Fast path: sub-pixel 세그먼트 (벡터화) ────────────────────
    if same.any():
        rs_parts.append(r0[same])
        cs_parts.append(c0[same])
        ts_parts.append(t[same])

    # ── Multi-pixel 세그먼트: 벡터화 보간 ─────────────────────────
    if multi.any():
        L = length[multi] + 1  # 라인당 픽셀 수 (양 끝점 포함)
        total = int(L.sum())

        # 세그먼트별 시작 offset (offsets 의 누적합)
        starts = np.zeros(L.size, dtype=np.int64)
        starts[1:] = np.cumsum(L[:-1])
        # 모든 픽셀 인덱스 = global index - 해당 세그먼트의 시작 인덱스
        global_idx = np.arange(total, dtype=np.int64)
        seg_id = np.repeat(np.arange(L.size, dtype=np.int64), L)
        local_idx = global_idx - starts[seg_id]
        # 분수 위치 0..1 (L=1 인 경우는 multi 에 들어오지 않으므로 L-1 ≥ 1)
        frac = local_idx.astype(np.float32) / np.repeat((L - 1).astype(np.float32), L)

        # 보간된 (r, c)
        r0_m = r0[multi].astype(np.float32)
        r1_m = r1[multi].astype(np.float32)
        c0_m = c0[multi].astype(np.float32)
        c1_m = c1[multi].astype(np.float32)
        t_m = t[multi]

        rs_m = np.round(np.repeat(r0_m, L) + frac * np.repeat(r1_m - r0_m, L)).astype(np.int32)
        cs_m = np.round(np.repeat(c0_m, L) + frac * np.repeat(c1_m - c0_m, L)).astype(np.int32)
        ts_m = np.repeat(t_m, L)

        rs_parts.append(rs_m)
        cs_parts.append(cs_m)
        ts_parts.append(ts_m)

    rs_all = np.concatenate(rs_parts)
    cs_all = np.concatenate(cs_parts)
    ts_all = np.concatenate(ts_parts)

    # 영역 안에 있는 픽셀만 유지 (multi-pixel 보간 결과가 외곽으로 나갈 수 있음)
    ok = (rs_all >= 0) & (rs_all < H) & (cs_all >= 0) & (cs_all < W)
    rs_all, cs_all, ts_all = rs_all[ok], cs_all[ok], ts_all[ok]

    # 같은 픽셀 여러 번 → 가장 빠른 시각 채택 (in-place min reduction)
    mt = np.full((H, W), np.inf, dtype=np.float32)
    np.minimum.at(mt, (rs_all, cs_all), ts_all)
    mt[np.isinf(mt)] = np.nan
    return mt


# ── 2) Return delay map ────────────────────────────────────────


def compute_return_delay_map(
    mt_map: np.ndarray,
    kernel_px: int,
    sat_s: float = 0.5,
) -> np.ndarray:
    """melt-time 맵 → return-delay 맵.

    이웃 영역(kernel_px) 안에서 (max 시각 − min 시각) 을 계산.
    같은 영역이 다시 스캔되기까지의 시간 → 냉각 시간 proxy.

    Args:
        mt_map: (H, W) float — 미용융은 NaN.
        kernel_px: 사각형 커널 크기 (홀수 권장).
        sat_s: 큰 점프 (스트라이프 경계, 분리된 영역 간 시간차) 를 제거하기 위한 saturation.

    Returns:
        delay: (H, W) float32 — 미용융 영역은 NaN.
    """
    H, W = mt_map.shape
    valid = ~np.isnan(mt_map)
    if not valid.any():
        return np.full((H, W), np.nan, dtype=np.float32)

    # NaN 을 max/min 에서 제외하기 위해 sentinel 로 대체
    mt_for_max = np.where(valid, mt_map, -np.inf)
    mt_for_min = np.where(valid, mt_map, +np.inf)

    max_map = maximum_filter(mt_for_max, size=kernel_px, mode="constant", cval=-np.inf)
    min_map = minimum_filter(mt_for_min, size=kernel_px, mode="constant", cval=+np.inf)

    # max 가 -inf 또는 min 이 +inf 면 = 커널 안에 유효 픽셀 0 개 → NaN
    valid_kernel = np.isfinite(max_map) & np.isfinite(min_map)
    delay = np.full((H, W), np.nan, dtype=np.float32)
    delay[valid_kernel] = (max_map[valid_kernel] - min_map[valid_kernel]).astype(np.float32)
    # saturation clip (양의 값만)
    delay = np.minimum(delay, sat_s)
    # 미용융 픽셀은 항상 NaN 유지
    delay[~valid] = np.nan
    return delay


# ── 3) Stripe boundaries map ───────────────────────────────────


def compute_stripe_boundaries_map(mt_map: np.ndarray) -> np.ndarray:
    """melt-time 맵 → 스트라이프 경계 밀도 맵 (Sobel RMS).

    Args:
        mt_map: (H, W) float — 미용융은 NaN.

    Returns:
        rms: (H, W) float32 — 미용융 픽셀은 0 (정의상 경계 신호 없음).
    """
    valid = ~np.isnan(mt_map)
    if not valid.any():
        return np.zeros_like(mt_map, dtype=np.float32)

    # NaN 을 0 으로 채워 Sobel 계산. 미용융 픽셀에서 인공 경계가 생기지만,
    # 슈퍼복셀 평균 시 미용융 픽셀은 patch.mean 에서 제외되므로 영향 미미.
    mt_filled = np.nan_to_num(mt_map, nan=0.0).astype(np.float32)
    sx = sobel(mt_filled, axis=0, mode="constant", cval=0.0)
    sy = sobel(mt_filled, axis=1, mode="constant", cval=0.0)
    rms = np.sqrt(sx * sx + sy * sy).astype(np.float32)
    # 미용융 픽셀은 의미 없으므로 0 (NaN 보다 평균 계산 편의)
    rms[~valid] = 0.0
    return rms
