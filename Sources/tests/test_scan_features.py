"""Sources/vppm/baseline/scan_features.py 단위 테스트.

실행:
    ./venv/bin/python -m pytest Sources/tests/test_scan_features.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from Sources.vppm.baseline.scan_features import (
    build_melt_time_map,
    compute_return_delay_map,
    compute_stripe_boundaries_map,
)

PIXEL_MM = 245.0 / 1842  # ≈ 0.1330 mm/px


# ── build_melt_time_map ────────────────────────────────────────


def test_empty_layer_returns_all_nan():
    mt = build_melt_time_map(
        np.empty((0, 5), dtype=np.float32),
        img_shape=(64, 64),
        pixel_size_mm=PIXEL_MM,
    )
    assert mt.shape == (64, 64)
    assert np.all(np.isnan(mt))


def _y_to_row(y_mm: float, H: int = 200) -> int:
    """build-plate y(mm) → image row. Y 축 flip 적용."""
    return (H - 1) - round(y_mm / PIXEL_MM)


def test_single_subpixel_segment_marks_one_pixel():
    """양 끝점이 같은 픽셀인 세그먼트 → 그 픽셀만 시각 기록."""
    scans = np.array([[10.0, 10.001, 20.0, 20.001, 0.5]], dtype=np.float32)
    mt = build_melt_time_map(scans, img_shape=(200, 200), pixel_size_mm=PIXEL_MM)

    r = _y_to_row(20.0)
    c = round(10.0 / PIXEL_MM)
    assert mt[r, c] == pytest.approx(0.5, abs=1e-5)
    nan_mask = np.isnan(mt)
    assert nan_mask.sum() == 200 * 200 - 1


def test_minimum_time_when_pixel_revisited():
    scans = np.array([
        [10.0, 10.0, 20.0, 20.0, 5.0],
        [10.0, 10.0, 20.0, 20.0, 1.0],
    ], dtype=np.float32)
    mt = build_melt_time_map(scans, img_shape=(200, 200), pixel_size_mm=PIXEL_MM)

    r = _y_to_row(20.0)
    c = round(10.0 / PIXEL_MM)
    assert mt[r, c] == pytest.approx(1.0)


def test_horizontal_line_segment_rasterizes_correctly():
    scans = np.array([[10.0, 11.0, 20.0, 20.0, 0.5]], dtype=np.float32)
    mt = build_melt_time_map(scans, img_shape=(200, 200), pixel_size_mm=PIXEL_MM)

    r = _y_to_row(20.0)
    c0 = round(10.0 / PIXEL_MM)
    c1 = round(11.0 / PIXEL_MM)
    assert not np.isnan(mt[r, c0])
    assert not np.isnan(mt[r, c1])
    assert not np.isnan(mt[r, (c0 + c1) // 2])
    assert mt[r, c0] == pytest.approx(0.5)
    assert mt[r, c1] == pytest.approx(0.5)
    # 인접 행은 NaN
    assert np.isnan(mt[r + 1, c0])


def test_vertical_line_segment():
    """수직 라인 (x=10mm 고정, y=20→21mm). Y flip 으로 row 가 작아짐."""
    scans = np.array([[10.0, 10.0, 20.0, 21.0, 0.7]], dtype=np.float32)
    mt = build_melt_time_map(scans, img_shape=(200, 200), pixel_size_mm=PIXEL_MM)

    c = round(10.0 / PIXEL_MM)
    r_y20 = _y_to_row(20.0)
    r_y21 = _y_to_row(21.0)
    assert mt[r_y20, c] == pytest.approx(0.7)
    assert mt[r_y21, c] == pytest.approx(0.7)
    assert mt[(r_y20 + r_y21) // 2, c] == pytest.approx(0.7)


def test_out_of_bounds_segment_is_dropped():
    scans = np.array([[1000.0, 1001.0, 2000.0, 2001.0, 0.5]], dtype=np.float32)
    mt = build_melt_time_map(scans, img_shape=(64, 64), pixel_size_mm=PIXEL_MM)
    assert np.all(np.isnan(mt))


# ── compute_return_delay_map ───────────────────────────────────


def test_return_delay_zero_for_uniform_time():
    """모든 픽셀이 같은 시각이면 return_delay = 0."""
    mt = np.full((50, 50), 1.0, dtype=np.float32)
    delay = compute_return_delay_map(mt, kernel_px=5, sat_s=0.5)
    assert np.all(delay == 0.0)


def test_return_delay_captures_time_difference():
    """두 인접 영역의 시간 차이가 return_delay 로 잡힌다."""
    mt = np.full((50, 50), np.nan, dtype=np.float32)
    mt[10, 10] = 0.0
    mt[10, 11] = 0.3
    delay = compute_return_delay_map(mt, kernel_px=5, sat_s=0.5)
    # 두 픽셀이 5x5 커널 안에 동시에 들어오는 곳에서 delay = 0.3
    assert np.nanmax(delay) == pytest.approx(0.3, abs=1e-5)


def test_return_delay_saturation_clips_large_jumps():
    """sat_s 보다 큰 시간 점프는 sat_s 로 clip 된다."""
    mt = np.full((50, 50), np.nan, dtype=np.float32)
    mt[10, 10] = 0.0
    mt[10, 11] = 10.0  # 큰 점프
    delay = compute_return_delay_map(mt, kernel_px=5, sat_s=0.5)
    assert np.nanmax(delay) == pytest.approx(0.5, abs=1e-5)


def test_return_delay_nan_outside_melted():
    """melt-time 이 NaN 인 픽셀의 return_delay 도 NaN."""
    mt = np.full((50, 50), np.nan, dtype=np.float32)
    mt[20:30, 20:30] = 1.0
    delay = compute_return_delay_map(mt, kernel_px=5, sat_s=0.5)
    # 멀리 떨어진 픽셀 (커널 영향 없음) 은 NaN
    assert np.isnan(delay[0, 0])
    # 용융 영역 안은 0 (모두 같은 시각)
    assert delay[25, 25] == pytest.approx(0.0)


# ── compute_stripe_boundaries_map ──────────────────────────────


def test_stripe_boundaries_zero_on_uniform():
    mt = np.full((50, 50), 1.0, dtype=np.float32)
    rms = compute_stripe_boundaries_map(mt)
    # Sobel of uniform image = 0 (단, 경계 효과로 외곽엔 작은 값 가능)
    inner = rms[5:-5, 5:-5]
    assert np.all(inner == 0.0)


def test_stripe_boundaries_nonzero_on_step():
    """melt-time 의 step 경계에서 Sobel 응답이 큼."""
    mt = np.zeros((50, 50), dtype=np.float32)
    mt[:, 25:] = 1.0  # 수직 step
    rms = compute_stripe_boundaries_map(mt)
    # step 경계에서 큰 값
    assert rms[25, 24] > 0.5
    assert rms[25, 25] > 0.5
    # 멀리 떨어진 곳은 0
    assert rms[10, 10] == pytest.approx(0.0)


def test_stripe_boundaries_handles_nan():
    mt = np.full((50, 50), np.nan, dtype=np.float32)
    mt[20:30, 20:30] = 1.0
    rms = compute_stripe_boundaries_map(mt)
    # NaN 영역은 0 (rms[~valid]=0 으로 강제)
    assert rms[0, 0] == 0.0
    assert rms[19, 25] == 0.0
    # 용융 영역 내부 경계 픽셀 (NaN→0 fill 의 step 효과로 응답 발생)
    assert rms[20, 25] > 0.0  # 위쪽 경계
    assert rms[29, 25] > 0.0  # 아래쪽 경계
    assert rms[25, 20] > 0.0  # 왼쪽 경계
    assert rms[25, 29] > 0.0  # 오른쪽 경계


# ── Integration ────────────────────────────────────────────────


def test_full_pipeline_smoke():
    """3 함수가 연쇄적으로 잘 동작하는지 smoke."""
    rng = np.random.default_rng(42)
    # 100 segments, 모두 5×5mm 영역 안 무작위
    n = 100
    scans = np.zeros((n, 5), dtype=np.float32)
    scans[:, 0] = rng.uniform(0, 5, n)
    scans[:, 1] = scans[:, 0] + rng.uniform(-0.05, 0.05, n)
    scans[:, 2] = rng.uniform(0, 5, n)
    scans[:, 3] = scans[:, 2] + rng.uniform(-0.05, 0.05, n)
    scans[:, 4] = np.linspace(0, 0.1, n)  # time 단조 증가

    mt = build_melt_time_map(scans, img_shape=(200, 200), pixel_size_mm=PIXEL_MM)
    rd = compute_return_delay_map(mt, kernel_px=8, sat_s=0.5)
    sb = compute_stripe_boundaries_map(mt)

    assert mt.shape == rd.shape == sb.shape == (200, 200)
    # 일부 픽셀이 melt 됨
    assert np.any(~np.isnan(mt))
    # rd 가 non-trivial
    assert np.nanmax(rd) > 0
    # sb 가 non-trivial
    assert np.max(sb) > 0
