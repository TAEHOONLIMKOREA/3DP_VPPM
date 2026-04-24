# E30–E33: G4 (Scan) 피처 재구현 및 재실험 계획

> **목적**: 현재 placeholder(=0) 로 비어있는 `laser_return_delay` 와 `laser_stripe_boundaries`
> 를 **논문 스펙대로 실제 구현**한 뒤, ablation 을 재실행하여 스캔 경로 피처의 "진짜 가치" 를 측정한다.
>
> **배경**: E4 (no-scan) 의 결과는 ΔUTS = −1.04 (오히려 개선) — 즉 현재 G4 는 기여가 없거나
> 오히려 노이즈로 작용하고 있다. 이 결과가 **placeholder 때문인지, 아니면 스캔경로 자체가
> 불필요인지** 구분이 필요.

---

## 1. 현 상태 진단

| 피처 | 파일 | 현재 구현 | 기대 구현 |
|:-----|:----|:---------|:---------|
| #19 `laser_module` | [features.py:105-112](../origin/features.py#L105-L112) | 0/1 바이너리 (레이저 모듈 ID) | 동일 — 변경 불필요 |
| #20 `laser_return_delay` | [features.py:115](../origin/features.py#L115) | **`0.0` (placeholder)** | melt-time map 기반 max–min (논문 Section 2.10) |
| #21 `laser_stripe_boundaries` | [features.py:116](../origin/features.py#L116) | **`0.0` (placeholder)** | melt-time map Sobel RMS (논문 Section 2.10) |

데이터 원천: `scans/{layer_id}` HDF5 데이터셋 (5열: `x_start, x_end, y_start, y_end, time`).

---

## 2. 알고리즘 사양

[implementation_spec.md:186-200](../../implementation_spec.md#L186-L200) 기반. 공통 전처리:

### 2.1 레이어별 melt-time 맵 생성 (공통)

각 레이어의 `scans/{l}` 데이터를 이미지 공간(1842×1842, pixel_size=0.133mm) 로 래스터화.

```python
def build_melt_time_map(scans_layer: np.ndarray, img_shape=(1842, 1842)) -> np.ndarray:
    """
    Args:
        scans_layer: (M, 5) — (x_start, x_end, y_start, y_end, time)
    Returns:
        melt_time_map: (H, W) float — 각 픽셀이 용융된 상대 시간(s).
                       미용융 픽셀은 NaN.
    """
    # 1) 각 스캔 세그먼트를 직선으로 래스터화
    # 2) 픽셀별로 그 스캔이 지나간 시각 기록 (최초 시각 채택)
    # 3) 미용융 픽셀은 NaN 유지
```

구현 메모:
- `skimage.draw.line` 또는 Bresenham 으로 line rasterization.
- 실수 좌표 → 픽셀 좌표: `px = int(x_mm / PIXEL_SIZE_MM)`.
- 픽셀이 여러 번 스캔되면 **최초 시각** 채택 (레이어 시작 기준 누적 시간).
- 레이어마다 melt_time_map 캐시를 디스크(`Sources/pipeline_outputs/scan_melttime/<build>/L{layer}.npy`) 에 저장해 재사용.

### 2.2 피처 #20 — Laser return delay

```python
def feature20_return_delay(mt_map: np.ndarray, sv_bbox, kernel_mm=1.0, sat_s=0.5):
    """슈퍼복셀 영역 내 평균 return-delay."""
    # 1) kernel_px = int(kernel_mm / PIXEL_SIZE_MM)  # ≈ 7~8 px
    # 2) max_map = maximum_filter(mt_map, size=kernel_px)  # NaN 처리 주의
    #    min_map = minimum_filter(mt_map, size=kernel_px)
    # 3) delay = max_map - min_map
    # 4) stripe 경계(=delay > sat_s)는 saturation 값으로 clip
    # 5) 슈퍼복셀 bbox 평균 반환
```

물리적 의미: 주변 kernel 안에서 같은 영역이 **다시 레이저에 노출되기까지의 시간차** → 냉각 시간 proxy.

### 2.3 피처 #21 — Laser stripe boundaries

```python
def feature21_stripe_boundaries(mt_map: np.ndarray, sv_bbox):
    """슈퍼복셀 영역 내 Sobel RMS 평균."""
    # 1) mt_filled = np.nan_to_num(mt_map, nan=0.0) 또는 inpaint
    # 2) sx = sobel(mt_filled, axis=0)
    #    sy = sobel(mt_filled, axis=1)
    # 3) rms = np.sqrt(sx**2 + sy**2)
    # 4) 슈퍼복셀 bbox 평균 반환
```

물리적 의미: melt-time 의 공간적 불연속성 = 스트라이프 경계 밀도.

### 2.4 70레이어 z-평균

슈퍼복셀당 `SV_Z_LAYERS=70` 레이어의 값을 단순 평균 (다른 피처와 동일 정책).

### 2.5 결측 레이어 처리

`scans/{l}` 가 없는 레이어는 skip, 해당 슈퍼복셀의 평균은 남은 레이어만으로 계산. 레이어가 하나도 없으면 NaN — `build_dataset` 의 NaN 마스크에서 자동 제거됨 (기존 로직).

---

## 3. 실험 설계

### 3.1 피처 재추출 선행

구현 후 반드시 **feature 재추출부터** 수행. Baseline(E0) 자체가 placeholder(0) 로 학습되었으므로, 공정 비교를 위해 baseline 도 재학습 필요.

```bash
# 1. 기존 all_features.npz 백업
mv Sources/pipeline_outputs/features/all_features.npz \
   Sources/pipeline_outputs/features/all_features.v1_placeholder.npz

# 2. 재추출 (features.py 개선 후)
./venv/bin/python -m Sources.vppm.run_pipeline --phase features

# 3. 새 baseline 재학습
./venv/bin/python -m Sources.vppm.run_pipeline --phase train,evaluate
# → pipeline_outputs/results/vppm_origin/ 를 v2_scan_impl 로 이동/복사 권장
```

### 3.2 실험 목록

| ID | 실험명 | 설명 | 피처 수 |
|:--:|-------|-----|:------:|
| E30  | **Baseline v2**      | 21 피처 재구현 후 재학습. 비교 기준 | 21 |
| E31  | **No-Scan v2**       | G4 3개 재구현 후 제거 — E4 재실행 | 18 |
| E32  | **No-ReturnDelay**   | #20 단독 제거 | 20 |
| E33  | **No-StripeBoundary**| #21 단독 제거 | 20 |

### 3.3 비교 테이블 템플릿

| 버전 | 실험 | ΔUTS vs 해당 baseline | 해석 |
|:----:|:----:|:---------------------:|:-----|
| v1 (placeholder) | E0 → E4 | −1.04 (current) | placeholder 포함이 오히려 방해 |
| v2 (실구현)      | E30 → E31 | ? | 참된 스캔 피처 가치 |
| v2 (실구현)      | E30 → E32 | ? | return_delay 단독 기여 |
| v2 (실구현)      | E30 → E33 | ? | stripe_boundaries 단독 기여 |

### 3.4 성공 기준

| 결과 패턴 | 결론 |
|:----|:-----|
| E30 이 E0 보다 낮은 RMSE (예: UTS 60.7 → 58.x) | 스캔 피처 재구현이 기여. v2 를 새 baseline 으로 채택 권장 |
| E30 ≈ E0, E31 의 ΔUTS 도 작음 (|Δ| < 0.5) | 스캔 피처는 물리적 구현을 해도 본질적으로 불필요 — **G4 전체 폐기** 고려 |
| E30 이 E0 보다 **높은** RMSE (v2 가 악화) | 재구현 코드 버그 의심 — 배포 전 단위 테스트 필수 |
| E32 은 유의미, E33 은 무시가능 | stripe_boundaries 는 redundant, return_delay 가 핵심 (또는 역) |

---

## 4. 구현 순서

### 4.1 단계 1 — 스캔 데이터 래스터화 유틸 (신규)

새 파일 `Sources/vppm/origin/scan_features.py`:

```python
def build_melt_time_map(scans_layer, img_shape):
    ...

def compute_return_delay_map(mt_map, kernel_mm=1.0, sat_s=0.5):
    ...

def compute_stripe_boundaries_map(mt_map):
    ...
```

단위 테스트 (`Sources/tests/test_scan_features.py`):
- 수평 스캔만 있는 합성 케이스 → stripe_boundaries ≈ 0
- 두 스트라이프 사이 gap = 2s → return_delay == 2.0 (saturation 미적용 시)
- 빈 레이어 → NaN 반환

### 4.2 단계 2 — features.py 패치

[features.py:114-116](../origin/features.py#L114-L116) 교체:

```python
# 기존
features[block_indices, 19] = 0.0  # placeholder
features[block_indices, 20] = 0.0  # placeholder

# 신규
from .scan_features import (
    build_melt_time_map,
    compute_return_delay_map,
    compute_stripe_boundaries_map,
)

# z-block 루프 안에서 레이어별 처리
rd_accum = np.zeros(n, dtype=np.float64)
sb_accum = np.zeros(n, dtype=np.float64)
count_accum = np.zeros(n, dtype=np.float64)
for layer in range(l0, l1):
    key = f"scans/{layer}"
    if key not in f:
        continue
    scans = f[key][...]
    mt = build_melt_time_map(scans, (IMAGE_PIXELS, IMAGE_PIXELS))
    rd_map = compute_return_delay_map(mt)
    sb_map = compute_stripe_boundaries_map(mt)
    for i, pidx in enumerate(block_indices):
        bbox = grid.voxel_bbox_px(block_voxels[i])
        rd_accum[i] += np.nanmean(rd_map[bbox])
        sb_accum[i] += np.nanmean(sb_map[bbox])
        count_accum[i] += 1

features[block_indices, 19] = np.where(count_accum > 0, rd_accum / count_accum, np.nan)
features[block_indices, 20] = np.where(count_accum > 0, sb_accum / count_accum, np.nan)
```

주의: 캐시 활용 — `melt_time_map` 은 빌드당 수천 레이어이므로 디스크 캐시 필수. 한 번 생성해두면 E32/E33 도 재사용.

### 4.3 단계 3 — ablation runner 업데이트

`Sources/vppm/common/config.py` 에 서브 그룹 추가:

```python
FEATURE_GROUPS_SCAN_SUB = {
    "scan_return_delay": [19],
    "scan_stripe_boundaries": [20],
}
FEATURE_GROUPS.update(FEATURE_GROUPS_SCAN_SUB)
```

`Sources/vppm/ablation/run.py` 의 `EXPERIMENTS` 에 추가:

```python
EXPERIMENTS.update({
    "E31": ("scan",                    "v2 No-Scan — 실구현 G4 3피처 제거"),
    "E32": ("scan_return_delay",       "No-ReturnDelay — #20 단독 제거"),
    "E33": ("scan_stripe_boundaries",  "No-StripeBoundary — #21 단독 제거"),
})
```

Baseline v2 (E30) 는 기존 `run_pipeline.py` 로 학습하고, 결과를
`pipeline_outputs/results/vppm_origin_v2/` 에 별도 저장해 v1 baseline 과 병행 보존.

### 4.4 단계 4 — 실행

```bash
# 0. 피처 추출 (가장 오래 걸림 — scan 래스터화 포함)
./venv/bin/python -m Sources.vppm.run_pipeline --phase features

# 1. Baseline v2
./venv/bin/python -m Sources.vppm.run_pipeline --phase train,evaluate

# 2. Ablation 3종 (순차)
for E in E31 E32 E33; do
  ./venv/bin/python -m Sources.vppm.ablation.run --experiment $E
done

# 3. 요약 갱신
./venv/bin/python -m Sources.vppm.ablation.run --rebuild-summary
```

---

## 5. 리소스 및 일정

| 단계 | 예상 시간 | 비고 |
|:----:|:--------:|:----|
| 스캔 래스터화 유틸 구현 + 테스트 | 3–4시간 | 버그 여지 큰 부분 |
| 피처 재추출 (5 빌드 × 수천 레이어) | **~6–10시간** | 캐시 디스크 필요 (~수 GB) |
| Baseline v2 학습 | ~30분 | GPU |
| E31 + E32 + E33 학습 | ~1.5시간 | GPU |
| 리포트 작성 | ~1시간 | |
| **총** | **~12–16시간 (절반이 피처 재추출)** | |

> **디스크 요구**: melt-time 맵 캐시 = 1842×1842×4byte × (~2500 layer × 5 build) ≈ **170 GB**.
> 캐시 없이 매번 재계산하면 E31/E32/E33 학습 시마다 수 시간 소요 → **캐시 사이즈 관리 중요**.
> 대안: `float16` 으로 다운캐스트 (≈85 GB) 또는 레이어별로 지연 계산 + LRU 캐시.

---

## 6. 성공 기준

- [x] `scan_features.py` 3개 함수 + 단위 테스트 통과
- [x] `features.py` 에서 placeholder 제거, 새 피처 실제 값으로 채워짐
- [x] `all_features.v2.npz` 재생성, NaN 비율 vs v1 비교 (대폭 증가해선 안 됨)
- [x] E30 Baseline v2 RMSE 가 v1 대비 합리적 범위 (±3 MPa 이내)
- [x] E31·E32·E33 학습 완료 및 `sensor_sub_summary.md` 와 별도로 `scan_sub_summary.md` 작성
- [x] v1 vs v2 비교표에서 시나리오 판정 (기여 있음 / 재구현도 무용 / 코드 버그)

---

## 7. 리스크

- **캐시 크기**: 170 GB 넘을 수 있음. 프로젝트 루트 `ORNL_Data_Origin/` 이 이미 230 GB 이므로 여유 디스크 확인 필요. 부족 시 빌드 단위 순차 처리 + 처리 끝난 빌드의 캐시 삭제.
- **Line rasterization 오차**: Bresenham 구현과 논문의 정확한 방법이 미세하게 다를 수 있음. "정량적 일치" 가 아닌 "순서 보존 + 물리적 타당성" 을 기준으로 검증.
- **NaN 처리**: `maximum_filter` 와 `sobel` 은 NaN 에 대해 설계되지 않음. 각 함수 내부에서 명시적 NaN-aware 구현 또는 0 채움 후 mask 적용.
- **Placeholder 제거가 baseline 을 *악화*시킬 가능성**: 원래 0 이었던 피처가 모델의 암묵적 bias 였다면, 실제 값으로 바뀌면 fold std 가 일시적으로 증가할 수 있음. 이 경우 E30 vs E0 비교에서 std 차이도 함께 주목.
- **G4 폐기 결정의 함의**: E31 도 Δ≈0 이면 config.py 의 `FEATURE_GROUPS["scan"]` 을 제거하고 `N_FEATURES=18` 로 상수 업데이트 고려 — 단, baseline/재현성을 해치므로 논문 비교용 기준은 21 피처 유지하고, 경량 모델만 18 피처 사용하는 dual-track 추천.
