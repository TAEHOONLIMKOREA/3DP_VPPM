# 슈퍼복셀 입력 21개 피처 명세

> **출처**: Scime et al., *Materials* 2023, 16, 7293 — Section 2.10 & Appendix D.
> **구현**: `Sources/vppm/origin/features.py` (`FeatureExtractor.extract_features`).
> **단위**: 1 슈퍼복셀 = 1 mm × 1 mm × 3.5 mm (= 70 layer × 0.05 mm).
> **공통 처리**: 각 피처는 레이어별 픽셀 맵에서 슈퍼복셀 영역 (≈ 7.52 × 7.52 px) 안의 평균을 z-방향(70 layer)으로 누적 평균.

---

## 그룹 요약

| Group | 인덱스 (1-based / 0-based) | 출처 | 개수 |
|:----|:----|:----|:--:|
| **G3 — CAD 기하** | #1–3 / 0–2 | `slices/part_ids` (이진 형상 마스크) | 3 |
| **G1 — DSCNN 세그멘테이션** | #4–11 / 3–10 | `slices/segmentation_results/{0–11}` | 8 |
| **G2 — 프린터 센서 (Temporal)** | #12–18 / 11–17 | `temporal/*` (레이어별 1D 시계열) | 7 |
| **G4 — 레이저 스캔 경로** | #19–21 / 18–20 | `parts/process_parameters/laser_module`, `scans/{layer}` | 3 |

---

## G3 — CAD 기하 피처 (#1–3)

레이어별 part 마스크(`part_ids > 0`)에서 도출. 가우시안 블러 σ ≈ 3.76 px (= 0.5 mm).

### #1  `distance_from_edge`  *(mm)*
- **의미**: 픽셀에서 가장 가까운 part 외곽까지의 in-plane 거리.
- **계산**: `scipy.ndimage.distance_transform_edt(cad_mask) × pixel_size_mm` → 가우시안 블러.
- **포화**: `DIST_EDGE_SATURATION_MM = 3.0 mm` 로 clip (외곽 ≥ 3mm 안쪽은 모두 동일 취급).
- **물리적 동기**: 외곽 근처는 열 전달 비대칭 → 미세조직 차이 발생.

### #2  `distance_from_overhang`  *(layers)*
- **의미**: 픽셀에서 가장 가까운 오버행 영역까지의 거리(레이어 단위).
- **계산**: 오버행 = `current_cad ∧ ¬previous_cad` (현재 레이어 part 인데 바로 아래에는 없음). 역마스크의 EDT × 가우시안.
- **포화**: `DIST_OVERHANG_SATURATION_LAYERS = 71` (= 슈퍼복셀 z-크기 70 + 1).
- **물리적 동기**: 오버행 부위는 분말층 위로 직접 용융 → 결함 발생률 ↑.

### #3  `build_height`  *(mm)*
- **의미**: 슈퍼복셀의 z-중심 높이.
- **계산**: `grid.get_z_center_mm(iz)` — z-블록 인덱스 → 빌드 플레이트 기준 mm.
- **물리적 동기**: 누적 열 이력, 분말 베드 상태가 높이에 따라 변화.

---

## G1 — DSCNN 세그멘테이션 피처 (#4–11)

`slices/segmentation_results/{class_id}` (HDF5 12 클래스 → 논문 8 클래스 매핑) 의 픽셀별 0/1 마스크에 가우시안 블러 σ ≈ 3.76 px 적용 후 슈퍼복셀 영역 내 CAD 픽셀의 평균을 누적.
**값 범위**: [0, 1] — 슈퍼복셀 내 해당 결함의 평균 발생 비율.

| # | 0-based | 이름 | HDF5 cls | 의미 |
|:--:|:--:|:----|:--:|:----|
| 4 | 3 | `seg_powder` | 0 | 분말 (정상) — 미용융 분말 영역 |
| 5 | 4 | `seg_printed` | 1 | 프린트 (정상) — 정상 용융 영역 |
| 6 | 5 | `seg_recoater_streaking` | 3 | 리코터 줄무늬 — 분말 도포 결함 |
| 7 | 6 | `seg_edge_swelling` | 5 | 엣지 융기 — 외곽 부풀음 |
| 8 | 7 | `seg_debris` | 6 | 잔해/스패터 |
| 9 | 8 | `seg_super_elevation` | 7 | 과돌출 — 부분적 과한 융기 |
| 10 | 9 | `seg_soot` | 8 | 매연/응축물 |
| 11 | 10 | `seg_excessive_melting` | 10 | 과용융 — Keyhole 모드 후보 |

매핑 정의: `Sources/vppm/common/config.py:DSCNN_FEATURE_MAP`.

---

## G2 — 프린터 센서 피처 (#12–18)

`temporal/*` 데이터셋(빌드당 레이어 수 만큼의 1D 시계열) 을 슈퍼복셀의 z-블록(70 layer) 구간에서 단순 평균.
**공간적으로는 균일** (한 z-블록 내 모든 슈퍼복셀이 같은 값).

| # | 0-based | 이름 | HDF5 키 | 단위 | 설명 |
|:--:|:--:|:----|:----|:--:|:----|
| 12 | 11 | `layer_print_time` | `temporal/layer_times` | s | 레이어 1장 출력 시간 — 처리량 / 안정성 지표 |
| 13 | 12 | `top_gas_flow_rate` | `temporal/top_flow_rate` | L/min | 상부 보호가스 유량 |
| 14 | 13 | `bottom_gas_flow_rate` | `temporal/bottom_flow_rate` | L/min | 하부 보호가스 유량 |
| 15 | 14 | `module_oxygen` | `temporal/module_oxygen` | ppm/% | 챔버 산소 농도 — 산화 위험 지표 |
| 16 | 15 | `build_plate_temperature` | `temporal/build_plate_temperature` | °C | 빌드 플레이트 온도 |
| 17 | 16 | `bottom_flow_temperature` | `temporal/bottom_flow_temperature` | °C | 하부 가스 온도 |
| 18 | 17 | `actual_ventilator_flow_rate` | `temporal/actual_ventilator_flow_rate` | L/min | 실측 환기 유량 |

매핑 정의: `Sources/vppm/common/config.py:TEMPORAL_FEATURES`.

---

## G4 — 레이저 스캔 경로 피처 (#19–21)

### #19  `laser_module`  *(이진 0/1)*
- **의미**: 해당 part 가 어느 레이저 모듈로 출력되었는지.
- **계산**: `parts/process_parameters/laser_module` 값 = 1 → `0.0`, 그 외 → `1.0`.
- **공간 해상도**: part 단위 (한 슈퍼복셀이 part 하나에 속하므로 동일 part 내 균일).
- **물리적 동기**: 다중 레이저 시스템에서 모듈별 캘리브레이션 차이 보정.

### #20  `laser_return_delay`  *(s)*
- **의미**: 1mm × 1mm 이웃 영역 안에서 *재용융 시간 간격*. 인접 픽셀이 시간 차이를 두고 다시 스캔될 때의 냉각 시간 proxy.
- **계산** (`scan_features.py`):
  1. `scans/{layer}` (M × 5 = `x_start, x_end, y_start, y_end, time`) 를 1842×1842 melt-time 맵으로 래스터화 (`build_melt_time_map`). Y 축은 build-plate ↔ image 변환 시 flip.
  2. 1mm 박스 커널(≈ 8 px) 안에서 `max(time) − min(time)` (`compute_return_delay_map`).
  3. `sat_s = 0.5 s` 로 clip (스트라이프 경계의 거대 점프 제거).
- **NaN 처리**: 미용융 픽셀은 NaN. 슈퍼복셀 안에 melt 픽셀이 1개도 없으면 누적 0 (= "스캔 활동 없음").
- **물리적 동기**: 짧은 return delay = 같은 자리 빠르게 재가열 → 누적열 증가, 미세조직 변화.

### #21  `laser_stripe_boundaries`  *(a.u. — Sobel RMS)*
- **의미**: 스트라이프(스캔 경계) 밀도. 스캔 패턴이 갈라지는 곳의 시간 점프 신호.
- **계산** (`scan_features.py:compute_stripe_boundaries_map`):
  1. melt-time 맵에서 NaN → 0 으로 채움.
  2. Sobel(축0), Sobel(축1) 의 RMS = √(Sx² + Sy²).
  3. 미용융 픽셀은 결과 = 0 (정의상 경계 신호 없음 + 슈퍼복셀 평균 시 NaN 회피).
- **물리적 동기**: 스트라이프 경계는 용융풀 간섭 / 결함 핵 생성 위치 — 밀도가 높을수록 잠재 결함 多.

스캔 피처 구현: `Sources/vppm/origin/scan_features.py`. 단위 테스트: `Sources/tests/test_scan_features.py`.

---

## 공간/시간 집계 규칙

각 슈퍼복셀(ix, iy, iz) 의 피처 값은 다음 절차로 계산:

```
for layer in [iz × 70, (iz+1) × 70):
    map_layer = (피처별 픽셀 맵 — 위 정의)
    patch     = map_layer[r0:r1, c0:c1]   # 슈퍼복셀의 7.52×7.52 px 영역
    valid     = (CAD 마스크 / melt 마스크 등 — 피처별)
    if valid.any():
        accum[v] += patch[valid].mean() × valid.sum()
        counts[v] += valid.sum()
feature[v] = accum[v] / counts[v]      # z-방향 픽셀 가중 평균
```

- **CAD 그룹 (#1–2)**: `cad_mask` 를 가중치로 사용 (CAD 픽셀이 많은 레이어가 더 큰 영향).
- **DSCNN 그룹 (#4–11)**: 동일하게 CAD 가중.
- **센서 그룹 (#12–18)**: 픽셀 맵 없이 z-블록 70 layer 단순 평균.
- **`build_height` (#3)**: 픽셀 집계 없이 슈퍼복셀의 z-중심.
- **`laser_module` (#19)**: part 단위 직접 할당 (집계 없음).
- **스캔 그룹 (#20–21)**: 슈퍼복셀 안에 melt 된 픽셀이 1개라도 있는 레이어만 카운트, 단순 평균.

---

## Ablation 그룹 인덱스

`Sources/vppm/common/config.py:FEATURE_GROUPS` (0-based):

```python
{
    "cad":    [0, 1, 2],                       # G3
    "dscnn":  [3, 4, 5, 6, 7, 8, 9, 10],        # G1
    "sensor": [11, 12, 13, 14, 15, 16, 17],     # G2
    "scan":   [18, 19, 20],                     # G4
}
```

서브 그룹(개별 채널 단독 ablation 등)은 `FEATURE_GROUPS_DSCNN_SUB`, `FEATURE_GROUPS_SENSOR_SUB`, `FEATURE_GROUPS_SCAN_SUB` 참조.

---

## 정규화 / 결측

- 학습 직전 `Sources/vppm/origin/run_pipeline.py` 에서 NaN 슈퍼복셀 드롭 후 z-score 정규화.
- 타겟(YS/UTS/UE/TE)은 [-1, 1] 정규화 → L1 손실.

---

## 참고 코드

- 피처 추출 메인: `Sources/vppm/origin/features.py` — `FeatureExtractor.extract_features`
- 스캔 경로 알고리즘: `Sources/vppm/origin/scan_features.py`
- 슈퍼복셀 그리드: `Sources/vppm/common/supervoxel.py`
- 설정 / 매핑: `Sources/vppm/common/config.py`
