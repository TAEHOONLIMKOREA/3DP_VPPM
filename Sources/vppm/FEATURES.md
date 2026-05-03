# 슈퍼복셀 입력 21개 피처 명세

> **출처**: Scime et al., *Materials* 2023, 16, 7293 — Section 2.10 & Appendix D.
> **구현**: [Sources/vppm/baseline/features.py](features.py) (`FeatureExtractor.extract_features`).
> **단위**: 1 슈퍼복셀 = 1 mm × 1 mm × 3.5 mm = 7.52 × 7.52 px × 70 layer.
> **공통 처리**: 21개 피처 모두 (xy 패치 × 70 layer) → **단일 스칼라**로 압축. LSTM 계열에서도 본 21 피처는 그대로 스칼라 입력이며, 시간축 시퀀스로 들어가는 건 카메라 8×8 크롭(별도 스트림)뿐.

---

## 그룹 요약

| Group | 인덱스 (1-based / 0-based) | 출처 | 개수 |
|:----|:----|:----|:--:|
| **G3 — CAD/좌표 기하** | #1–3 / 0–2 | `slices/part_ids` (이진 형상 마스크) + SV 좌표 | 3 |
| **G1 — DSCNN 세그멘테이션** | #4–11 / 3–10 | `slices/segmentation_results/{0–11}` | 8 |
| **G2 — 프린터 센서 (Temporal)** | #12–18 / 11–17 | `temporal/*` (레이어별 1D 시계열) | 7 |
| **G4 — 레이저 스캔 시간 신호 피처** | #19–21 / 18–20 | `parts/process_parameters/laser_module`, `scans/{layer}` | 3 |

---

## 좌표계 / 픽셀 단위 (먼저 알아둘 것)

| 양 | 값 | 의미 |
|:--|:--|:--|
| 빌드 플레이트 | 245 × 245 mm | `REAL_SIZE_MM` |
| 카메라 이미지 | 1842 × 1842 px | `IMAGE_PIXELS` |
| `pixel_size_mm` | ≈ 0.1330 mm/px | 픽셀당 실측 길이 |
| `SV_XY_PIXELS` | ≈ 7.52 px | SV 한 변(이미지 슬라이싱 시 `int(round(7.52))=8` px) |
| `SV_Z_LAYERS` | 70 layer | SV z-방향 = 70 × 0.05 mm = 3.5 mm |
| `GAUSSIAN_STD_PIXELS` | ≈ 3.76 px | σ = 0.5 mm/px (CAD/DSCNN 블러용) |

### Y 축 flip — 스캔 데이터 ↔ 이미지 좌표 변환
- `scans/{layer}` 는 **build-plate Cartesian** (Y 위 방향, 원점 좌하단).
- `slices/*` 는 **이미지 좌표** (row 0 이 위, 원점 좌상단).
- 래스터화 시:
  ```
  col = round(x_mm / pixel_size_mm)              # X: 동일
  row = (H - 1) - round(y_mm / pixel_size_mm)    # Y: 반전
  ```
  ([scan_features.py:48-51](scan_features.py#L48-L51))

### 슈퍼복셀 → 픽셀 영역
`SuperVoxelGrid.get_pixel_range(ix, iy)` → `(r0, r1, c0, c1)`. 마지막 행/열은 이미지 끝까지 clip(잔여 ≤ 7 px).

---

## G3 — CAD/좌표 기하 피처 (#1–3)

3개 모두 `slices/part_ids` (uint, 0=배경 / 양수=part ID) 또는 SV 좌표만 사용. **CAD 픽셀이 많은 레이어에 더 큰 가중치**.

### #1 `distance_from_edge` *(mm)*

- **의미**: 각 픽셀에서 part 외곽까지의 in-plane(같은 레이어) Euclidean 거리.
- **출처**: `slices/part_ids` 한 레이어.
- **레이어별 픽셀 맵 계산** ([features.py:155-166](features.py#L155-L166)):
  ```python
  cad_mask = part_layer > 0                              # (1842, 1842) bool
  if cad_mask.any():
      dist = distance_transform_edt(cad_mask) * pixel_size_mm   # 외곽까지 mm
      dist = min(dist, DIST_EDGE_SATURATION_MM = 3.0 mm)        # ≥3 mm 동일 취급
      dist_smooth = gaussian_filter(dist, σ = 3.76 px)
  else:
      dist_smooth = zeros((1842, 1842))
  ```
  `scipy.ndimage.distance_transform_edt`: True 픽셀에서 가장 가까운 False 픽셀까지의 Euclidean 거리(픽셀 단위).
- **SV 단위 집계** (xy → 70 layer; [features.py:183-195](features.py#L183-L195)):
  ```
  for layer in [iz*70, (iz+1)*70):
      patch     = dist_smooth[r0:r1, c0:c1]   # ≈8×8 px
      patch_cad = cad_mask[r0:r1, c0:c1]
      n_cad     = patch_cad.sum()
      if n_cad > 0:
          accum  += patch[patch_cad].mean() * n_cad   # CAD 픽셀 수 가중
          counts += n_cad
  feature[v, 0] = accum / counts                       # CAD-가중 z-평균
  ```
- **물리적 동기**: 외곽 근처는 열 전달 비대칭 → 미세조직 차이.
- **포화 의미**: ≥ 3 mm = "part 안쪽" 으로 동일 취급 (외곽 효과는 ~3 mm 이내).

### #2 `distance_from_overhang` *(layers)*

- **의미**: 각 픽셀이 그 z-column 의 가장 최근 **오버행 표면** 위로 몇 layer 떨어져 있는지. 거리값이 작을수록 오버행 인접.
- **논문 정의** (Scime et al. 2023 Appendix D Table A2): "This distance is calculated for **a vertical column of pixels** with values allowed to saturate above 71 layers." → **수직(z-축) column 거리**.
- **오버행 정의**: 현재 레이어에 part 가 있고 **바로 직전(prev) 레이어**에는 없는 픽셀.
  ```
  overhang(x, y, L) = cad(x, y, L) ∧ ¬cad(x, y, L-1)
  ```
  → 분말층 위에 새로 출력되는 영역. 빌드 첫 레이어는 빌드 플레이트 위 출력이므로 overhang 미검출(saturate).
- **상태 누적** ([features.py:71-79](features.py#L71-L79)) — `extract_features` 진입 시 1회 초기화 후 **모든 z-block 에 걸쳐 carry-over**:
  ```python
  self._prev_cad_layer       = None                                # 직전 레이어의 CAD 마스크
  self._last_overhang_layer  = full(image, -inf, dtype=float32)    # (H, W) 픽셀별 가장 최근 overhang layer index
  ```
- **레이어별 픽셀 맵 계산** ([features.py:177-187](features.py#L177-L187)):
  ```python
  if self._prev_cad_layer is not None:
      overhang = cad_mask & (~self._prev_cad_layer)
      if overhang.any():
          self._last_overhang_layer[overhang] = float(layer)       # 최근 발생 시점 갱신
  # 빌드 첫 레이어(prev = None)는 갱신하지 않음 → 플레이트 위 출력은 overhang 아님

  dist_oh_layers = float(layer) - self._last_overhang_layer        # 미검출 픽셀: +inf
  dist_oh_layers = min(dist_oh_layers, DIST_OVERHANG_SATURATION_LAYERS = 71)
  dist_oh_smooth = gaussian_filter(dist_oh_layers, σ = 3.76 px)
  self._prev_cad_layer = cad_mask.copy()
  ```
- **단위**: **layers** (정수 z-축 거리). 코드상 모두 float32 로 보존되며, saturation 71 = `SV_Z_LAYERS + 1`.
- **SV 단위 집계**: #1 과 동일한 CAD-가중 z-평균.
- **z-블록 경계**: 상태가 build 전체에 걸쳐 carry-over 되므로 z-block 경계에서 거리값이 reset 되지 않음 (이전 구현과의 핵심 차이).
- **물리적 동기**: 오버행 영역은 분말 위에 직접 용융 → 결함 발생률 ↑. 인접도가 높을수록(거리 ↓) 결함 위험.

> **변경 이력**: 이전 구현은 같은 레이어 내 `~overhang` 의 in-plane EDT(픽셀 단위)를 계산했음 — 논문의 "vertical column" 정의와 어긋남. 본 버전은 논문 정의를 따라 z-축 layer 카운트로 재작성. 재학습이 동반되어야 새 값이 모델에 반영됨.

### #3 `build_height` *(mm)*

- **의미**: 슈퍼복셀의 z-방향 중심 높이 (빌드 플레이트 기준).
- **출처**: SV 좌표 `iz` 만 사용 — **픽셀 데이터 접근 없음**.
- **계산** ([supervoxel.py:64-68](../common/supervoxel.py#L64-L68), [features.py:96](features.py#L96)):
  ```python
  feature[v, 2] = ((l0 + l1) / 2) * LAYER_THICKNESS_MM
  # 예) iz=0  → (0+70)/2 × 0.05 = 1.75 mm
  # 예) iz=10 → (700+770)/2 × 0.05 = 36.75 mm
  ```
- **공간/시간 집계**: 없음. 같은 `iz` 의 모든 SV 가 동일 값.
- **물리적 동기**: 누적 열 이력, 분말 베드 두께 변화, 챔버 환경 시계열 효과를 단일 스칼라로 캡처.

---

## G1 — DSCNN 세그멘테이션 피처 (#4–11)

- **출처**: `slices/segmentation_results/{class_id}` (HDF5 12 클래스 → 논문 8 채널 매핑, [config.py:DSCNN_FEATURE_MAP](../common/config.py#L53)).
- **레이어별 픽셀 맵 계산** ([features.py:215-223](features.py#L215-L223)):
  ```python
  for cls in [0, 1, 3, 5, 6, 7, 8, 10]:                       # HDF5 cls id 8개
      seg = f["slices/segmentation_results/{cls}"][layer]      # (1842, 1842) 0/1
      seg_smoothed[ci] = gaussian_filter(seg.astype(float32), σ = 3.76 px)
  ```
- **SV 단위 집계**: G3 와 동일한 **CAD 가중 z-평균** ([features.py:225-234](features.py#L225-L234)). CAD 픽셀이 0인 레이어는 무시 (분말 영역 등).
- **값 범위**: [0, 1] — SV 내 CAD 픽셀 위에서 본 해당 결함 클래스의 평균 발생 비율.

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

매핑 정의: [Sources/vppm/common/config.py:DSCNN_FEATURE_MAP](../common/config.py#L53).

---

## G2 — 프린터 센서 피처 (#12–18)

- **출처**: `temporal/{key}` — (num_layers,) 1D 시계열 (빌드 전체 길이).
- **계산** ([features.py:104-108](features.py#L104-L108)):
  ```python
  for ti, key in enumerate(TEMPORAL_FEATURES):
      vals = f["temporal/{key}"][l0:l1]                  # 길이 70
      feature[v, 11+ti] = vals.mean()                    # 단순 산술 평균
  ```
- **공간 분포**: **한 z-블록 안의 모든 SV 가 동일 값**. 픽셀 맵 없음.
- **결측**: 키가 HDF5 에 없으면 NaN — 후속 NaN 마스크가 SV 전체를 드롭.

| # | 0-based | 이름 | HDF5 키 | 단위 | 설명 |
|:--:|:--:|:----|:----|:--:|:----|
| 12 | 11 | `layer_print_time` | `temporal/layer_times` | s | 레이어 1장 출력 시간 — 처리량 / 안정성 지표 |
| 13 | 12 | `top_gas_flow_rate` | `temporal/top_flow_rate` | L/min | 상부 보호가스 유량 |
| 14 | 13 | `bottom_gas_flow_rate` | `temporal/bottom_flow_rate` | L/min | 하부 보호가스 유량 |
| 15 | 14 | `module_oxygen` | `temporal/module_oxygen` | ppm/% | 챔버 산소 농도 — 산화 위험 지표 |
| 16 | 15 | `build_plate_temperature` | `temporal/build_plate_temperature` | °C | 빌드 플레이트 온도 |
| 17 | 16 | `bottom_flow_temperature` | `temporal/bottom_flow_temperature` | °C | 하부 가스 온도 |
| 18 | 17 | `actual_ventilator_flow_rate` | `temporal/actual_ventilator_flow_rate` | L/min | 실측 환기 유량 |

매핑 정의: [Sources/vppm/common/config.py:TEMPORAL_FEATURES](../common/config.py#L68).

---

## G4 — 레이저 스캔 시간 신호 피처 (#19–21)

스캔 그룹은 두 출처를 모두 사용 — `parts/process_parameters/laser_module` (part 메타) + `scans/{layer}` (레이저 경로). 좌표계 변환·래스터화·이웃 필터가 가장 정교한 그룹.

### #19 `laser_module` *(이진 0/1)*

- **출처**: `parts/process_parameters/laser_module` — (n_parts,) int.
- **계산** ([features.py:111-117](features.py#L111-L117), [features.py:137-143](features.py#L137-L143)):
  ```python
  lm_data = f["parts/process_parameters/laser_module"][...]
  laser_modules = {pid: int(lm_data[pid]) for pid where !nan}
  pid = valid_voxels["part_ids"][v]
  feature[v, 18] = 0.0 if laser_modules[pid] == 1 else 1.0
  ```
- **공간 해상도**: part 단위. SV → 우세 part_id (find_valid_supervoxels) → laser_module 룩업.
- **z 집계**: 없음 (한 part 내 모든 SV 동일 값).
- **물리적 동기**: 다중 레이저 시스템(예: 2 모듈)에서 모듈별 캘리브레이션·출력 변동 보정 신호.

### #20 `laser_return_delay` *(s)*

같은 영역에 레이저가 다시 돌아오기까지의 시간 차 = 냉각 시간 proxy.

#### Step 1 — Rasterization ([scan_features.py:24-119](scan_features.py#L24-L119))

- 입력: `scans/{layer}` = (M, 5) — 각 행 = `(x_start, x_end, y_start, y_end, time)`. 단위 mm, s.
- 좌표 변환 (Y flip):
  ```python
  c0 = round(x_start / pixel_size_mm);  c1 = round(x_end / pixel_size_mm)
  r0 = (H-1) - round(y_start / pixel_size_mm)
  r1 = (H-1) - round(y_end   / pixel_size_mm)
  ```
- 영역 밖 세그먼트 폐기 (양 끝점 모두 밖이면 drop, 한쪽만 밖이면 부분 라인 유지).
- 세그먼트 길이별 분기 (성능 최적화):
  - **Sub-pixel 세그먼트** (`max(|c1-c0|, |r1-r0|) == 0`, 약 89%) — 끝점 1픽셀만 기록. `np.minimum.at(mt, (r,c), t)` 한 번에 처리.
  - **Multi-pixel 세그먼트** — `L = length+1` 픽셀로 선형 보간:
    ```
    frac = local_idx / (L-1)        # 0..1
    r = round(r0 + frac*(r1-r0))
    c = round(c0 + frac*(c1-c0))
    ```
    벡터화: `np.repeat` + `linspace` 합산.
- 충돌 처리: 같은 픽셀에 여러 세그먼트 → **가장 빠른 시각** (`np.minimum.at`) — 최초 용융 시각만 기록.
- **출력**: `mt_map (1842, 1842) float32` — 미용융 픽셀 = NaN. 메모리 13 MB, 영구 캐시하지 않고 레이어별 즉시 계산 + 폐기.

#### Step 2 — Return delay map ([scan_features.py:125-163](scan_features.py#L125-L163))

```python
kernel_px = round(1 mm / pixel_size_mm) = 8                 # 정사각형 박스
mt_for_max = where(valid, mt_map, -inf)                      # NaN sentinel
mt_for_min = where(valid, mt_map, +inf)
max_map = scipy.ndimage.maximum_filter(mt_for_max, size=8)
min_map = scipy.ndimage.minimum_filter(mt_for_min, size=8)
valid_kernel = isfinite(max_map) & isfinite(min_map)         # 커널 안에 유효 픽셀 0개면 NaN
delay = max_map - min_map                                    # (s)
delay = min(delay, sat_s = 0.75)                             # saturation
delay[~mt_valid] = NaN                                       # 미용융 픽셀은 항상 NaN
```
- 1 mm 이웃 안 `max(time) − min(time)` = 같은 영역이 다시 스캔되는 데 걸린 시간.
- `sat_s = 0.75 s` clip: 스트라이프 경계 / 분리된 부품 사이의 거대 점프(여러 분 단위) 제거 — 같은 mm 안에서의 의미 있는 재용융 간격만 남김. **값 출처**: Scime et al. 2023 Appendix D Table A5 가 보고한 #20 의 max(0.750)에 맞춰 정함 — 이전 코드의 `0.5` 는 자연 신호 일부를 잘라먹는 더 공격적인 클립이었음.

#### Step 3 — SV 단위 집계 ([features.py:240-288](features.py#L240-L288))

```python
for layer in [l0, l1):
    if "scans/{layer}" not in f: continue
    mt = build_melt_time_map(scans, (1842, 1842), pixel_size_mm)
    rd_map = compute_return_delay_map(mt, kernel_px=8, sat_s=0.75)
    sb_map = compute_stripe_boundaries_map(mt)               # #21 도 같이 계산
    for each SV in z-block:
        rd_patch = rd_map[r0:r1, c0:c1]
        rd_valid = ~isnan(rd_patch)                          # SV 안 melt 픽셀
        if rd_valid.any():
            accum[v, 0]  += rd_patch[rd_valid].mean()
            accum[v, 1]  += sb_patch[rd_valid].mean()        # #21 동일 mask
            counts[v]    += 1                                 # 레이어 단위 카운트
feature[v, 19] = accum[v, 0] / counts[v]                     # 단순 평균 (레이어 가중 동일)
# 70 layer 모두 melt 없음 → 0 (NaN 으로 두면 SV 전체 드롭됨, 의도적으로 0)
```

- **G3/G1 와의 누적 규칙 차이**: G3/G1 은 "CAD 픽셀 수" 가중, G4 는 "유효 레이어(melt 1픽셀 이상)" 단위 단순 평균.
- **물리적 동기**: 짧은 return delay = 같은 영역 빠르게 재가열 → 누적열 ↑, 결정립 / 석출물 변화. 0.5 s clip 으로 부품 간 시간 점프 제거 → 진짜 "재용융 간격" 만 보전.

### #21 `laser_stripe_boundaries` *(a.u. — Sobel RMS)*

스캔 패턴(스트라이프) 경계 = melt-time 의 시간 점프가 큰 곳 = 용융풀 간섭 / 결함 핵 발생 위치.

#### Step 1 — Rasterization
#20 과 동일한 `mt_map` 재사용 (한 레이어당 한 번만 계산).

#### Step 2 — Sobel RMS ([scan_features.py:169-190](scan_features.py#L169-L190))

```python
mt_filled = nan_to_num(mt_map, nan=0.0).astype(float32)      # NaN → 0
sx = sobel(mt_filled, axis=0, mode="constant", cval=0.0)     # row 미분
sy = sobel(mt_filled, axis=1, mode="constant", cval=0.0)     # col 미분
sb = sqrt(sx² + sy²).astype(float32)                          # gradient magnitude
sb[~mt_valid] = 0.0                                           # 미용융 픽셀 = 0
```
- 미용융 픽셀에서 NaN→0 으로 인공 경계가 생기지만, SV 평균 시 melt 픽셀(`rd_valid`)만 카운트하므로 영향 미미.
- 미용융 픽셀에 0 을 넣는 건 NaN 의 평균 전파를 회피하기 위한 의도적 선택.

#### Step 3 — SV 단위 집계
#20 의 Step 3 와 동일 (같은 `rd_valid` 마스크, 같은 레이어 카운트).

- **물리적 동기**: 스트라이프 경계 = 인접 스캔 라인 사이 시간차 큰 곳. 용융풀 간섭이 약하고 미용융·산화·LOF 결함 발생 위치. 밀도가 높을수록 잠재 결함 多.

---

## 공간/시간 집계 규칙 — 그룹별 비교

| 그룹 | xy 패치 (한 레이어) | z-축 (70 layer) | 레이어 가중치 | 결측 처리 |
|:--|:--|:--|:--|:--|
| **G3 #1–2** | CAD 픽셀 위 평균 | 가중 평균 | `n_cad_pixels` | NaN (이론상 발생 X) |
| **G3 #3** | — | (직접 할당) | — | — |
| **G1 #4–11** | CAD 픽셀 위 평균 | 가중 평균 | `n_cad_pixels` | NaN |
| **G2 #12–18** | (공간 균일) | 단순 평균 | 레이어당 1 | 키 부재 시 NaN |
| **G4 #19** | (part 단위) | (직접 할당) | — | NaN (lm = NaN) |
| **G4 #20–21** | melt 픽셀 위 평균 | 단순 평균 | melt 있는 레이어당 1 | 모두 미용융 → 0 (의도적) |

### 통합 의사코드

```
for each SV (ix, iy, iz):
    accum, counts = 0, 0
    for layer in [iz*70, (iz+1)*70):
        map_layer = featureSpecificMap(layer)        # 위 정의
        patch     = map_layer[r0:r1, c0:c1]          # ≈ 8×8 px
        valid     = featureSpecificMask(layer)       # CAD 마스크 / melt 마스크 / 항상 True
        if valid.any():
            weight  = valid.sum()  if group in (G3, G1)  else 1
            accum  += patch[valid].mean() * weight
            counts += weight
    feature[v] = accum / counts                       # NaN 가능 (counts == 0)
```

---

## 평균 처리 방식별 분류 (1D CNN 시퀀스화 관점)

위 G1/G2/G3/G4 는 **데이터 출처**별 분류고, 1D CNN 으로 z-축 시퀀스 입력을 만들 때는 **레이어 평균이 어떻게 들어갔는지**가 더 중요하다. baseline 의 21개 피처는 **평균 처리 방식**을 기준으로 4가지 패턴 (P1–P4)으로 갈라진다.

| 패턴 | # | 피처 수 | 픽셀 평균 (xy patch) | 레이어 평균 (z) | 1D CNN 시퀀스화 |
|:--:|:--:|:--:|:--|:--|:--:|
| **P1** | 1, 2, 4–11 | 10 | ✅ CAD 픽셀에 한정한 평균 | ✅ **CAD 픽셀 수 가중** | ✅ |
| **P2** | 20, 21 | 2 | ✅ NaN-제외 melt 픽셀 평균 | ✅ **유효 레이어 수 단순평균** | ✅ |
| **P3** | 12–18 | 7 | ❌ (1D 시계열) | ✅ 70-layer 단순평균 | ✅ |
| **P4** | 3, 19 | 2 | ❌ | ❌ (스칼라 직접 할당) | ❌ |

### P1 — 픽셀 평균 + 레이어 CAD-가중평균 (10개)

[features.py:_extract_cad_features_block](features.py#L156-L216), [features.py:_extract_dscnn_features_block](features.py#L218-L258):

```python
for layer in range(l0, l1):                                        # 70 layers
    patch_cad = cad_mask[r0:r1, c0:c1]
    n_cad     = patch_cad.sum()
    if n_cad > 0:
        accum  += dist_smooth[r0:r1, c0:c1][patch_cad].mean() * n_cad   # ① 픽셀 평균 (CAD-only)
        counts += n_cad
accum[valid] /= counts                                              # ② 레이어 CAD-가중평균
```

→ #1 distance_from_edge, #2 distance_from_overhang, #4–11 DSCNN 8개. raw 정보 shape `(70 layers, 8×8 픽셀, CAD-only)` → 2단계 평균 → 스칼라 1개.

### P2 — 픽셀 평균 + 레이어 단순평균 (2개)

[features.py:_extract_scan_features_block](features.py#L260-L308):

```python
for layer in range(l0, l1):
    rd_valid = ~np.isnan(rd_patch)
    if rd_valid.any():
        accum[vi, 0] += rd_patch[rd_valid].mean()                   # ① 픽셀 평균 (NaN 제외)
        counts[vi]   += 1
out[ok] = accum[ok] / counts[ok]                                     # ② 레이어 단순평균
```

→ #20 laser_return_delay, #21 laser_stripe_boundaries. P1 과 차이: **CAD-가중 대신 "스캔 데이터 있는 layer 수"로 단순평균**.

### P3 — 레이어 단순평균만 (7개)

[features.py:104-108](features.py#L104-L108):

```python
vals = temporal_data[key][l0:l1]                                    # (70,) layer-level 시계열
features[block_indices, 11+ti] = np.mean(vals)                       # 레이어 단순평균
```

→ #12–18 프린터 센서 7개. 이미 layer 단위 1D 시계열이므로 픽셀 처리 없음. 같은 z-블록 내 모든 supervoxel 이 **동일 값** 공유.

### P4 — 평균 없음 (2개)

[features.py:96](features.py#L96), [features.py:140](features.py#L140):

```python
features[block_indices, 2]  = self.grid.get_z_center_mm(iz)           # #3 build_height
features[pidx, 18]          = 0.0 if laser_modules[pid] == 1 else 1.0 # #19 laser_module
```

→ #3 build_height (z-블록 중심 위치), #19 laser_module (part 단위 binary). 픽셀도 layer 도 보지 않음.

### 1D CNN 시퀀스화 입력 결정

P1+P2+P3 = **19개 채널** 을 `(70 layers, 19)` 시퀀스로 변환 가능.
P4 = **2개 스칼라** 는 시퀀스화 의미 없음 (#3 은 등차수열, #19 는 상수).

상세 설계는 [Sources/vppm/1dcnn/PLAN.md](../1dcnn/PLAN.md#23-어떤-피처를-시퀀스화하는가) 참조.

---

## Ablation 그룹 인덱스

[Sources/vppm/common/config.py:FEATURE_GROUPS](../common/config.py) (0-based):

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

- 학습 직전 [run_pipeline.py](../run_pipeline.py)에서 NaN 슈퍼복셀 드롭 후 z-score 정규화.
- 타겟(YS/UTS/UE/TE)은 [-1, 1] 정규화 → L1 손실.
- NaN 발생 케이스:
  1. **G2 센서**: `temporal/{key}` 가 HDF5 에 없을 때 → 해당 SV 드롭.
  2. **G3 #1–2 / G1**: z-블록 70 layer 모두 CAD 픽셀 0 — 이론상 valid SV 에서는 발생 X.
  3. **G4 #20–21**: melt 픽셀 0 → **0 으로 채움** (NaN 아님 — 드롭 회피용 의도적 처리).
  4. **G4 #19**: `laser_module` 이 NaN 인 part → SV 드롭.

---
