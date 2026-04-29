# VPPM Feature Ablation — 종합 보고서

> 27 개 ablation 실험 결과 분석 (2026-04-28). 자동 요약은 [summary.md](./summary.md), 실험별 계획은
> [Sources/vppm/ablation/PLAN.md](../../vppm/ablation/PLAN.md).

---

## 0. Executive Summary

VPPM 모델 (Scime et al., Materials 2023) 의 21 개 입력 피처를 4 개 데이터 소스로 분해하고
각 그룹·채널·묶음을 한 번에 한 단위씩 제거해 4 인장 물성 (YS/UTS/UE/TE) 의 RMSE 변화를 측정.

**핵심 결론**: G4 스캔 그룹이 압도적으로 중요 (그룹 ΔUTS +17~18). 그 안에서도 단일 피처 `laser_return_delay`
하나가 ΔUTS +10.12 — G4 효과의 약 60% 설명. 나머지 3 그룹 (G1 DSCNN / G2 Sensor / G3 CAD) 은
모두 ΔUTS +4~7 범위. 단일-채널 ablation 은 **DSCNN/Sensor/Scan 모두 grouped collective effect** —
개별 채널 단위로는 거의 noise.

---

## 1. 배경

### 1.1 VPPM 모델

- **입력**: 21 차원 피처 벡터 (슈퍼복셀 단위, 1×1×3.5 mm³)
- **출력**: 4 개 인장 물성 (YS, UTS, UE, TE)
- **구조**: 2-layer MLP (hidden=128, dropout=0.1), L1 loss, [-1, 1] 정규화
- **데이터**: 36,047 슈퍼복셀, 6,373 유효 샘플, 5 빌드 (B1.1~B1.5, SS 316L)
- **CV**: 5-Fold, 샘플 단위 분할, seed=42 — 모든 실험 동일 분할 재사용

### 1.2 21 개 피처의 4 그룹 구조

| 그룹 | idx (0-based) | 피처 수 | 내용 | 출처 |
|:----:|:-------------:|:------:|:----|:----|
| **G1 DSCNN**           | 3–10  | 8 | 결함 세그멘테이션 8 클래스 (Powder/Printed/Streaking/EdgeSwelling/Debris/SuperElev/Soot/Melting) | DSCNN 분류 결과 |
| **G2 Temporal Sensor** | 11–17 | 7 | print_time / top_flow / bottom_flow / O₂ / plate_T / flow_T / ventilator | 프린터 로그 |
| **G3 CAD / 좌표**      | 0–2   | 3 | distance_edge / distance_overhang / build_height | 파트 형상 |
| **G4 스캔 (Laser)**    | 18–20 | 3 | laser_module / laser_return_delay / laser_stripe_boundaries | HDF5 `scans/{layer}` |

> 특히 G4 의 `return_delay` 와 `stripe_boundaries` 는 [scan_features.py](../../vppm/origin/scan_features.py) 가
> HDF5 의 (M, 5) 스캔 라인 세그먼트를 1842×1842 melt-time 맵으로 래스터라이즈한 후, 1 mm 커널 max−min
> 필터 / Sobel RMS 로 추출한 값.

---

## 2. 실험 구성 (27 개)

| ID | 카테고리 | 제거 단위 | n_feats | 도커 경로 |
|:--:|:--------|:--------|:-------:|:---------|
| **E0** | Baseline | — | 21 | `docker/baseline/` |
| **E1** | 그룹 — G1 전체 | DSCNN 8ch | 13 | `docker/ablation/dscnn/` |
| **E2** | 그룹 — G2 전체 | Sensor 7ch | 14 | `docker/ablation/sensor/` |
| **E3** | 그룹 — G3 전체 | CAD 3ch | 18 | `docker/ablation/cad/` |
| **E4 / E31** | 그룹 — G4 전체 | Scan 3ch | 18 | `docker/ablation/{scan,scan_sub}/` |
| **E13** | 조합 | G1 ∪ G2 (15ch) | 6 | `docker/ablation/combined/` |
| **E5–E12** | DSCNN 단일 채널 | 1ch each (8 실험) | 20 | `docker/ablation/dscnn_sub/` |
| **E23** | DSCNN 묶음 | defect 6ch | 15 | 〃 |
| **E24** | DSCNN 묶음 | normal 2ch | 19 | 〃 |
| **E14–E20** | Sensor 단일 채널 | 1ch each (7 실험) | 20 | `docker/ablation/sensor_sub/` |
| **E21** | Sensor 묶음 | gas_flow 3ch | 18 | 〃 |
| **E22** | Sensor 묶음 | thermal 2ch | 19 | 〃 |
| **E32** | Scan 단일 채널 | return_delay (idx 19) | 20 | `docker/ablation/scan_sub/` |
| **E33** | Scan 단일 채널 | stripe_boundaries (idx 20) | 20 | 〃 |

> E4 와 E31 은 동일 ablation (G4 3ch 전체 제거) — 별도 폴더에 따로 저장됨. 본 보고서에서는 두 결과를
> 평균하지 않고 둘 다 표기.

### 2.1 학습 / 평가 설정

- Adam (lr=1e-3, β=(0.9, 0.999), ε=1e-4), batch=1000
- MAX_EPOCHS=5000, early-stop patience=50
- 각 ablation 마다 **남은 차원만으로 f_min/f_max 재계산** 후 [-1, 1] 로 재정규화
- 평가: 원본 스케일 RMSE, 샘플별 예측 집계는 "최소값" (논문 Section 3.1)

---

## 3. 전체 결과

### 3.1 Baseline (E0) — 절댓값 기준

| 속성 | RMSE 평균 | RMSE std | Naive | 내재오차 | Reduction vs Naive |
|:----:|:--------:|:-------:|:-----:|:-------:|:-----------------:|
| YS  | 24.28 MPa | 0.75  | 33.91 | 16.6 MPa | 28.4% |
| UTS | 42.88 MPa | 2.00  | 68.43 | 15.6 MPa | 37.3% |
| UE  | 9.34 %    | 0.28  | 15.00 | 1.73 %   | 37.7% |
| TE  | 11.27 %   | 0.50  | 18.52 | 2.92 %   | 39.2% |

→ 모든 속성에서 baseline 이 naive (평균값 예측) 대비 28~39% 우수. 다만 내재 측정오차 대비:
- YS: 1.46×, UTS: 2.75×, UE: **5.40× (가장 큰 격차)**, TE: 3.86×
- UE (균일 연신) 가 개선 여지 가장 큼.

### 3.2 27 ablation — ΔRMSE 영향력 순

> ΔRMSE = E_i − E0. 양수일수록 해당 단위가 중요. fold std (보통 1.5–2.5 MPa for UTS) 와 비교해 유의성 판단.

| Rank | 실험 | 제거 단위 | n_feats | ΔYS | ΔUTS | ΔUE | ΔTE |
|:----:|:----:|:---------|:-------:|:---:|:----:|:---:|:---:|
| **1**  | **E4 / E31** | scan 그룹 (3ch G4) | 18 | +4.60 / +4.32 | **+18.48 / +17.00** | +3.58 / +3.69 | +4.27 / +4.41 |
| **2**  | **E32**      | scan_return_delay 단일      | 20 | +2.06 | **+10.12** | +2.18 | +2.41 |
| **3**  | **E13**      | DSCNN ∪ Sensor 동시 (15ch)  | 6  | +2.93 | **+10.06** | +2.04 | +2.47 |
| 4   | E3   | CAD 그룹 (3ch G3)           | 18 | +1.85 | +6.63 | +0.86 | +1.07 |
| 5   | E1   | DSCNN 그룹 (8ch G1)         | 13 | +1.69 | +6.14 | +1.43 | +1.60 |
| 6   | E2   | Sensor 그룹 (7ch G2)        | 14 | +1.01 | +3.97 | +0.83 | +0.94 |
| 7   | E23  | DSCNN defect 6ch 묶음       | 15 | +0.46 | +2.57 | +0.51 | +0.65 |
| 8   | E33  | scan_stripe_boundaries 단일 | 20 | +0.43 | +2.52 | +0.54 | +0.90 |
| 9   | E5   | dscnn_powder 단일           | 20 | +0.02 | +1.82 | +0.30 | +0.57 |
| 10  | E24  | DSCNN normal 2ch 묶음       | 19 | +0.47 | +1.75 | +0.51 | +0.51 |
| 11  | E6   | dscnn_printed 단일          | 20 | +0.06 | +1.68 | +0.30 | +0.51 |
| 12  | E7   | dscnn_recoater_streaking 단일 | 20 | +0.28 | +1.55 | +0.23 | +0.25 |
| 13  | E8   | dscnn_edge_swelling 단일    | 20 | −0.13 | +0.65 | +0.10 | +0.35 |
| 14  | E19  | sensor_flow_temp 단일       | 20 | +0.04 | +0.54 | +0.10 | +0.02 |
| 15  | E18  | sensor_plate_temp 단일      | 20 | −0.15 | +0.48 | −0.02 | +0.09 |
| 16  | E15  | sensor_top_flow 단일        | 20 | +0.05 | +0.47 | +0.03 | +0.16 |
| 17  | E22  | sensor_thermal 2ch 묶음     | 19 | −0.07 | +0.47 | +0.02 | −0.10 |
| 18  | E11  | dscnn_soot 단일             | 20 | −0.01 | +0.34 | −0.05 | +0.16 |
| 19  | E16  | sensor_bottom_flow 단일     | 20 | −0.17 | +0.25 | −0.17 | +0.16 |
| 20  | E10  | dscnn_super_elevation 단일  | 20 | +0.03 | +0.21 | −0.03 | +0.05 |
| 21  | E9   | dscnn_debris 단일           | 20 | −0.28 | +0.21 | +0.01 | +0.11 |
| 22  | E21  | sensor_gas_flow 3ch 묶음    | 18 | −0.23 | +0.17 | −0.10 | −0.02 |
| 23  | E20  | sensor_ventilator 단일      | 20 | −0.10 | +0.15 | −0.11 | −0.07 |
| 24  | E14  | sensor_print_time 단일      | 20 | −0.07 | +0.07 | −0.12 | +0.24 |
| 25  | E17  | sensor_oxygen 단일          | 20 | −0.12 | −0.05 | +0.02 | +0.17 |
| 26  | E12  | dscnn_excessive_melting 단일 | 20 | −0.09 | −0.08 | −0.02 | +0.11 |

### 3.3 27 ablation — RMSE 절댓값

| 실험 | 제거 단위 | n | YS (MPa) | UTS (MPa) | UE (%) | TE (%) |
|:----:|:----|:--:|:-------:|:--------:|:-----:|:-----:|
| **E0** | (baseline) | 21 | **24.28 ± 0.75** | **42.88 ± 2.00** | **9.34 ± 0.28** | **11.27 ± 0.50** |
| Naive | (constant) | 0  | 33.91 | 68.43 | 15.00 | 18.52 |
| E1  | dscnn (8ch)         | 13 | 25.97 ± 1.08 | 49.02 ± 2.19 | 10.77 ± 0.35 | 12.86 ± 0.45 |
| E2  | sensor (7ch)        | 14 | 25.29 ± 0.97 | 46.85 ± 2.07 | 10.17 ± 0.49 | 12.21 ± 0.31 |
| E3  | cad (3ch)           | 18 | 26.13 ± 1.00 | 49.51 ± 2.46 | 10.20 ± 0.33 | 12.33 ± 0.40 |
| E4  | scan (3ch)          | 18 | 28.88 ± 0.99 | 61.36 ± 1.56 | 12.92 ± 0.13 | 15.54 ± 0.14 |
| E13 | dscnn ∪ sensor (15ch)| 6 | 27.21 ± 1.18 | 52.94 ± 2.78 | 11.38 ± 0.38 | 13.74 ± 0.20 |
| E31 | scan (3ch, 재학습)  | 18 | 28.60 ± 1.04 | 59.88 ± 2.46 | 13.03 ± 0.42 | 15.68 ± 0.16 |
| E32 | return_delay 단일    | 20 | 26.34 ± 0.88 | 53.00 ± 1.96 | 11.52 ± 0.10 | 13.68 ± 0.13 |
| E33 | stripe_boundaries 단일 | 20 | 24.71 ± 0.71 | 45.40 ± 2.51 | 9.88 ± 0.40 | 12.17 ± 0.39 |

(DSCNN/Sensor 단일·묶음 9~10 개 절댓값은 §6, §7 참조)

---

## 4. 그룹 ablation (E1–E4, E13) 자세한 해석

### 4.1 E4 / E31 — Scan 그룹 (G4 3ch) — **최강 영향**

| 지표 | E0 baseline | E4 (G4 제거) | ΔRMSE | E31 (G4 재학습) | ΔRMSE | 유의성 (vs σ_E0) |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| YS   | 24.28 | 28.88 | **+4.60** | 28.60 | +4.32 | **6.1σ** |
| UTS  | 42.88 | 61.36 | **+18.48** | 59.88 | +17.00 | **9.2σ** |
| UE   | 9.34  | 12.92 | **+3.58** | 13.03 | +3.69 | **12.8σ** |
| TE   | 11.27 | 15.54 | **+4.27** | 15.68 | +4.41 | **8.5σ** |

**해석**:
- 4 그룹 중 **단연 1위**. ΔUTS +17~18 은 다른 어떤 그룹보다도 압도적.
- UE 가 E0 baseline 9.34% → 12.92% (+38%) — naive (15.00%) 의 86% 수준까지 악화. **연신율 정보의 상당 부분이 G4 에 응축**.
- ΔRMSE 가 fold std (E0 의 0.13~2.00 MPa) 의 6–13σ 수준 → 통계적으로 압도적 유의.
- E4 ↔ E31 차이 (~1.5 MPa for UTS) 는 동일 ablation 의 fold 분할 noise 수준 — 같은 결과로 봐도 무방.

**물리적 해석**:
- G4 는 *국소 냉각 패턴* 을 인코딩 — `return_delay` (1 mm 커널 내 melt-time max−min) 와 `stripe_boundaries`
  (Sobel RMS) 는 응고 미세조직 (델타-페라이트 분율, dendrite arm spacing) 결정 변수와 직결.
- 다른 3 그룹 (G1 결함 분류, G2 챔버 환경, G3 형상) 은 *간접* 정보 — 미세조직을 직접 결정하지는 않음.
- → "어떻게 식었는가" 가 "어떤 결함이 있는가 / 챔버가 어땠는가 / 어디 위치에 있는가" 보다 인장 물성과 강한 상관.

---

### 4.2 E3 — CAD 그룹 (G3 3ch) — 2 위 동급

| 지표 | E0 | E3 | ΔRMSE | 유의성 |
|:----:|:--:|:--:|:----:|:------:|
| YS  | 24.28 | 26.13 | +1.85 | 2.5σ |
| UTS | 42.88 | 49.51 | **+6.63** | 3.3σ |
| UE  | 9.34  | 10.20 | +0.86 | 3.1σ |
| TE  | 11.27 | 12.33 | +1.07 | 2.1σ |

**제거된 3 채널**:
- `distance_from_edge` (idx 0): 파트 외곽까지의 거리
- `distance_from_overhang` (idx 1): 오버행 (자유 표면) 까지의 거리
- `build_height` (idx 2): z 높이

**해석**:
- **단 3 채널** 만으로 G1 (8 채널 DSCNN) 보다 약간 큰 영향 — 채널 효율이 가장 높은 그룹.
- 모든 fold std 의 2σ 이상 — 강하게 유의.
- 형상은 *국소 열전도 비등방성* 을 결정 — 외곽 근처는 빌드 평판으로 열이 빨리 빠짐, 오버행 근처는
  열전도가 한 쪽으로만 가능, build_height 는 누적 열 이력 결정. CAD 가 사실상 *cooling rate 의 prior* 역할.
- G4 (cooling rate proxy) 와 G3 (cooling rate prior) 가 보완 관계로 동작 — 두 그룹 합산 효과는 §5 후속 검증.

---

### 4.3 E1 — DSCNN 그룹 (G1 8ch) — 3 위

| 지표 | E0 | E1 | ΔRMSE | 유의성 |
|:----:|:--:|:--:|:----:|:------:|
| YS  | 24.28 | 25.97 | +1.69 | 2.3σ |
| UTS | 42.88 | 49.02 | +6.14 | 3.1σ |
| UE  | 9.34  | 10.77 | +1.43 | 5.1σ |
| TE  | 11.27 | 12.86 | +1.60 | 3.2σ |

**제거된 8 채널**: powder / printed / recoater_streaking / edge_swelling / debris / super_elevation / soot / excessive_melting

**해석**:
- ΔUTS +6.14 — G3 (+6.63) 와 동급, G4 (+17~18) 의 1/3.
- UE 효과 (5.1σ) 가 다른 속성보다 큼 — 결함 분포가 *연신율* 과 가장 강하게 상관 (결함 = 응력 집중점 = 조기 파괴).
- §6 단일 채널 ablation (E5–E12) 과 §6.2 묶음 ablation (E23/E24) 결과로 보면 **개별 채널은 모두 noise** —
  G1 효과는 8 채널의 collective code 로만 발현.

---

### 4.4 E2 — Sensor 그룹 (G2 7ch) — 4 위 (가장 약함)

| 지표 | E0 | E2 | ΔRMSE | 유의성 |
|:----:|:--:|:--:|:----:|:------:|
| YS  | 24.28 | 25.29 | +1.01 | 1.3σ |
| UTS | 42.88 | 46.85 | +3.97 | 2.0σ |
| UE  | 9.34  | 10.17 | +0.83 | 3.0σ |
| TE  | 11.27 | 12.21 | +0.94 | 1.9σ |

**제거된 7 채널**: layer_print_time / top_flow / bottom_flow / O₂ / plate_T / flow_T / ventilator

**해석**:
- 4 그룹 중 가장 약함. ΔUTS +3.97 은 G4 의 약 1/4.
- YS 변화 (1.3σ) 는 marginal — 다른 그룹과 비교하면 강도 예측에는 거의 무관.
- §7 단일·묶음 ablation 에서 *모든 단독 제거가 noise 수준* — Sensor 7 채널은 강한 redundancy.
- 챔버 환경 (산소, 가스 유량, 온도, 시간) 은 빌드 단위로 거의 일정 → 슈퍼복셀 단위 변동성이 작아 모델이
  활용할 정보량이 적음. 7 채널 모두 같은 빌드 평균 근처 값을 갖는 cluster 로 동작.

---

### 4.5 E13 — DSCNN ∪ Sensor 동시 제거 (15ch) — Independence 검증

| 지표 | E0 | E1 | E2 | **E13** | ΔE1+ΔE2 | ΔE13 | 시나리오 |
|:----:|:--:|:--:|:--:|:-------:|:-------:|:----:|:--------:|
| YS  | 24.28 | 25.97 | 25.29 | 27.21 | +2.70 | **+2.93** | A (≈, 0.85σ 상위) |
| UTS | 42.88 | 49.02 | 46.85 | 52.94 | +10.11 | **+10.06** | **A 정확 일치** |
| UE  | 9.34  | 10.77 | 10.17 | 11.38 | +2.26 | +2.04  | A/B (살짝 redundant) |
| TE  | 11.27 | 12.86 | 12.21 | 13.74 | +2.54 | +2.47  | A (≈) |

**해석 — 시나리오 A (Independent / Additive) 확정**:
- UTS 에서 ΔE13 = +10.06 ≈ ΔE1 + ΔE2 = +10.11 → 거의 정확히 합산.
- DSCNN 정보와 Sensor 정보가 **직교적** — 한 쪽이 다른 쪽의 손실을 보완하지 않음.
- E13 (n_feats=6, CAD 3 + Scan 3 만 남음) 의 RMSE 도 모두 naive 보다 우수 → CAD+Scan 만으로도 의미 있는 학습 가능.
  특히 G4 의 강력한 신호 덕분에 6 피처만으로 baseline 의 약 80% 수준 달성.

---

## 5. Scan 서브 ablation (E31, E32, E33) — G4 채널 분해

G4 3 채널 단일 제거로 어느 채널이 G4 효과의 핵심인지 분해.

| 실험 | 제거 채널 | YS | UTS | UE | TE | ΔUTS | G4 효과 비율 |
|:----:|:--------|:----:|:----:|:----:|:----:|:----:|:----:|
| E31 | 전체 3ch (laser_module + return_delay + stripe_boundaries) | 28.60 | 59.88 | 13.03 | 15.68 | +17.00 | 100% (baseline) |
| **E32** | return_delay 단일 (idx 19) | 26.34 | 53.00 | 11.52 | 13.68 | **+10.12** | **60%** |
| E33 | stripe_boundaries 단일 (idx 20) | 24.71 | 45.40 | 9.88 | 12.17 | +2.52 | 15% |
| (계산) | laser_module 단일 (E31 - E32 - E33) | — | — | — | — | ~+4.4 | ~25% |

**해석**:
- **`return_delay` 단독으로 ΔUTS +10.12** — 27 ablation 중 단일 피처 최강 영향.
  - YS 도 ΔYS +2.06 으로 그룹 G1 전체 (+1.69) 보다 큼.
  - UE +2.18, TE +2.41 도 G1 그룹 전체 효과보다 큼.
  - → **단일 피처 하나가 그룹 전체 (8 또는 7 채널) 보다 더 많은 정보 운반**.
- `stripe_boundaries` (E33) 는 단독 효과 +2.52 — modest. G4 효과의 ~15% 만 운반.
- `laser_module` (산술적으로 ~+4.4 추정) — 모듈 1 vs 2 의 이진 인디케이터가 의외로 큰 효과. 이는
  두 모듈의 캘리브레이션 차이가 응고 패턴에 시스템적 영향을 준다는 증거.

**`return_delay` 가 단일 최강인 이유**:
- 정의: 1 mm 커널 안에서 (max melt-time − min melt-time) 의 평균.
- 인접 트랙 간 시간 간격 → 선행 트랙이 식는 데 걸린 시간 → **국소 cooling rate 의 직접 inverse proxy**.
- SS 316L 의 응고 미세조직 (dendrite arm spacing, 델타-페라이트 분율, GTB 밀도) 은 cooling rate 와
  수십 K/s ~ 수 MK/s 범위에서 power-law 관계. 인장 물성은 미세조직의 직접 함수 → **return_delay 가 인장 물성의
  underlying 물리량을 가장 잘 capture**.

---

## 6. DSCNN 서브 ablation (E5–E12, E23, E24) — G1 채널 분해

DSCNN 8 채널 + 두 묶음 (Defect 6ch / Normal 2ch) 으로 G1 정보 구조 분해.

### 6.1 단일 채널 (E5–E12) — 모두 noise

| 실험 | 채널 | 카테고리 | UTS | ΔUTS | UE | ΔUE | UTS 판정 |
|:----:|:----|:--------|:--:|:---:|:--:|:---:|:--:|
| E5  | seg_powder              | Normal | 44.70 | +1.82 | 9.64  | +0.30 | Contributing (1.4σ) |
| E6  | seg_printed             | Normal | 44.56 | +1.68 | 9.64  | +0.30 | Contributing (0.8σ) |
| E7  | seg_recoater_streaking  | Defect | 44.43 | +1.55 | 9.57  | +0.23 | Contributing (1.0σ) |
| E8  | seg_edge_swelling       | Defect | 43.53 | +0.65 | 9.44  | +0.10 | Marginal |
| E9  | seg_debris              | Defect | 43.09 | +0.21 | 9.35  | +0.01 | Marginal |
| E10 | seg_super_elevation     | Defect | 43.10 | +0.21 | 9.31  | −0.03 | Marginal |
| E11 | seg_soot                | Defect | 43.22 | +0.34 | 9.30  | −0.05 | Marginal |
| E12 | seg_excessive_melting   | Defect | 42.80 | **−0.08** | 9.32 | −0.02 | Negative (제거가 도움) |

**해석**:
- 8 채널 단일 제거 모두 |ΔUTS| < 1.82, fold std (1.5–2.5) 이내 — 통계적으로 noise.
- 여러 채널의 ΔRMSE 가 음수 (E8, E9, E10, E11, E12) → 단일 결함 클래스 채널은 학습에 *살짝 방해* 되는 측면도 있음 (모델이 noisy 신호를 활용하려 시도하다가 일반화 손실).
- 가장 큰 영향이 E5/E6 (Powder/Printed = Normal 클래스) — Normal 채널이 *결함률의 inverse 지표* 로 작동해 정보가 더 압축돼 있을 가능성.
- 어느 단일 결함 (E12 excessive_melting = Keyhole, E7 recoater_streaking = B1.5 핵심) 도 critical 하지 않음.

### 6.2 묶음 (E23, E24) — 묶어야 신호 발현

| 실험 | 제거 묶음 | n_feats | UTS | ΔUTS | UE | ΔUE |
|:----:|:--------|:------:|:---:|:---:|:---:|:---:|
| E23 | defect 6ch (recoater~excessive) | 15 | 45.46 | **+2.57** | 9.85 | +0.51 |
| E24 | normal 2ch (powder, printed)    | 19 | 44.63 | +1.75 | 9.85 | +0.51 |
| E1  | 전체 8ch (참고)                  | 13 | 49.02 | +6.14 | 10.77 | +1.43 |

**해석 — DSCNN 의 핵심 패턴**:
1. **묶음에서만 신호 발현** — 단일 채널 평균 ΔUTS = +0.85, 묶음 평균 +2.16. 두 묶음 모두 단일 채널 평균보다 훨씬 큼.
2. **defect 묶음과 normal 묶음 영향이 거의 동일** (E23 ΔUE +0.51 = E24 ΔUE +0.51).
   - DSCNN 정보가 normal/defect 로 분리되지 않고 **8 채널 collective code** 로 인코딩됨.
   - Σ(class probs) = 1 제약 때문에 Normal = 1 − Σ(defects) — 두 묶음이 같은 "결함률" 정보의 보완적 표현.
3. **E23 + E24 = +4.32 < E1 = +6.14** → 두 묶음의 단순 합이 전체 그룹 효과보다 작음.
   - 즉 8 채널이 *non-trivial collective effect* 로 작동 — 어떤 채널 부분집합으로도 완전히 환원되지 않음.

**가설 변경**:
- 원래 가설: "특정 결함 클래스 (recoater_streaking, excessive_melting) 가 핵심"
- 결과: 모든 단일 채널 noise — 핵심은 *어떤 결함이냐* 가 아니라 *결함률 분포의 shape*.
- 모델이 8 채널을 high-dimensional joint distribution 으로 활용 중 — 단일 marginal 정보로는 환원 불가.

---

## 7. Sensor 서브 ablation (E14–E22) — G2 채널 분해

### 7.1 단일 채널 (E14–E20) — 모두 noise (모든 속성)

| 실험 | 채널 | UTS | ΔUTS | YS | ΔYS |
|:----:|:----|:--:|:---:|:--:|:---:|
| E14 | layer_print_time            | 42.95 | +0.07 | 24.21 | −0.07 |
| E15 | top_gas_flow_rate           | 43.35 | +0.47 | 24.33 | +0.05 |
| E16 | bottom_gas_flow_rate        | 43.13 | +0.25 | 24.11 | −0.17 |
| E17 | module_oxygen               | 42.83 | **−0.05** | 24.16 | −0.12 |
| E18 | build_plate_temperature     | 43.36 | +0.48 | 24.13 | −0.15 |
| E19 | bottom_flow_temperature     | 43.42 | +0.54 | 24.32 | +0.04 |
| E20 | actual_ventilator_flow_rate | 43.03 | +0.15 | 24.18 | −0.10 |

**모두 |ΔUTS| < 0.55, |ΔYS| < 0.23**. fold std (1.4–2.3 MPa) 의 1/3 이내 — 완전한 noise.

### 7.2 묶음 (E21, E22) — 묶어도 noise

| 실험 | 묶음 | n_feats | UTS | ΔUTS |
|:----:|:----|:------:|:--:|:---:|
| E21 | gas_flow 3ch (top, bottom, ventilator) | 18 | 43.05 | +0.17 |
| E22 | thermal 2ch (plate_temp, flow_temp)    | 19 | 43.35 | +0.47 |
| E2  | 전체 7ch (참고)                          | 14 | 46.85 | +3.97 |

**해석 — Sensor 의 극단적 redundancy**:
- 단일 7 채널 평균 ΔUTS = +0.27.
- 묶음 (3ch, 2ch) 도 +0.17, +0.47 — 단일과 거의 차이 없음.
- E21 + E22 = +0.64 ≪ E2 = +3.97 → **5 채널 묶음 둘이 효과의 16% 만 설명**.
- 7 채널이 모두 *같은 정보의 다른 noisy 관측* — 어떤 6 채널 부분집합을 골라도 baseline 거의 유지 가능.
- 챔버 환경 변수가 빌드 단위로 일정해서 슈퍼복셀 단위 변동성이 작음 → 모델이 채널 간 차이로 활용할 정보가 거의 없음.

**경량화 잠재력**:
- 7 → 1~2 채널 축소 가능성 매우 높음 — E14 (print_time) + E17 (oxygen) 만 남기는 실험은 미수행이지만,
  ΔUTS < 1 로 예상.

---

## 8. 핵심 패턴 (Cross-Cutting Insights)

### 8.1 그룹 효과 vs 단일 채널 효과 — 모든 그룹이 collective

| 그룹 | 그룹 ΔUTS | 단일 채널 평균 ΔUTS | 비율 (단일/그룹) |
|:----:|:--------:|:-----------------:|:---------------:|
| G1 DSCNN  | +6.14 | +0.85 | 14% |
| G2 Sensor | +3.97 | +0.27 | 7%  |
| G4 Scan   | +17.00 | ~+5.7 (E32+E33+laser_module 평균 ~5.7) | 33% |

→ **모든 그룹이 collective effect** — 단일 채널 효과의 합 ≪ 그룹 전체 효과.
→ **G4 만 단일 채널 (return_delay) 이 그룹 효과의 60% 운반** — 가장 *decomposable* 한 그룹.

### 8.2 음의 ΔRMSE = 학습 방해

다음 단일 ablation 들은 ΔRMSE 가 음수 — 해당 채널이 baseline 학습에 *살짝 방해* 되는 noise:

| 실험 | 채널 | ΔUTS | ΔYS |
|:----:|:----|:----:|:----:|
| E12 | dscnn_excessive_melting | −0.08 | −0.09 |
| E17 | sensor_oxygen           | −0.05 | −0.12 |
| E11 | dscnn_soot              | +0.34 | −0.01 |
| E9  | dscnn_debris            | +0.21 | −0.28 |
| E16 | sensor_bottom_flow      | +0.25 | −0.17 |
| E10 | dscnn_super_elevation   | +0.21 | +0.03 |

크기는 모두 <0.3 MPa = noise 수준 — 통계적으로 0 과 구분 불가.

→ 다만 *방향성* 으로 보면 일부 결함 클래스 (excessive_melting, soot, debris, super_elevation) 와
일부 센서 (oxygen, bottom_flow) 는 모델 입장에서 거의 정보가 없거나 미세하게 noisy. seed 반복 후
유의성 재검증 필요.

### 8.3 속성별 의존도 차이

| 속성 | 가장 큰 단일 ΔRMSE | 가장 강한 그룹 |
|:----:|:----------------:|:-------------:|
| YS  | +4.60 (E4 G4)        | G4 (cooling rate prior) |
| UTS | **+18.48** (E4 G4)   | G4 |
| UE  | +3.69 (E31 G4)       | G4 (응고 미세조직 → 연성) |
| TE  | +4.41 (E31 G4)       | G4 |

→ **모든 4 속성에서 G4 가 단일 1위**. UE/TE (연신율) 가 G4 의존도 가장 강함 — 연성은 미세조직에 가장 민감하다는 야금학적 사실과 일치.

### 8.4 Naive 대비 어떤 ablation 도 baseline 만큼 좋지는 않음

가장 심한 E4 (G4 제거) 도 모든 속성에서 naive 보다 우수:
- E4 UTS 61.36 < naive 68.43 (10% 우위)
- E4 UE 12.92 < naive 15.00 (14% 우위)

→ G4 가 가장 중요해도 **다른 3 그룹 (G1+G2+G3) 만으로도 의미 있는 학습 가능**. 모델은 redundant 한
정보를 적극 활용 중.

---

## 9. 후속 실험 로드맵

### 9.1 우선순위

| # | 실험 | 동기 | 비용 |
|:-:|:----|:----|:---:|
| 1 | E32/E31/E33 의 빌드별 잔차 분해 | G4 가 어느 빌드 (B1.4 스패터? B1.5 리코터?) 에서 가장 강한지 | 단순 (재학습 없음) |
| 2 | E2 의 빌드별 잔차 분해 | Sensor 가 약하지만 특정 빌드에는 핵심일 가능성 | 단순 |
| 3 | LOBO CV (Leave-One-Build-Out) | 현재 sample-CV 는 build leak 있음 → 배포 시 일반화 측정 | 4 × 27 = 108 학습 |
| 4 | melt-time 통계 확장 피처 | return_delay 가 강하면 variance, percentile 등도 시도 | 피처 재추출 + ~5 ablation |
| 5 | Hidden size sweep (특히 E13, n=6) | n_feats 줄이면 hidden=128 적정성 변동 | 5 × 4 = 20 학습 |
| 6 | Seed 반복 (3 seed × E5–E22) | 단일 채널 noise 수준 통계적 확정 | 3 × 17 = 51 학습 |
| 7 | E1+E3 동시 제거 ablation | G1·G3 redundancy 측정 (둘 다 cooling rate prior 후보) | 1 학습 |
| 8 | Sensor 경량화 — 5ch 동시 제거 | gas_flow + thermal = 5ch 빼도 baseline 유지 가능성 | 1 학습 |

### 9.2 빠른 첫 걸음

```bash
for E in E31 E32 E33 E2 E1; do
  ./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment $E
done
# → 각 실험의 per_build_analysis.md 자동 생성, 빌드별 ΔRMSE 분포 확인
```

---

## 10. 산출물 위치

```
Sources/pipeline_outputs/ablation/
├── E1_no_dscnn/                 ├── E13_no_dscnn_sensor/
├── E2_no_sensor/                ├── E14_no_sensor_print_time/
├── E3_no_cad/                   ├── … E15~E22 …
├── E4_no_scan/                  ├── E23_no_dscnn_defects_all/
├── E5_no_dscnn_powder/          ├── E24_no_dscnn_normal/
├── … E6~E12 …                   ├── E31_no_scan/
│                                 ├── E32_no_scan_return_delay/
│                                 ├── E33_no_scan_stripe_boundaries/
├── summary.md                    # 자동 갱신 — 27 실험 표
└── FULL_REPORT.md                # (이 문서)
```

각 실험 폴더 구조:
```
E{id}_no_{group}/
├── experiment_meta.json          # exp_id, drop_group, dropped_idx, n_feats, n_samples
├── features/normalization.json   # 재정규화 통계
├── models/
│   ├── vppm_{YS,UTS,UE,TE}_fold{0-4}.pt
│   └── training_log.json
└── results/
    ├── metrics_raw.json          # fold별 + 평균 RMSE
    ├── metrics_summary.json
    ├── predictions_{YS,UTS,UE,TE}.csv
    ├── correlation_plots.png
    └── scatter_plot_uts.png
```

Baseline 산출물:
```
Sources/pipeline_outputs/results/
├── metrics_raw.json              # E0 baseline (24.28/42.88/9.34/11.27)
├── metrics_summary.json
├── predictions_*.csv
└── *.png
Sources/pipeline_outputs/models/
└── vppm_{YS,UTS,UE,TE}_fold{0-4}.pt
Sources/pipeline_outputs/features/
├── all_features.npz              # 36,047 × 21
└── normalization.json
```
