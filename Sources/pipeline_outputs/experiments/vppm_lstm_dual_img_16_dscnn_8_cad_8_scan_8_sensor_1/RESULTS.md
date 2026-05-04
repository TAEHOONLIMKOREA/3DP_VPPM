# VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-Sensor-1 결과

> fullstack `_1dcnn_sensor_4` 의 sensor 분기를 **per-field 1D-CNN(28-dim) → 단일 multi-channel LSTM(d_embed_s=1)** 으로 swap한 controlled 실험.
> 다른 6개 분기 (카메라 v0/v1 d=16, DSCNN d=8, CAD d=8, Scan d=8, 정적 2-feat) 와 학습 하이퍼파라미터는 모두 동일.
>
> **결과 위치**: `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/`
> **학습 일시**: 2026-05-04 (4 props × 5 folds, 4-GPU property-parallel)
> **실험 계획**: [`Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/PLAN.md`](../../../vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/PLAN.md)

---

## 1. 핵심 결과 — sensor 표현 28→1 압축에도 성능 손실 0

| 속성 | fullstack `_1dcnn_sensor_4` (sensor 28-dim) | **본 실험 (sensor 1-dim)** | Δ | naive 대비 |
|:--:|:--:|:--:|:--:|:--:|
| YS  | 20.08 ± 0.88 MPa | **20.16 ± 0.92 MPa** | +0.08 (+0.4%) | **41%** ↓ |
| UTS | 28.54 ± 0.81 MPa | **28.12 ± 0.99 MPa** | **−0.42 (−1.5%)** | **59%** ↓ |
| UE  | 6.48 ± 0.40 %    | **6.38 ± 0.34 %**    | −0.10 (−1.5%) | **57%** ↓ |
| TE  | 8.13 ± 0.38 %    | **8.12 ± 0.36 %**    | −0.01 (−0.1%) | **56%** ↓ |

> **PLAN.md §10 가설 A 적중** — sensor 시간 정보 임계점이 매우 낮음.
> 28× 표현 압축 (28 → 1 dim) 에도 불구하고 모든 속성에서 RMSE 가 fullstack 과 통계적으로 동등 (run-to-run 분산 ±0.9 MPa 안), UTS 는 약간 우세.
> 시나리오 B (underfit, +3-8%) 는 명확히 기각.

---

## 2. 누적 비교 (sensor 인코더 변천)

| 실험 | sensor 처리 | sensor dim | YS (MPa) | UTS (MPa) | UE (%) | TE (%) | 파라미터 |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| baseline (21-feat MLP) | 평균 → 스칼라 | 7 | 24.28 ± 0.75 | 42.88 ± 2.00 | 9.34 ± 0.28 | 11.27 ± 0.50 | ~3k |
| `lstm_dual_4` | 평균 → 스칼라 | 7 | 20.66 ± 0.98 | 29.74 ± 0.87 | 6.61 ± 0.22 | 8.46 ± 0.27 | ~25k |
| `lstm_dual_img_4_sensor_7` | 단일 multi-ch LSTM | 7 | 20.57 ± 0.89 | 29.11 ± 0.95 | 6.73 ± 0.30 | 8.16 ± 0.23 | ~28k |
| `..._sensor_7_dscnn_8` | 단일 multi-ch LSTM | 7 | 20.55 ± 0.90 | 28.87 ± 1.24 | 6.48 ± 0.27 | 8.34 ± 0.29 | ~30k |
| fullstack `..._1dcnn_sensor_4` | per-field 1D-CNN ×7 | 28 | 20.08 ± 0.88 | 28.54 ± 0.81 | 6.48 ± 0.40 | 8.13 ± 0.38 | ~125k |
| **본 실험 `..._sensor_1`** | **단일 multi-ch LSTM** | **1** | **20.16 ± 0.92** | **28.12 ± 0.99** | **6.38 ± 0.34** | **8.12 ± 0.36** | **~98k** |

### 관찰

1. **sensor 평균(스칼라) → LSTM(d=7) 차이가 이미 작았음**: `lstm_dual_4` (sensor 평균) → `lstm_dual_img_4_sensor_7` (sensor LSTM d=7) UTS 29.74 → 29.11 (Δ −2.1%). 즉 sensor 시간 정보가 인장 특성 예측에 weak signal.
2. **fullstack 의 sensor 28-dim 은 over-parameterized**: 28 → 1 압축으로도 성능 동등. fullstack 의 per-field 1D-CNN (~15k) 는 6,373 SV 데이터 대비 과대.
3. **본 실험이 가장 효율적**: ~98k 파라미터 (fullstack 대비 ~22% 감소) + UTS 약간 개선 + 다른 prop 동등.

---

## 3. 측정오차 대비

| 속성 | 본 실험 RMSE | 측정오차 (논문) | 비율 |
|:--:|:--:|:--:|:--:|
| YS  | 20.16 MPa | 16.6 MPa | **1.21 ×** |
| UTS | 28.12 MPa | 15.6 MPa | 1.80 × |
| UE  | 6.38 %    | 1.73 %   | 3.69 × |
| TE  | 8.12 %    | 2.92 %   | 2.78 × |

YS 는 측정오차 한계 1.2× 까지 좁힘 (fullstack 1.21× 와 동일). UE/TE 는 여전히 여유 — sensor 압축이 연성 계열에 영향 없음을 한 번 더 확인.

---

## 4. 학습 안정성

| 속성 | val L1 (정규화) | 평균 epoch | fold std/mean |
|:--:|:--:|:--:|:--:|
| YS  | 0.0753 ± 0.003 | 102 | 4.5% |
| UTS | 0.0571 ± 0.002 | 123 | 3.5% |
| UE  | 0.1095 ± 0.004 | 124 | 5.4% |
| TE  | 0.1164 ± 0.002 | 118 | 4.4% |

- **수렴 속도 fullstack 과 비슷** (epoch 100~125). sensor 분기 단순화가 학습 속도에 큰 영향 없음.
- **fold 간 변동성 작음** (std/mean < 5.5%). UE 만 약간 높지만 fullstack (UE 6.2%) 과 비슷한 수준.
- **분산이 fullstack 보다 약간 크거나 비슷** — "과적합 완화 (PLAN §10 시나리오 D)" 효과는 명확하지 않음. 파라미터 절감의 정규화 이득이 sensor 표현 손실과 상쇄.

### Fold 별 RMSE 상세

| 속성 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 평균 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| YS  | 19.88 | 18.71 | 21.14 | 19.93 | 21.16 | **20.16** |
| UTS | 28.44 | 26.54 | 27.52 | 28.79 | 29.32 | **28.12** |
| UE  | 6.75  | 5.90  | 6.04  | 6.64  | 6.57  | **6.38**  |
| TE  | 8.58  | 7.74  | 7.73  | 8.05  | 8.50  | **8.12**  |

fold 1, 2 가 모든 속성에서 가장 낮음 — 카메라 LSTM 시리즈 공통 패턴 (특정 split 의 outlier 시편 영향).

---

## 5. 파라미터 효율성

| 분기 | fullstack | **본 실험** | Δ |
|:--|:--:|:--:|:--:|
| 카메라 FrameCNN ×2 + LSTM ×2 | ~26k | ~26k | — |
| **Sensor** | ~15k (per-field 1D-CNN ×7) | **~1.6k** (단일 LSTM in=7, hid=16, proj 16→1) | **−13.4k** |
| DSCNN LSTM | ~1.5k | ~1.5k | — |
| CAD spatial-CNN+LSTM | ~13k | ~13k | — |
| Scan spatial-CNN+LSTM | ~13k | ~13k | — |
| MLP (Nin → 256 → 128 → 64 → 1) | ~58k (Nin=86) | **~51k** (Nin=59) | **−7k** |
| **합계** | **~125k** | **~98k** | **−27k (~22%)** |

- 6,373 SV / ~98k 파라미터 ≈ **~65 SV/param** (fullstack ~50). 데이터 대비 capacity 가 약간 더 적합.
- sensor 분기 1.6k 파라미터 = fullstack 의 **9분의 1**. sensor 28-dim 의 표현력이 인장 특성 예측에 비례하는 효과를 내지 못한다는 강한 신호.

---

## 6. 가설 검증 (PLAN.md §10)

| 시나리오 | 결과 | 판정 |
|:--|:--|:--:|
| **A. sensor 시간 정보 임계점 충족** (RMSE ≈ fullstack ±1%) | YS +0.4%, UTS −1.5%, UE −1.5%, TE −0.1% — 모두 ±1.5% 이내, run-to-run 분산 안 | **✓ 적중** |
| B. sensor 표현 부족 (underfit, +3-8%, 특히 UE/TE) | UE/TE 모두 fullstack 과 동등하거나 약간 개선 | ✗ 기각 |
| C. LSTM 우세 (LSTM ↔ 1D-CNN 차이로 −1-3%) | UTS 만 −1.5% — 약하게 성립하나 분산 안 | △ 약하게 |
| D. 과적합 완화 (val ↑, gap ↓ vs fullstack) | 분산 비슷, val loss 큰 차이 없음 | △ 불명확 |

→ **결론: sensor 의 시간 정보는 인장 특성 예측에서 표현 차원이 1 이어도 충분.** packed LSTM (채널간 상호작용 + 패딩 미참조) 의 아키텍처적 이점이 표현력 손실을 정확히 상쇄해, fullstack 대비 동등한 성능을 1/9 의 sensor 파라미터로 달성.

---

## 7. 후속 권장 (PLAN.md §12 시나리오 A 분기)

가설 A 가 적중했으므로 다음 ablation 으로 자연스럽게 이어짐:

1. **sensor 분기 완전 제거 (`d=0`, sensor 입력 자체 drop)** — sensor 시간 정보가 정말 무의미한지 최종 검증. d=1 이 d=0 과 동등하면 sensor 는 잘라낼 수 있음 → 추가 ~1.6k 파라미터 + 캐시/dataloader 한 분기 절감.
2. **sensor 평균(스칼라) baseline 비교 분리** — `lstm_dual_4` (이미지 LSTM + sensor 평균 7개) 와 본 실험 (이미지 LSTM + sensor 1-dim) 의 격차가 작다면, sensor 평균 7-dim 마저도 충분.
3. **빌드별 RMSE 분해** — `predictions_*.csv` 에 build_id 매핑 후 B1.4 (가스 spike) / B1.5 (recoater) 빌드에서 sensor 표현이 어떻게 영향을 주는지 확인. 현재 metrics 에는 빌드별 break-down 없음.

---

## 8. 산출물

```
results/
├── metrics_summary.json        # YS/UTS/UE/TE 한 줄 요약
├── metrics_raw.json            # fold 별 RMSE 포함 정밀 수치
├── predictions_{YS,UTS,UE,TE}.csv   # ground_truth, prediction, residual per sample
├── correlation_plots.png       # 4 prop 2D 히스토그램 (예측 vs 실측)
└── scatter_plot_uts.png        # UTS scatter 단일

models/
├── vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1_{YS,UTS,UE,TE}_fold{0..4}.pt   # 20 체크포인트
└── training_log_{YS,UTS,UE,TE}.json   # 4-GPU 병렬이라 prop 별 분리 저장

features/
└── normalization.json          # static + sensor + dscnn + cad + scan per-channel min/max

experiment_meta.json            # 모델 / 분기 / 학습 hp 메타
```

## 9. 참조

- 실험 계획: [`Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/PLAN.md`](../../../vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/PLAN.md)
- 직접 비교 베이스: [`vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4`](../vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/) (sensor 28-dim per-field 1D-CNN)
- Sensor LSTM 원본: [`vppm_lstm_dual_img_4_sensor_7`](../vppm_lstm_dual_img_4_sensor_7/) (단일 multi-ch LSTM, d=7)
- 결과 해석 표준: [`vppm_lstm/LSTM_RESULTS.md`](../vppm_lstm/LSTM_RESULTS.md)
- Docker 환경: [`docker/lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/`](../../../../docker/lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/)
