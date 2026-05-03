# LSTM Ablation 결과

본 ablation 의 base 모델은 **`vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4`** (full 7-분기, MLP 입력 86).
이하 표에서 "E0" 는 이 base 모델의 기존 5-fold 학습 결과를 그대로 인용한 값.

| 실험 | 모델 | 유지된 분기 (Only-X 계열) | 제거된 분기 (No-X 계열) | MLP 입력 | 결과 위치 |
|:--:|:--|:--|:--|:--:|:--|
| **E0 (base)** | `vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4` | 전부 | — | 86 | `experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/results/` |
| **E1** | base − `branch_v0` | — | visible/0 카메라 | 70 | `experiments/lstm_ablation/E1_no_v0/results/` |
| **E2** | base − `branch_v0` − `branch_v1` | — | visible/0 + visible/1 카메라 | 54 | `experiments/lstm_ablation/E2_no_cameras/results/` |
| **E3** | only `branch_v0` | visible/0 카메라 단독 | feat_static, v1, sensor, dscnn, cad, scan | 16 | `experiments/lstm_ablation/E3_only_v0_img/results/` |
| **E4** | only `branch_dscnn` | DSCNN 8-class 단독 | feat_static, v0, v1, sensor, cad, scan | 8 | `experiments/lstm_ablation/E4_only_dscnn/results/` |
| **E5** | only `branch_cad` | CAD geometry patch 단독 | feat_static, v0, v1, sensor, dscnn, scan | 8 | `experiments/lstm_ablation/E5_only_cad/results/` |
| **E6** | only `branch_scan` | Scan path patch 단독 | feat_static, v0, v1, sensor, dscnn, cad | 8 | `experiments/lstm_ablation/E6_only_scan/results/` |
| **E7** | only `branch_sensor` | Sensor 7 필드 단독 | feat_static, v0, v1, dscnn, cad, scan | 28 | `experiments/lstm_ablation/E7_only_sensor/results/` |

학습 조건: 4 properties (YS / UTS / UE / TE) × 5 folds, n_samples = 6,373.

---

## RMSE 요약 (5-fold mean ± std)

| Property | E0 (base) | E1 (no v0) | E2 (no cameras) | E3 (only v0) | E4 (only dscnn) | E5 (only cad) | E6 (only scan) | E7 (only sensor) | Naive | 측정한계 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| YS  | 20.08 ± 0.88 | 20.12 ± 0.87 | 20.24 ± 0.87 | 21.42 ± 1.10 | 31.33 ± 0.27 | 35.15 ± 1.59 | 20.93 ± 0.98 | 32.26 ± 1.49 | 33.91 | 16.6 |
| UTS | 28.54 ± 0.81 | 28.27 ± 0.86 | 29.05 ± 1.23 | 31.01 ± 1.18 | 64.84 ± 1.76 | 75.34 ± 2.84 | 29.82 ± 1.49 | 69.39 ± 2.88 | 68.43 | 15.6 |
| UE  | 6.48 ± 0.40  | 6.49 ± 0.34  | 6.48 ± 0.34  | 7.17 ± 0.30  | 13.07 ± 0.67 | 16.30 ± 0.26 | 6.79 ± 0.27  | 15.58 ± 0.29 | 15.00 | 1.73 |
| TE  | 8.13 ± 0.38  | 8.19 ± 0.29  | 8.21 ± 0.29  | 9.09 ± 0.26  | 16.09 ± 0.51 | 18.73 ± 0.45 | 8.45 ± 0.29  | 18.39 ± 0.29 | 18.52 | 2.92 |

## Reduction vs Naive (%)

| Property | E0    | E1    | E2    | E3    | E4    | E5     | E6    | E7    |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| YS  | 40.77 | 40.67 | 40.32 | 36.82 |  7.59 |  −3.67 | 38.27 |  4.86 |
| UTS | 58.29 | 58.69 | 57.55 | 54.69 |  5.25 | −10.09 | 56.42 | −1.39 |
| UE  | 56.80 | 56.77 | 56.81 | 52.22 | 12.88 |  −8.68 | 54.76 | −3.84 |
| TE  | 56.10 | 55.78 | 55.68 | 50.89 | 13.13 |  −1.13 | 54.35 |  0.68 |

## ΔRMSE (vs E0)

| 비교 | YS | UTS | UE | TE |
|:--|:--:|:--:|:--:|:--:|
| E1 − E0 | +0.04  | −0.27  | +0.01 | +0.06  |
| E2 − E0 | +0.16  | +0.51  | 0.00  | +0.08  |
| E3 − E0 | +1.34  | +2.46  | +0.69 | +0.96  |
| E4 − E0 | +11.25 | +36.30 | +6.59 | +7.96  |
| E5 − E0 | +15.07 | +46.80 | +9.82 | +10.60 |
| E6 − E0 | +0.85  | +1.28  | +0.31 | +0.32  |
| E7 − E0 | +12.18 | +40.85 | +9.10 | +10.26 |

---

## Fold-by-fold RMSE

### YS (yield_strength)

| Fold | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 19.90 | 19.84 | 19.66 | 21.09 | 31.42 | 35.78 | 20.50 | 32.62 |
| 1 | 18.58 | 18.70 | 18.89 | 19.73 | 31.19 | 34.18 | 19.73 | 30.74 |
| 2 | 20.87 | 21.12 | 21.17 | 23.02 | 31.81 | 37.87 | 22.02 | 34.62 |
| 3 | 19.99 | 19.97 | 20.33 | 21.17 | 31.26 | 34.75 | 20.25 | 32.73 |
| 4 | 21.06 | 20.95 | 21.12 | 22.09 | 31.00 | 33.19 | 22.16 | 30.59 |

### UTS (ultimate_tensile_strength)

| Fold | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 29.07 | 28.32 | 28.41 | 30.68 | 62.48 | 76.04 | 28.54 | 70.15 |
| 1 | 27.38 | 26.86 | 27.58 | 29.07 | 64.44 | 74.58 | 28.52 | 67.55 |
| 2 | 27.79 | 27.99 | 28.24 | 30.88 | 66.32 | 79.14 | 28.92 | 72.67 |
| 3 | 28.97 | 28.75 | 30.39 | 31.86 | 67.33 | 76.46 | 30.93 | 71.77 |
| 4 | 29.51 | 29.43 | 30.64 | 32.54 | 63.64 | 70.48 | 32.21 | 64.79 |

### UE (uniform_elongation)

| Fold | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 7.10 | 7.00 | 7.02 | 7.66 | 12.83 | 16.44 | 7.20 | 15.56 |
| 1 | 6.06 | 6.01 | 6.01 | 6.75 | 12.77 | 16.30 | 6.65 | 15.47 |
| 2 | 6.03 | 6.23 | 6.15 | 7.10 | 13.99 | 16.73 | 6.38 | 16.11 |
| 3 | 6.69 | 6.60 | 6.66 | 7.04 | 12.12 | 16.02 | 6.86 | 15.55 |
| 4 | 6.53 | 6.59 | 6.56 | 7.29 | 13.63 | 16.04 | 6.85 | 15.21 |

### TE (total_elongation)

| Fold | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 8.68 | 8.37 | 8.71 | 9.60 | 15.56 | 18.58 | 8.84 | 18.36 |
| 1 | 7.84 | 7.77 | 7.81 | 8.82 | 16.14 | 18.91 | 8.26 | 18.27 |
| 2 | 7.65 | 7.92 | 7.94 | 9.00 | 16.62 | 19.50 | 8.11 | 18.95 |
| 3 | 8.44 | 8.39 | 8.31 | 9.02 | 15.46 | 18.36 | 8.75 | 18.25 |
| 4 | 8.03 | 8.49 | 8.28 | 9.02 | 16.66 | 18.28 | 8.31 | 18.14 |

---

## MLP 입력 차원

| 실험 | 입력 dim | static | v0 | v1 | sensor | dscnn | cad | scan |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| E0 | 86 | 2 | 16 | 16 | 28 | 8 | 8 | 8 |
| E1 | 70 | 2 | —  | 16 | 28 | 8 | 8 | 8 |
| E2 | 54 | 2 | —  | —  | 28 | 8 | 8 | 8 |
| E3 | 16 | — | 16 | —  | —  | — | — | — |
| E4 |  8 | — | —  | —  | —  | 8 | — | — |
| E5 |  8 | — | —  | —  | —  | — | 8 | — |
| E6 |  8 | — | —  | —  | —  | — | — | 8 |
| E7 | 28 | — | —  | —  | 28 | — | — | — |

---

## 산출 파일

각 실험 디렉터리:
- `results/metrics_raw.json` — fold별 RMSE 원시값 + naive_rmse + reduction
- `results/metrics_summary.json` — mean ± std 요약
- `results/predictions_{YS,UTS,UE,TE}.csv` — sample 단위 예측/실측
- `results/correlation_plots.png`, `scatter_plot_uts.png`
- `models/training_log.json` — fold별 epochs / val loss
- `models/vppm_lstm_ablation_{E1..E7}_{YS,UTS,UE,TE}_fold{0..4}.pt`
- `experiment_meta.json`
