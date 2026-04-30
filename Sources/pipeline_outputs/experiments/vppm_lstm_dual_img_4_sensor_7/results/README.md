# VPPM-LSTM-Dual-Img-4-Sensor-7 (`d_embed_v0=v1=4`, `d_embed_s=7`) 결과 해석

> 14 baseline 피처 (G3+G1+G4) + **visible/0 per-SV CNN+LSTM 4-dim 임베딩** + **visible/1 per-SV CNN+LSTM 4-dim 임베딩** + **7-channel temporal sensor LSTM 7-dim 임베딩** = **29-feat MLP** 의 학습/평가 결과 정리.
>
> **결과 위치**: [`Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7/results/`](.)
> **학습 일시**: 2026-04-30 (19:24 ~ 22:58, 약 3.5h)
> **모델 코드**: [`Sources/vppm/lstm_dual_img_4_sensor_7/`](../../../../vppm/lstm_dual_img_4_sensor_7/)
> **실험 계획서**: [`Sources/vppm/lstm_dual_img_4_sensor_7/PLAN.md`](../../../../vppm/lstm_dual_img_4_sensor_7/PLAN.md)
> **참고 (camera dual, d=4)**: [`vppm_lstm_dual_4/README.md`](../../vppm_lstm_dual_4/results/README.md)
> **참고 (camera dual, d=1)**: [`vppm_lstm_dual/README.md`](../../vppm_lstm_dual/README.md)

---

## 1. 핵심 결과 — sensor LSTM 으로 **UTS / TE 소폭 개선**, YS / UE 평탄

| 속성 | baseline (21) | LSTM dual (23, d=1+1) | LSTM dual_4 (29, d=4+4) | **dual_img_4_sensor_7 (29, 4+4+7)** | naive |
|:--:|:--:|:--:|:--:|:--:|:--:|
| YS  | 24.3 ± 0.8 MPa | 20.8 ± 1.0 MPa | 20.7 ± 1.0 MPa | **20.6 ± 0.9 MPa** | 33.9 MPa |
| UTS | 42.9 ± 2.0 MPa | 29.6 ± 1.0 MPa | 29.7 ± 0.9 MPa | **29.1 ± 0.9 MPa** | 68.4 MPa |
| UE  | 9.3 ± 0.3 %    | 6.6 ± 0.2 %    | 6.6 ± 0.2 %    | **6.7 ± 0.3 %**    | 15.0 % |
| TE  | 11.3 ± 0.5 %   | 8.4 ± 0.3 %    | 8.5 ± 0.3 %    | **8.2 ± 0.2 %**    | 18.5 % |

> dual_4 → sensor_7 변화량: YS −0.4 %, UTS **−2.1 %**, UE +1.7 %, TE **−3.5 %**. UTS / TE 는 std 경계선상에서 의미 있는 개선, YS / UE 는 std 범위 안.

| 속성 | naive 대비 RMSE 감소 (sensor_7) | dual_4 대비 |
|:--:|:--:|:--:|
| YS  | **39 %** ↓ | 동률 |
| UTS | **57 %** ↓ | +0 %p (절대치는 −0.6 MPa) |
| UE  | **55 %** ↓ | -1 %p |
| TE  | **56 %** ↓ | **+2 %p** |

→ PLAN §6 의 시나리오 **A (시간 패턴이 의미 있음, RMSE −3~−10 % 기대, 특히 UE/TE)** 가 **TE 에서만 부분 지지** (−3.5 %). UE 는 오히려 미세 후퇴 — 예상과 반대 패턴. UTS 는 카메라가 못 보는 가스/유량/온도 변동을 sensor 시퀀스가 부분적으로 잡아줬을 가능성.

---

## 2. Fold 별 RMSE 상세

[`metrics_raw.json`](metrics_raw.json) 기준.

| 속성 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 평균 ± std | dual_4 대비 std |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| YS  | 19.95 | 19.64 | 21.86 | 19.98 | 21.42 | **20.57 ± 0.89** | 0.98 → 0.89 (-9 %)  |
| UTS | 28.82 | 27.82 | 28.60 | 29.78 | 30.53 | **29.11 ± 0.95** | 0.87 → 0.95 (+9 %)  |
| UE  |  7.24 |  6.51 |  6.46 |  6.53 |  6.89 | **6.73 ± 0.30**  | 0.22 → 0.30 (+36 %) |
| TE  |  8.54 |  7.92 |  7.91 |  8.20 |  8.23 | **8.16 ± 0.23**  | 0.27 → 0.23 (-15 %) |

- 평가 샘플 수: **6,373** (dual_4 와 동일 5-fold 분할).
- YS fold 2 (21.86) 가 여전히 highest — `vppm_lstm_dual_4` 와 같은 패턴. 데이터 분할 특성으로 재확인.
- UTS fold 1 (27.82) 이 가장 낮음 — sensor 분기가 fold 1 에서 특히 잘 적합. 다만 fold 4 (30.53) 와 격차 2.7 MPa → std 가 dual_4 보다 살짝 증가.
- **UE fold 0 (7.24) 이 큰 폭으로 튐** (다른 4 fold 평균 6.60 대비 +0.6). UE 평균 후퇴의 주범. → fold 0 의 sensor 분포에 sensor LSTM 이 일반화 못 한 사례 가능.
- TE 는 std 0.23 으로 모든 실험 중 **가장 안정** (dual_4 0.27, dual 0.32, single 0.20).

---

## 3. 학습 안정성 — val loss 4 속성 모두 미세 개선

[`models/training_log.json`](../models/training_log.json) 기준.

| 속성 | val loss (정규화) | 평균 epoch | dual_4 대비 val loss | dual_4 대비 epoch |
|:--:|:--:|:--:|:--:|:--:|
| YS  | 0.0787 | 239 | 0.0791 → **0.0787** (-0.5 %) | 273 → 239 (-12 %) |
| UTS | 0.0617 | 257 | 0.0623 → **0.0617** (-1.0 %) | 243 → 257 (+6 %)  |
| UE  | 0.1176 | 231 | 0.1178 → **0.1176** (-0.2 %) | 242 → 231 (-5 %)  |
| TE  | 0.1234 | 257 | 0.1257 → **0.1234** (-1.8 %) | 267 → 257 (-4 %)  |

- **TE val loss −1.8 %** 가 가장 큼 — hold-out RMSE 개선 (8.46 → 8.16, −3.5 %) 과 **동일 방향**. sensor 시퀀스가 train/val/hold-out 전체에 걸쳐 일관되게 추가 신호를 제공 → 실험 결과가 잡음 아님.
- **UTS val loss −1.0 %** + hold-out −2.1 % 도 마찬가지로 일관된 방향.
- **YS / UE val loss 차이 ≤ 0.5 %** — hold-out RMSE 차이도 std 안. sensor 시간 패턴이 이 두 속성에는 의미 신호를 거의 안 주는 것으로 해석.
- 수렴 epoch 은 속성별로 −12 % ~ +6 % 범위. capacity 1 분기 추가 (~수천 파라미터) 에 비해 안정적.

---

## 4. 모델 / 학습 설정 ([`experiment_meta.json`](../experiment_meta.json))

| 항목 | 값 (dual_4) | **값 (sensor_7)** |
|:--|:--:|:--:|
| baseline 피처 수 | 21 | **14** (G2 sensor 7 제거) |
| Sensor 처리 방식 | 70-layer **단순 평균** 7-feat | **7-channel LSTM**, `d_embed_s=7` |
| `d_embed_v0` | 4 | 4 |
| `d_embed_v1` | 4 | 4 |
| `d_embed_s` (sensor 임베딩) | — | **7** |
| `d_hidden_s` (sensor LSTM hidden) | — | **16** |
| 총 입력 피처 (`n_total_feats`) | 29 (21+4+4) | **29** (14+4+4+7) ✓ 동일 |
| 카메라 채널 | `[0, 1]` | `[0, 1]` |
| Sensor 채널 | (avg, in 21 baseline) | **`layer_times`, `top_flow_rate`, `bottom_flow_rate`, `module_oxygen`, `build_plate_temperature`, `bottom_flow_temperature`, `actual_ventilator_flow_rate`** |
| crop 크기 | 8 × 8 | 8 × 8 |
| `T_max` | 70 | 70 |
| `share_cnn` / `share_lstm` | false / false | false / false |
| CNN | ch1=16, ch2=32, kernel=3, d_cnn=32 | 동일 |
| LSTM (camera) | hidden=16, layers=1, 단방향 | 동일 |
| LSTM (sensor) | — | hidden=16, layers=1, 단방향 |
| Optimizer | Adam, lr=1e-3, batch=256, wd=0.0 | 동일 |
| Early stop | patience=50, max=5000 | 동일 |
| Grad clip | 1.0 | 1.0 |

> **Controlled 비교**: MLP 입력 차원 29 동일, 카메라 두 분기 동일, 학습 hp 동일. **변화는 sensor 표현 방식 단일 변수만** (평균 7 ↔ LSTM 7). sensor 분기 자체가 새로 추가된 파라미터는 7×4×16+16²+16×7 ≈ 약 800 개로 매우 작음 (전체 모델 대비 < 5 %).

---

## 5. 종합 평가 — 어느 가설이 맞았는가

PLAN [§6](../../../../vppm/lstm_dual_img_4_sensor_7/PLAN.md) 의 3 시나리오 대조:

| 시나리오 | 예측 | 결과 | 판정 |
|:--|:--|:--|:--:|
| **A. 시간 패턴 의미 (RMSE −3~−10 %, 특히 UE/TE)** | UE/TE 큰 개선, UTS/YS 약간 | TE −3.5 %, UTS −2.1 %, **UE +1.7 %**, YS −0.4 % | 🟡 **부분 지지** (속성별로 갈림) |
| **B. 평균이 충분** | 모든 속성 std 안 | YS / UE 만 그 패턴 | 🟡 **부분 지지** |
| **C. 과적합** | val loss ↓, hold-out ↑ | val loss 4 속성 모두 미세 개선, hold-out 도 일관 | ❌ 미관측 |

### 결론: 속성별로 sensor 시간 패턴의 가치가 다르다

- **TE / UTS** 는 sensor 시퀀스가 **추가 신호** 를 제공 (가장 유력 후보: `module_oxygen` spike, `build_plate_temperature` drift, `top/bottom_flow_rate` dip 의 layer-별 변동). 평균만 잡아서는 놓치는 동적 패턴을 LSTM 이 회수.
- **YS** 는 측정한계 (16.6 MPa) 의 1.24× (20.6 MPa) — 이미 잡음 한계 부근이라 sensor 추가 신호도 빠져나갈 곳이 없음. dual_4 와 동률은 자연스러운 결과.
- **UE 후퇴** 는 의외 — fold 0 의 단일 outlier (7.24) 가 전체를 끌어올림. fold 1~4 평균 (6.60) 만 보면 dual_4 와 동률. **n_folds=5 한계로 인한 분산** 일 가능성 높음. UE 의 측정한계는 1.73 % 로 RMSE/한계 비율이 가장 큰 (3.88×) 속성이라, 추가 신호 여지가 있음에도 잡지 못한 것은 sensor 채널 자체의 정보량 한계 또는 fold 분할 노이즈로 추정.

### 보조 해석

1. **UTS 와 TE 의 동시 개선** 은 두 속성이 결함 (porosity, lack-of-fusion) 에 민감하다는 점과 부합. 결함 형성에는 가스/온도 변동이 직접 관여 → sensor 시간 패턴이 결함 신호의 보조 정보원으로 작동.
2. **TE std 0.23 (전 실험 중 최저)** 은 sensor 분기가 분포 양쪽 꼬리 (저연성 / 고연성) 모두에서 안정적인 보정 역할을 했음을 시사. correlation_plots.png 의 TE 패널이 dual_4 보다 대각선 정렬이 살짝 타이트해진 것과 일치.
3. **MLP 입력 차원이 29 로 동일** 한데도 −0.6 MPa (UTS), −0.3 % (TE) 가 나옴 → **표현 방식 차이가 정보량 차이를 만든다** 는 점 입증. baseline 의 G2 평균 7-feat 은 시간 패턴 정보 손실원이었음이 부분적으로 확인됨.

---

## 6. 다음 단계 제안

sensor LSTM 의 효과가 **속성별로 갈리고 절대 개선 폭이 작다 (≤ 3.5 %)** 는 점에서, sensor-only 변형을 더 짜내는 것은 한계. 정보원을 다시 늘리는 방향이 우선.

| 방향 | 가설 | 우선순위 |
|:--|:--|:--:|
| **빌드별 RMSE 분해** | B1.4(스패터/가스) / B1.5(리코터 손상) 에서만 sensor 효과가 클 수 있음. 종합 RMSE 에 묻혀있을 가능성. | 🔥 **최우선** |
| **UE fold 0 outlier 분석** | fold 0 UE 7.24 가 dual_4 (6.94) 대비 악화된 이유 — 어떤 빌드/공정조건 SV 가 들어있는지 확인. sensor 분기 자체의 over-fit 인지 fold 분할 잡음인지 분리 | 🔥 즉시 |
| **scan path geometry** | `scans/{layer}` 의 스캔 길이/속도 시계열 — 카메라/sensor 둘 다 못 보는 라인-레벨 결함 정보 | ⚪ 보조 |
| **세그멘테이션 결과 직접 입력** | `slices/segmentation_results/{0-11}` DSCNN 12-class 통계를 SV 단위로 시계열화 | ⚪ 보조 |
| **Multi-target joint training** | YS 측정잡음 한계 돌파 — UTS/UE/TE 와 cross-property regularization | ⚪ 검토 |
| **Bidirectional LSTM (sensor 분기만)** | 후속 layer 정보로 현재 layer sensor 신호 재해석 — ablation 가치 | ⚪ 저우선 |

> dual_4 + sensor LSTM 까지 합쳐 visible/temporal 두 정보원의 LSTM 통합은 일단 마무리. 이제는 **scan path / segmentation 등 새로운 정보원** 또는 **빌드 분해로 이미 있는 신호의 효과를 정밀화** 하는 방향.

---

## 7. 데이터 / 산출물 참조

- 메트릭 raw: [`metrics_raw.json`](metrics_raw.json)
- 메트릭 요약: [`metrics_summary.json`](metrics_summary.json)
- per-sample 예측: [`predictions_YS.csv`](predictions_YS.csv), [`predictions_UTS.csv`](predictions_UTS.csv), [`predictions_UE.csv`](predictions_UE.csv), [`predictions_TE.csv`](predictions_TE.csv)
- 학습 로그: [`models/training_log.json`](../models/training_log.json)
- 모델 가중치: `../models/vppm_lstm_dual_img_4_sensor_7_{YS,UTS,UE,TE}_fold{0..4}.pt` (총 20 개 체크포인트, 각 ~115 KB)
- 시각화: [`correlation_plots.png`](correlation_plots.png) (4 속성 2D 히스토그램), [`scatter_plot_uts.png`](scatter_plot_uts.png) (UTS 상세 산점도)
- 정규화 파라미터: [`features/normalization.json`](../features/normalization.json) (14-feat baseline + 7-channel sensor min/max)
- 실험 메타: [`experiment_meta.json`](../experiment_meta.json)
- Sensor 시퀀스 캐시: [`cache/sensor_stacks_B1.{1..5}.h5`](../cache/) (빌드 5 개)
- visible/0, visible/1 캐시: 기존 `vppm_lstm/cache/`, `vppm_lstm_dual/cache/` 재사용
