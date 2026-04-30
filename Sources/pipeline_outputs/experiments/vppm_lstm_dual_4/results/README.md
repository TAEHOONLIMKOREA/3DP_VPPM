# VPPM-LSTM-Dual-4 (`d_embed_v0=v1=4`) 결과 해석

> 21 baseline 피처 + **visible/0 per-SV CNN+LSTM 4-dim 임베딩** + **visible/1 per-SV CNN+LSTM 4-dim 임베딩** = **29-feat MLP** 의 학습/평가 결과 정리.
>
> **결과 위치**: [`Sources/pipeline_outputs/experiments/vppm_lstm_dual_4/results/`](.)
> **학습 일시**: 2026-04-30 (15:20 ~ 19:01)
> **모델 코드**: [`Sources/vppm/lstm_dual_4/`](../../../../vppm/lstm_dual_4/)
> **실험 계획서**: [`Sources/vppm/lstm_dual_4/PLAN.md`](../../../../vppm/lstm_dual_4/PLAN.md)
> **참고 (dual d=1)**: [`vppm_lstm_dual/README.md`](../../vppm_lstm_dual/README.md)
> **참고 (단일 채널)**: [`vppm_lstm/LSTM_RESULTS.md`](../../vppm_lstm/LSTM_RESULTS.md)

---

## 1. 핵심 결과 — projection 통로 4 배 확장도 **추가 개선 없음**

| 속성 | baseline (21) | LSTM single (22, d=1) | LSTM dual (23, d=1+1) | **LSTM dual_4 (29, d=4+4)** | naive |
|:--:|:--:|:--:|:--:|:--:|:--:|
| YS  | 24.3 ± 0.8 MPa | 20.9 ± 0.7 MPa | 20.8 ± 1.0 MPa | **20.7 ± 1.0 MPa** | 33.9 MPa |
| UTS | 42.9 ± 2.0 MPa | 29.5 ± 1.3 MPa | 29.6 ± 1.0 MPa | **29.7 ± 0.9 MPa** | 68.4 MPa |
| UE  | 9.3 ± 0.3 %    | 6.5 ± 0.3 %    | 6.6 ± 0.2 %    | **6.6 ± 0.2 %**    | 15.0 % |
| TE  | 11.3 ± 0.5 %   | 8.4 ± 0.2 %    | 8.4 ± 0.3 %    | **8.5 ± 0.3 %**    | 18.5 % |

> 모든 속성에서 dual (d=1+1) 대비 차이가 std 범위 안. PLAN 의 가설 **A (projection 병목)** 는 **반증**, **B (정보 중복)** 가설이 강하게 지지됨.

| 속성 | naive 대비 RMSE 감소 (dual_4) |
|:--:|:--:|
| YS  | **39 %** ↓ |
| UTS | **57 %** ↓ |
| UE  | **56 %** ↓ |
| TE  | **54 %** ↓ |

→ baseline → LSTM 단계의 큰 도약(YS −15 %p, UTS −37 %p, UE −24 %p, TE −22 %p)은 LSTM 시계열 정보가 만들어낸 것이고, **시점·통로폭을 더 늘려도 추가 이득은 사라진다**.

---

## 2. Fold 별 RMSE 상세

[`metrics_raw.json`](metrics_raw.json) 기준.

| 속성 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 평균 ± std |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| YS  | 20.14 | 19.41 | 22.09 | 20.18 | 21.47 | **20.66 ± 0.98** |
| UTS | 28.68 | 28.71 | 30.24 | 30.26 | 30.80 | **29.74 ± 0.87** |
| UE  | 6.94  | 6.43  | 6.33  | 6.65  | 6.72  | **6.61 ± 0.22**  |
| TE  | 8.94  | 8.40  | 8.11  | 8.38  | 8.48  | **8.46 ± 0.27**  |

- 평가 샘플 수: **6,373**
- YS fold 2 (22.09) 가 다른 fold 대비 살짝 튐 — `vppm_lstm` / `vppm_lstm_dual` 에서도 같은 fold 가 highest 였음 → **데이터 분할 자체의 특성**이지 모델 capacity 문제 아님.
- UTS 는 fold 0/1 (28.7) vs fold 2~4 (30.2~30.8) 로 두 그룹. 이전 dual 에서는 fold 4 만 31.7 로 튀었으나, dual_4 에서는 fold 2~4 가 다 같이 30 대로 평탄해져 std (1.02 → 0.87) 가 약간 개선.
- UE / TE 는 모든 fold std/mean ≤ 4 % 로 매우 안정적.

---

## 3. 학습 안정성 — val loss 미세 개선, 수렴 속도는 속성별로 엇갈림

[`models/training_log.json`](../models/training_log.json) 기준.

| 속성 | val loss (정규화) | 평균 epoch | dual 대비 val loss | dual 대비 epoch |
|:--:|:--:|:--:|:--:|:--:|
| YS  | 0.0791 | 273 | 0.0802 → **0.0791** (-1.4 %) | 232 → 273 (+18 %) |
| UTS | 0.0623 | 243 | 0.0631 → **0.0623** (-1.3 %) | 247 → 243 (-2 %) |
| UE  | 0.1178 | 242 | 0.1186 → **0.1178** (-0.7 %) | 312 → 242 (-22 %) |
| TE  | 0.1257 | 267 | 0.1264 → **0.1257** (-0.6 %) | 272 → 267 (-2 %) |

- **val loss 는 모든 속성에서 약간 더 낮음** (≤ 1.4 %). 통로폭이 늘어 학습 단계에서 살짝 더 fit 되긴 함.
- **그러나 hold-out RMSE 에는 그 차이가 살아남지 않음** — 즉 추가 capacity 가 일반화로 이어지지 못하고 train/val 양쪽에서만 미세하게 흡수됨.
- 수렴 속도는 속성별로 다름: UE 는 -22 % 로 빨라졌으나 YS 는 +18 % 로 오히려 늦어짐. 일관된 패턴 없음.
- 과적합 징후 (시나리오 C) 는 미관측. 6,373 샘플 / 29 feat 으로는 capacity 충분.

---

## 4. 모델 / 학습 설정 ([`experiment_meta.json`](../experiment_meta.json))

| 항목 | 값 (dual) | **값 (dual_4)** |
|:--|:--:|:--:|
| baseline 피처 수 | 21 | 21 |
| `d_embed_v0` (visible/0 임베딩 차원) | 1 | **4** |
| `d_embed_v1` (visible/1 임베딩 차원) | 1 | **4** |
| 카메라 정보 통로폭 | 1 + 1 = 2 | 4 + 4 = **8** |
| 총 입력 피처 (`n_total_feats`) | 23 | **29** |
| 카메라 채널 | `[0, 1]` | `[0, 1]` |
| crop 크기 | 8 × 8 | 8 × 8 |
| `T_max` (레이어 수) | 70 | 70 |
| `share_cnn` / `share_lstm` | false / false | false / false |
| CNN | ch1=16, ch2=32, kernel=3, d_cnn=32 | 동일 |
| LSTM | hidden=16, layers=1, 단방향 | 동일 |
| Optimizer | Adam, lr=1e-3, batch=256, wd=0.0 | 동일 |
| Early stop | patience=50, max=5000 | 동일 |
| Grad clip | 1.0 | 1.0 |

> **변경된 곳은 단 한 곳: 채널별 `Linear(16 → d_embed)` projection 의 출력 차원 1 → 4** (`Sources/vppm/lstm_dual/model.py` `_LSTMBranch.proj` L41 와 동일 구조). CNN/LSTM 내부 capacity 와 학습 hp 는 모두 고정 — 통로폭 효과만 분리됨.

---

## 5. 종합 평가 — 어느 가설이 맞았는가

PLAN 의 3 시나리오 ([`PLAN.md` §4](../../../../vppm/lstm_dual_4/PLAN.md)) 와 대조:

| 시나리오 | 예측 | 결과 | 판정 |
|:--|:--|:--|:--:|
| **A. 통로 병목** | RMSE -5~-15 % 추가 개선 | 모든 속성 차이 std 안 (≤ 0.1) | ❌ **반증** |
| **B. 정보 중복** | RMSE 차이 std 범위 안 | 모든 속성에서 정확히 그 패턴 | ✅ **지지** |
| **C. 과적합 시작** | val loss 정체 / 상승, train↔val 격차 확대 | val loss 오히려 미세 개선, 격차 증대 없음 | ❌ 미관측 |

### 결론: visible/1 추가 정보의 한계가 본질

dual (d=1) 에서 평탄했던 이유가 "통로가 좁아서" 가 아니라 **visible/1 이 visible/0 대비 추가 신호를 거의 갖고 있지 않기 때문**임이 확정.

뒷받침 근거:
1. **통로폭을 8 배 늘려도 (2 → 8 dim) RMSE 가 변하지 않음** — 정보 자체가 없으면 통로를 넓혀도 들어올 게 없음.
2. **val loss 는 미세하게 줄지만 (-0.6 ~ -1.4 %) hold-out RMSE 에 반영되지 않음** — 추가된 4-dim 표현은 train 분포 안에서만 미세하게 fit 되고, 일반화에 기여 못함.
3. **CNN/LSTM 둘 다 채널별 독립 (`share_*=false`) 이라 정보 융합의 자유도가 충분함에도** 결과가 평탄.

### 보조 해석

- 분말 도포 후 (visible/1) 결함은 후속 용융 (visible/0) 에 흔적을 남기므로 visible/0 만으로 대부분 포착되는 구조. 두 시점이 본질적으로 같은 결함 신호의 다른 노출.
- YS RMSE 20.7 MPa 는 측정오차 16.6 MPa 의 **1.25×** — 측정 잡음 한계에 거의 도달. 통로 확장으로도 더 줄지 않음 (애초에 모델이 찾을 수 있는 신호가 잡음 한계에 가까움).
- UTS / UE / TE 는 측정오차 대비 1.89× / 3.76× / 2.88× 로 여유는 있으나, **현재 입력 (시계열 카메라 + baseline 21)** 으로는 추가로 빼낼 정보가 visible/* 채널에는 없음.

---

## 6. 다음 단계 제안

dual_4 가 시나리오 B 를 확정한 이상, **visible/0/1 채널 추가 압축이 아닌 다른 정보원** 을 들여야 함.

| 방향 | 가설 | 우선순위 |
|:--|:--|:--:|
| **빌드별 RMSE 분해** | B1.4(스패터)/B1.5(리코터 손상)에서만 visible/1 효과가 있을 수 있음. 종합 RMSE 에 묻혀있을 가능성 잔존. | 🔥 최우선 |
| **온도/산소/유량 등 temporal 센서 통합** | per-layer scalar 시계열 (`temporal/*`) 은 아직 baseline 21 feat 의 통계량으로만 들어있음. LSTM 으로 직접 시퀀스 입력 | 🔥 즉시 |
| **scan path geometry** | `scans/{layer}` 의 스캔 길이/속도 분포 — 카메라가 못 보는 라인-레벨 결함 정보 | ⚪ 보조 |
| **Multi-target joint training** | YS 측정잡음 한계 돌파 — UTS/UE/TE 와 cross-property regularization | ⚪ 보조 |
| **세그멘테이션 결과 직접 입력** | `slices/segmentation_results/{0-11}` 의 DSCNN 12-class 통계를 SV 단위로 시계열화 | ⚪ 검토 |

> visible-only 채널을 더 짜내는 것은 **명백히 수렴**. 다음 실험은 정보원 자체를 늘리는 방향으로 가야 함. PLAN 에 이미 존재하는 `vppm_lstm_dual_img_4_sensor_7` 실험이 그 첫 시도로 보임 (`Sources/vppm/lstm_dual_img_4_sensor_7/`).

---

## 7. 데이터 / 산출물 참조

- 메트릭 raw: [`metrics_raw.json`](metrics_raw.json)
- 메트릭 요약: [`metrics_summary.json`](metrics_summary.json)
- per-sample 예측: [`predictions_YS.csv`](predictions_YS.csv), [`predictions_UTS.csv`](predictions_UTS.csv), [`predictions_UE.csv`](predictions_UE.csv), [`predictions_TE.csv`](predictions_TE.csv)
- 학습 로그: [`models/training_log.json`](../models/training_log.json)
- 모델 가중치: `../models/vppm_lstm_dual_4_{YS,UTS,UE,TE}_fold{0..4}.pt` (총 20 개 체크포인트, 각 ~104 KB)
- 시각화: [`correlation_plots.png`](correlation_plots.png) (4 속성 2D 히스토그램), [`scatter_plot_uts.png`](scatter_plot_uts.png) (UTS 상세 산점도)
- 정규화 파라미터: [`features/normalization.json`](../features/normalization.json)
- 실험 메타: [`experiment_meta.json`](../experiment_meta.json)
- visible/0, visible/1 캐시: 기존 `vppm_lstm/cache/`, `vppm_lstm_dual/cache/` 재사용
