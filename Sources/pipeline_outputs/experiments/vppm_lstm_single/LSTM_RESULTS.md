# VPPM-LSTM (visible/0) 결과 해석

> `vppm_lstm_dual` (visible/0 + visible/1, 23-feat) 실험을 진행하기 전에 참고하기 위한 **기존 `vppm_lstm` (visible/0 단일 채널, 22-feat) 학습 결과 분석**.
>
> **결과 위치**: `Sources/pipeline_outputs/experiments/vppm_lstm/results/`
> **학습 일시**: 2026-04-29 ~ 04-30
> **모델 설명**: [`Sources/vppm/lstm/MODEL.md`](../lstm/MODEL.md)

---

## 1. 핵심 결과 — baseline 대비 압도적 개선

| 속성 | baseline (21 feat) | **LSTM (22 feat)** | 개선 | naive 대비 |
|:--:|:--:|:--:|:--:|:--:|
| YS  | 28.7 ± 0.6 MPa | **20.9 ± 0.7 MPa** | -27% | **38%** ↓ |
| UTS | 60.7 ± 2.6 MPa | **29.5 ± 1.3 MPa** | **-51%** | **57%** ↓ |
| UE  | 12.8 ± 0.3 %   | **6.5 ± 0.3 %**   | -49% | **56%** ↓ |
| TE  | 15.5 ± 0.2 %   | **8.4 ± 0.2 %**   | -46% | **55%** ↓ |

> PLAN 가설 ("lstm 보다 1–3 MPa 추가 개선 기대") 을 **훌쩍 넘는 개선**. 1-dim 카메라 임베딩 하나 추가가 모든 속성에서 RMSE 를 거의 절반으로 줄임.

---

## 2. 측정오차 대비 모델 능력

| 속성 | LSTM RMSE | 측정오차 | 비율 |
|:--:|:--:|:--:|:--:|
| YS  | 20.9 MPa | 16.6 MPa | **1.26 ×** |
| UTS | 29.5 MPa | 15.6 MPa | 1.89 × |
| UE  | 6.5 %    | 1.73 %   | 3.76 × |
| TE  | 8.4 %    | 2.92 %   | 2.88 × |

YS 는 거의 측정오차 한계 근처까지 좁혀짐. UTS / 연성 계열은 아직 여유 있음 — visible/1 추가 (`lstm_dual`) 로 더 줄일 여지.

---

## 3. 학습 안정성 — 빠르고 일관됨

| 속성 | val loss (정규화) | 평균 epoch | fold std/mean |
|:--:|:--:|:--:|:--:|
| YS  | 0.080 (baseline 0.099, **-19%**) | **257** (baseline 454) | 3.6% |
| UTS | 0.062 (baseline 0.101, **-39%**) | 322 (baseline 601) | 4.5% |
| UE  | 0.118 (baseline 0.197, **-40%**) | 312 (baseline 1,016) | 4.0% |
| TE  | 0.126 (baseline 0.211, **-40%**) | 306 (baseline 928)   | 2.8% |

- **수렴 속도가 baseline 의 1/3 ~ 1/2** — early-stop 이 더 일찍 발동. CNN+LSTM 임베딩이 정보를 풍부하게 압축해 MLP 학습이 쉬워짐.
- **fold 간 변동성 작음** (std/mean < 5%). UTS 만 fold 4 에서 31.8 MPa 로 상승 (다른 fold 27~30) — 큰 문제는 아니지만 fold 분포에 약간 outlier sample.
- 연성 계열 (UE/TE) val loss 감소 폭이 강도 계열보다 큼 — **시공간 카메라 정보가 결함 (= 연성에 직접 영향) 을 잘 잡음**.

### Fold 별 RMSE 상세

| 속성 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 평균 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| YS  | 20.52 | 19.87 | 21.96 | 20.51 | 21.42 | **20.86** |
| UTS | 28.94 | 27.94 | 28.81 | 29.99 | 31.81 | **29.50** |
| UE  | 7.00  | 6.43  | 6.21  | 6.52  | 6.59  | **6.55**  |
| TE  | 8.75  | 8.35  | 8.03  | 8.27  | 8.47  | **8.37**  |

---

## 4. correlation_plots.png 관찰

- **YS / UTS**: 데이터가 좁은 범위 (각각 350~400 MPa, 550~600 MPa) 에 몰림 — SS 316L 정상 빌드 대부분이 좁은 강도 분포. 그 범위 안에서 대각선 위에 깔끔히 정렬.
- **UE / TE**: 0~70%, 0~90% 전 범위에 걸쳐 대각선 추종. 50~80% 영역 (정상 시편) 이 가장 dense. 좌측 하단 (저연성 = 결함 시편) 도 따라옴 → **모델이 "약한 SV" 를 식별하는 핵심 task 를 잘 수행**.
- 명확한 bias / saturation 없음.

---

## 5. 종합 평가

LSTM 임베딩은 **단순 21 피처가 놓치는 정보** — 즉 `seg_*` 평균값으로는 표현되지 않는 **시간축 패턴 (레이어별 진행) + 공간 패턴 (8×8 국소 텍스처)** — 을 효과적으로 압축하고 있다. 특히:

- UTS 에서 **-51% 개선**은 매우 의미 있는 결과. 논문 재구현치 (60.7 MPa) 를 근본적으로 뛰어넘음.
- 학습이 baseline 대비 더 빠르고 더 안정적 — **정보가 깔끔하다는 신호**.
- 이미 측정오차 한계에 근접하는 속성 (YS) 도 등장.

---

## 6. lstm_dual 진행 시 기대치

기존 PLAN 예상은 "1–3 MPa 추가 개선" 이었지만, visible/0 단독으로 이 정도 효과면 두 가지 시나리오가 가능:

1. **추가 마진 작음 (정보 중복)** — visible/1 이 visible/0 와 비슷한 정보 (분말층 + 용융 흔적) 를 담고 있다면 추가 임베딩의 한계 효용이 작음.
2. **특정 빌드에서 큰 개선** — 분말 도포 후 결함이 직접 보이는 **B1.4 (스패터)** / **B1.5 (리코터 손상)** 에서 추가 개선이 클 수 있음. visible/0 (용융 직후) 에는 이미 가려졌을 수 있는 분말 단계 결함을 visible/1 이 직접 캡쳐.

학습 후 **빌드별 RMSE 분해** 해 baseline / lstm / lstm_dual 3-way 비교하면 효과 출처가 명확해진다.

### 이미 좁아진 RMSE 의 한계

YS 는 측정오차 (16.6 MPa) 의 1.26× 에 도달 — visible/1 추가로 측정오차 한계까지 좁히는 게 가능한지가 관전 포인트. UTS / UE / TE 는 측정오차 대비 여유가 남아있어 더 큰 개선 여지.

---

## 7. 데이터 / 산출물 참조

- 메트릭 raw: `experiments/vppm_lstm/results/metrics_raw.json`
- 메트릭 요약: `experiments/vppm_lstm/results/metrics_summary.json`
- per-sample 예측: `experiments/vppm_lstm/results/predictions_{YS,UTS,UE,TE}.csv`
- 학습 로그: `experiments/vppm_lstm/models/training_log.json`
- 모델 가중치: `experiments/vppm_lstm/models/vppm_lstm_{YS,UTS,UE,TE}_fold{0..4}.pt`
- 시각화: `correlation_plots.png`, `scatter_plot_uts.png`
- 실험 메타: `experiments/vppm_lstm/experiment_meta.json`
