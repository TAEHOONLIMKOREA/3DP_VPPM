# VPPM-LSTM-Dual (visible/0 + visible/1) 결과 해석

> 21 baseline 피처 + **visible/0 (용융 직후) per-SV CNN+LSTM 1-dim 임베딩** + **visible/1 (분말 도포 직후) per-SV CNN+LSTM 1-dim 임베딩** = **23-feat MLP** 의 학습/평가 결과 정리.
>
> **결과 위치**: `Sources/pipeline_outputs/experiments/vppm_lstm_dual/results/`
> **학습 일시**: 2026-04-30
> **모델 코드**: [`Sources/vppm/lstm_dual/`](../../../vppm/lstm_dual/)
> **참고 (단일 채널 결과)**: [`vppm_lstm/LSTM_RESULTS.md`](../vppm_lstm/LSTM_RESULTS.md)

---

## 1. 핵심 결과 — 단일 LSTM 대비 **추가 개선 사실상 없음**

| 속성 | baseline (21 feat) | **LSTM single (22 feat)** | **LSTM dual (23 feat)** | naive |
|:--:|:--:|:--:|:--:|:--:|
| YS  | 24.3 ± 0.8 MPa | **20.9 ± 0.7 MPa** | 20.8 ± 1.0 MPa | 33.9 MPa |
| UTS | 42.9 ± 2.0 MPa | **29.5 ± 1.3 MPa** | 29.6 ± 1.0 MPa | 68.4 MPa |
| UE  | 9.3 ± 0.3 %    | **6.5 ± 0.3 %**    | 6.6 ± 0.2 %    | 15.0 %   |
| TE  | 11.3 ± 0.5 %   | **8.4 ± 0.2 %**    | 8.4 ± 0.3 %    | 18.5 %   |

> **모든 속성에서 single 대비 차이가 std 범위 안**. PLAN 가설 "1–3 MPa 추가 개선" 은 **달성하지 못함**. visible/1 추가 임베딩은 RMSE 를 의미 있게 줄이지 못함.

| 속성 | naive 대비 RMSE 감소 (dual) |
|:--:|:--:|
| YS  | **39%** ↓ |
| UTS | **57%** ↓ |
| UE  | **56%** ↓ |
| TE  | **55%** ↓ |

→ baseline 대비 큰 개선은 LSTM 자체의 효과이고, dual 변환이 추가로 기여하는 부분은 미미.

---

## 2. Fold 별 RMSE 상세

| 속성 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 평균 ± std |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| YS  | 20.32 | 19.65 | 22.01 | 20.10 | 21.82 | **20.78 ± 0.95** |
| UTS | 29.17 | 29.08 | 29.03 | 29.23 | 31.67 | **29.64 ± 1.02** |
| UE  | 6.90  | 6.61  | 6.30  | 6.62  | 6.61  | **6.61 ± 0.19**  |
| TE  | 8.96  | 8.35  | 7.91  | 8.23  | 8.44  | **8.38 ± 0.34**  |

- single 과 동일하게 **UTS fold 4 만 31.7 MPa 로 살짝 튐** (다른 fold 는 29 근처). visible/1 추가로도 이 fold 의 outlier sample 이 해소되지 않음 — 데이터 분할 자체의 특성으로 보임.
- 그 외 fold 분포는 안정적 (std/mean < 5%).

---

## 3. 학습 안정성 — single 과 거의 동일, 약간 더 빠른 수렴

| 속성 | val loss (정규화, dual) | val loss (single) | 평균 epoch (dual) | 평균 epoch (single) |
|:--:|:--:|:--:|:--:|:--:|
| YS  | 0.0802 | 0.0796 | **232** | 257 |
| UTS | 0.0631 | 0.0620 | **247** | 322 |
| UE  | 0.1186 | 0.1179 | 312 | 312 |
| TE  | 0.1264 | 0.1261 | **272** | 305 |

- **val loss 는 단일 채널과 거의 같음** (모든 속성에서 차이 < 0.001).
- **수렴은 dual 에서 일관되게 약간 빠름** (UTS 322 → 247 epoch, -23%). 추가 1-dim 임베딩이 정보를 조금 더 컴팩트하게 압축해 MLP 학습이 빨라지지만, 최종 정확도는 동일.
- fold 간 변동성도 single 과 비슷한 수준 (std/mean ≈ 2~5%).

---

## 4. 모델 / 학습 설정 (`experiment_meta.json`)

| 항목 | 값 |
|:--|:--|
| baseline 피처 수 | 21 |
| visible/0 임베딩 차원 (`d_embed_v0`) | 1 |
| visible/1 임베딩 차원 (`d_embed_v1`) | 1 |
| 총 입력 피처 | 23 |
| 카메라 채널 | `[0, 1]` (visible/0 + visible/1) |
| crop 크기 | 8 × 8 |
| `T_max` (레이어 수) | 70 |
| `share_cnn` / `share_lstm` | **false** / **false** (채널별 독립) |
| CNN | ch1=16, ch2=32, kernel=3, d_cnn=32 |
| LSTM | hidden=16, layers=1, **단방향** |
| Optimizer | Adam, lr=1e-3, batch=256, wd=0.0 |
| Early stop | patience=50, max=5000 epoch |
| Grad clip | 1.0 |

> 두 채널이 **CNN/LSTM weight 를 공유하지 않음**. 따라서 visible/0 과 visible/1 각각의 시공간 특성을 자유롭게 학습할 수 있는 구조였음에도, 최종 RMSE 가 single 과 같은 수준이라는 점은 **두 채널의 정보 중복** 또는 **MLP 가 흡수할 수 있는 정보의 포화** 를 시사.

---

## 5. 종합 평가 — 왜 효과가 없는가

[`LSTM_RESULTS.md`](../vppm_lstm/LSTM_RESULTS.md) 의 두 시나리오 중 **시나리오 1 (정보 중복)** 가설이 강하게 지지됨.

1. **시나리오 1: 정보 중복** ✅ 성립
   visible/1 (분말 도포 직후) 이 visible/0 (용융 직후) 와 비슷한 결함 신호를 담고 있을 가능성. 분말 단계 결함이 후속 용융에도 흔적을 남기므로 visible/0 단독으로 이미 대부분 포착.

2. **시나리오 2: 특정 빌드에서 큰 개선** ❌ 전체 RMSE 로는 미관측
   B1.4 (스패터) / B1.5 (리코터 손상) 빌드에서 visible/1 이 분말 단계 결함을 직접 캡쳐할 거라는 기대였으나, **종합 RMSE 에 드러날 만큼 개선되지는 않음**.
   → 빌드별 RMSE 분해 해보면 특정 빌드에서만 개선이 있을 가능성은 남아있음 (현재 결과에는 분해되어 있지 않음).

### 그 외 보조 해석

- **MLP 의 정보 흡수 한계**: 23-feat → 4-prop MLP 의 모델 용량이 크지 않아, 추가 1-dim feature 가 들어와도 가중치 재배치로 흡수되어 RMSE 변화가 없을 가능성.
- **임베딩 차원이 너무 작음**: `d_embed_v1 = 1` 은 매우 압축된 표현. visible/1 이 가진 정보가 1-dim 으로 충분히 표현되지 않으면 추가 채널의 효과가 사라짐.
- **YS 측정오차 한계**: YS RMSE 20.8 MPa 는 측정오차 16.6 MPa 의 1.25× — 측정 잡음 한계에 거의 도달. 더 줄이려면 다른 방향 (multi-target learning, label denoising 등) 이 필요.

---

## 6. 다음 단계 제안

| 방향 | 가설 | 우선순위 |
|:--|:--|:--:|
| **빌드별 RMSE 분해** | B1.4/B1.5 에서만 visible/1 효과가 있을 수 있음 | 🔥 즉시 |
| `d_embed_v1` 확장 (1 → 4 또는 8) | 1-dim 압축이 너무 강해 visible/1 정보 손실 가능 | 🔥 즉시 |
| `share_cnn=true` 실험 | 채널 간 공유로 일반화 향상 가능성 | ⚪ 보조 |
| Mid-fusion (LSTM 입력 단계에서 v0/v1 concat) | late-fusion 보다 시간축 상호작용 학습 가능 | ⚪ 보조 |
| Multi-target joint training (YS/UTS/UE/TE 한 모델) | 측정 잡음을 cross-property regularization 으로 완화 | ⚪ 추가 검토 |

> [`vppm_lstm_dual_4`](../vppm_lstm_dual_4/) 는 `d_embed_*` 4-dim 확장 실험으로 보임 (`PLAN.md` 참조).

---

## 7. 데이터 / 산출물 참조

- 메트릭 raw: [`results/metrics_raw.json`](results/metrics_raw.json)
- 메트릭 요약: [`results/metrics_summary.json`](results/metrics_summary.json)
- per-sample 예측: `results/predictions_{YS,UTS,UE,TE}.csv`
- 학습 로그: [`models/training_log.json`](models/training_log.json)
- 모델 가중치: `models/vppm_lstm_dual_{YS,UTS,UE,TE}_fold{0..4}.pt`
- 시각화: [`results/correlation_plots.png`](results/correlation_plots.png), [`results/scatter_plot_uts.png`](results/scatter_plot_uts.png)
- 정규화 파라미터: [`features/normalization.json`](features/normalization.json)
- 실험 메타: [`experiment_meta.json`](experiment_meta.json)
- visible/0 캐시: `vppm_lstm/cache/crop_stacks_B1.x.h5` (재사용)
- visible/1 캐시: [`cache/crop_stacks_v1_B1.x.h5`](cache/)
