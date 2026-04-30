# VPPM-LSTM-Dual (d_embed=4) 실험 계획

> **한 줄 요약**: LSTM 의 16-dim 출력 → Linear 로 1-dim 으로 짜내던 **projection 통로를 16 → 4 로 넓힌다**. CNN/LSTM 내부 구조와 학습 hp 는 그대로 두고 "나가는 통로"만 확장하는 단일 변수 실험.

- **결과 위치 (예정)**: `Sources/pipeline_outputs/experiments/vppm_lstm_dual_4/`
- **비교 대상**: `vppm_lstm` (단일 채널, 22-feat), `vppm_lstm_dual` (dual, 23-feat)
- **학습 일시**: 2026-04-30 ~

---

## 1. 데이터 흐름 — 어디가 병목인가

채널 한 개 분기의 흐름 (`Sources/vppm/lstm_dual/model.py`):

```
stack (B, T=70, 8, 8)
  ──[FrameCNN]──>        (B, T, 32)        # 8×8 프레임 한 장 → 32-dim
  ──[LSTM(hidden=16)]──> (B, 16)           # 시퀀스 요약 = 마지막 hidden
  ──[Linear(16 → d_embed)]──> (B, d_embed) # ★ 이 projection 이 병목
```

채널별로 위 분기가 하나씩 (v0, v1) → `feats21 ⊕ embed_v0 ⊕ embed_v1` concat → MLP.

**기존 dual (`d_embed=1`)**: LSTM 이 만든 16-dim 요약을 **단 1 차원**으로 짜냄. 두 채널을 합쳐도 카메라 정보의 총 통로폭이 1 + 1 = 2 dim. layer 별 미세 패턴이 살아남기에는 너무 좁다는 가설.

**본 실험 (`d_embed=4`)**: 통로를 16 → 4 로 넓힘. 카메라 정보 총 통로폭 4 + 4 = 8 dim. LSTM/CNN 의 capacity 는 건드리지 않음 — **projection 단계만 4 배 확장**.

---

## 2. 동기 — 기존 dual 결과의 평탄성

`vppm_lstm_dual` 결과 요약:

| 속성 | lstm (단일, d_embed=1) | lstm_dual (d_embed=1+1) | 차이 |
|:--:|:--:|:--:|:--:|
| YS  | 20.9 ± 0.7 | 20.8 ± 1.0 | ~0 |
| UTS | 29.5 ± 1.3 | 29.6 ± 1.0 | ~0 |
| UE  | 6.5 ± 0.3  | 6.6 ± 0.2  | ~0 |
| TE  | 8.4 ± 0.2  | 8.4 ± 0.3  | ~0 |

dual 로 채널을 늘렸는데도 단일 채널과 동일. 가능한 원인 두 가지:

1. **두 채널이 본질적으로 같은 정보** — visible/1 이 visible/0 대비 추가 정보 거의 없음.
2. **projection 통로가 너무 좁아 정보가 병목** — 16-dim LSTM 출력을 1-dim 으로 짜내면 layer 별 미세 패턴이 손실. 두 번째 채널을 추가해도 똑같이 1-dim 으로 짜내니 의미가 없음.

본 실험은 **(2) 를 검증** 하기 위해 d_embed_v0 = d_embed_v1 = **4** 로 확장. 성능이 의미 있게 개선되면 dual 의 평탄 결과는 **projection capacity 문제**. 개선이 없으면 **(1) 정보 중복** 가설이 강해짐.

---

## 3. 변경 사항 (vppm_lstm_dual 대비)

| 항목 | dual (기존) | **dual_4 (신규)** |
|:--:|:--:|:--:|
| LSTM hidden 출력 | 16 | 16 (유지) |
| `d_embed_v0` (proj 출력) | **1** | **4** |
| `d_embed_v1` (proj 출력) | **1** | **4** |
| 카메라 정보 통로폭 | 1 + 1 = **2** | 4 + 4 = **8** |
| `n_total_feats` (MLP 입력) | 21 + 1 + 1 = 23 | 21 + 4 + 4 = **29** |
| `share_cnn` / `share_lstm` | False / False | False / False (유지) |
| `cnn` `{ch1, ch2, kernel, d_cnn}` | {16, 32, 3, 32} | 동일 |
| `lstm` `{d_hidden, num_layers, bidir}` | {16, 1, False} | 동일 |
| `T_max`, `crop_h/w` | 70, 8, 8 | 동일 |
| 학습 hp (lr/batch/patience) | 1e-3 / 256 / 50 | 동일 |

> **단일 변수 통제**: `proj` 의 출력 차원만 4 배 확장. CNN/LSTM 내부, 학습 hp 모두 고정해 "통로폭" 효과만 분리.

---

## 4. 가설별 기대치

| 시나리오 | 예상 결과 | 해석 |
|:--|:--|:--|
| **A. 통로 병목이 원인** | UTS / UE / TE 에서 RMSE -5~-15% 추가 개선 | LSTM 이 만든 정보는 이미 풍부. 1-dim 으로 짜내는 게 손실원이었음. d_embed=8 추가 실험 가치 있음. |
| **B. 정보 중복** | RMSE 차이 std 범위 안 (≤ 0.1) | visible/1 이 사실상 visible/0 의 약간 다른 시점 스냅샷. 통로 넓혀도 새 정보 없음. |
| **C. 과적합 시작** | val loss 정체 / 상승, train↔val 격차 확대 | 23 → 29 feat 늘었으나 6,373 샘플로는 충분. 가능성 낮음. |

특히 **B1.4 (스패터)** / **B1.5 (리코터 손상)** 빌드의 **빌드별 RMSE 분해** 가 핵심 진단:
- 분말 도포 후 결함이 직접 보이는 빌드에서만 개선되면 → 정보는 있었으나 통로가 좁았던 것 (시나리오 A).
- 모든 빌드에서 평탄하면 → visible/1 정보량 자체가 한계 (시나리오 B).

---

## 5. 측정오차 한계와의 관계

기존 `vppm_lstm` 시점에서 YS 는 측정오차의 1.26× 까지 좁혀졌음. 통로 확장으로 **YS 가 측정오차 한계 (16.6 MPa)** 에 더 근접하는지 관찰. UTS (1.89×), UE (3.76×), TE (2.88×) 는 여전히 여유 있어 가장 큰 개선 여지가 있는 구간.

---

## 6. 실행 체크리스트

- [ ] `experiment_meta.json` 작성 — `d_embed_v0=4`, `d_embed_v1=4`, `n_total_feats=29`
- [ ] feature 캐시 빌드 (proj 출력 4-dim × 2 채널)
- [ ] 4 속성 × 5-fold 학습 (예상 ~3시간, 기존 dual 과 동일 epoch 규모 가정)
- [ ] `metrics_raw.json` / `metrics_summary.json` / `predictions_{YS,UTS,UE,TE}.csv` 생성
- [ ] correlation_plots.png, scatter_plot_uts.png
- [ ] 빌드별 RMSE 분해 (baseline / lstm / lstm_dual / lstm_dual_4 4-way)
- [ ] 결과 해석 문서 (`RESULTS.md`) 작성 — 본 PLAN 의 가설 A/B/C 중 어느 쪽인지 결론

---

## 7. 참조

- 모델 코드: `Sources/vppm/lstm_dual/model.py` (proj: `_LSTMBranch.proj`, L41)
- 기존 dual 결과: `experiments/vppm_lstm_dual/results/`
- 기존 단일 결과: `experiments/vppm_lstm/results/`
- 결과 해석: `experiments/vppm_lstm/LSTM_RESULTS.md`
