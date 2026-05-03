# E2: No-Cameras (visible/0 + visible/1 양쪽 카메라 LSTM 분기 제거) 실험 계획

> **공통 설정** (가설, CV, 모델, 해석 기준) 은 [PLAN.md](./PLAN.md) 참조.
> 본 문서는 E2 실험 고유 정보만 기술.

---

## 1. 실험 정의

| 항목 | 값 |
|:--|:--|
| **실험 ID** | E2 |
| **실험명** | No-Cameras (visible/0 + visible/1 양쪽 분기 제거) |
| **제거 분기** | `branch_v0` + `branch_v1` (FrameCNN(in=1) + LSTM + proj, d_embed=16 ×2) |
| **유지 분기** | `branch_sensor`, `branch_dscnn`, `branch_cad`, `branch_scan`, `feat_static` |
| **MLP 입력 차원** | 86 → **54** (= 2 + **0** + **0** + 28 + 8 + 8 + 8) |
| **출력 디렉터리** | `Sources/pipeline_outputs/experiments/lstm_ablation/E2_no_cameras/` |
| **flag** | `use_v0=False`, `use_v1=False` |

### 1.1 제거되는 분기의 물리적 의미

| 채널 | HDF5 경로 | 캐시 빌더 | 물리적 의미 |
|:--|:--|:--|:--|
| `visible/0` | `slices/camera_data/visible/0` | [`lstm/crop_stacks.py`](../lstm/crop_stacks.py) | **용융 직후** — 잔열, melt pool, super-elevation, debris |
| `visible/1` | `slices/camera_data/visible/1` | [`lstm_dual/crop_stacks_v1.py`](../lstm_dual/crop_stacks_v1.py) | **분말 도포 직후** — 새 layer 표면, recoater streak, soot |

- 카메라 분기 입력: (B, T≤70, 8, 8) ×2
- 인코더: per-frame CNN(in=1) → LSTM(d_hidden=16) → proj(16) ×2
- 풀-스택에서 차지하는 임베딩 비중: **32 / 86 ≈ 37.2 %** — MLP 입력의 1/3 이상이 카메라

> v0 와 v1 은 **시간적으로 인접** (같은 layer 의 직전/직후) 하므로 redundancy 가능성 높음. 그러나 v0 = 용융 시점 / v1 = 분말 도포 시점이라 표층 결함의 종류가 다를 수 있음.

---

## 2. 가설

> **양쪽 카메라 분기 동시 제거** 시:
> - **RMSE 변화 < 2σ** → 카메라 분기 전체가 풀-스택의 핵심 기여원이 아님. baseline → LSTM 점프의 큰 부분은 sensor / dscnn 의 시간성 처리에서 왔을 가능성. 시나리오 A 결정적 입증
> - **ΔE2 ≈ ΔE1** (E1 결과 대비 추가 악화 작음) → v0/v1 redundant. 둘 중 하나면 충분
> - **ΔE2 ≫ ΔE1** → v1 이 v0 와 독립적인 신호. 양쪽 모두 의미

### 2.1 정량 기대치

| 속성 | E0 (풀-스택) | E2 예상 ΔRMSE | 시나리오별 판정 임계 |
|:--:|:--:|:--:|:--|
| YS  | 20.1 ± 0.9 | +0.0 ~ +1.2 | < 1.0 (≈1σ) → 시나리오 A 결정적 입증 |
| UTS | 28.5 ± 0.8 | +0.0 ~ +1.5 | < 1.0 → 시나리오 A; > 2.0 → D (둘 다 핵심) |
| UE  |  6.5 ± 0.4 | +0.0 ~ +0.5 | < 0.5 → A |
| TE  |  8.1 ± 0.4 | +0.0 ~ +0.6 | < 0.5 → A |

> 풀-스택 vs `vppm_baseline` (21-feat MLP, 평균 처리) 차이: YS −4.2, UTS −14.4, UE −2.9, TE −3.2.
>
> 만약 시나리오 A 가 맞다면 (ΔE2 < 1σ) → 이 큰 점프의 출처가 카메라가 아닌 **sensor 시간성 + dscnn LSTM + cad/scan spatial-CNN** 에 있다는 의미. 매우 중요한 발견.

### 2.2 E1 vs E2 비교 (additivity 검증)

| 관계 | 해석 |
|:--|:--|
| ΔE2 ≈ ΔE1 + ε (ε ≈ 0) | v1 redundant — v0 가 커버한 정보를 다른 분기도 가지고 있음 |
| ΔE2 ≈ 2 × ΔE1 | additive — v0/v1 독립 기여 |
| ΔE2 ≪ ΔE1 + ΔE1 | strong redundancy — 둘 사이 정보 중복 |
| ΔE2 > ΔE1 + ΔE1 | super-additive (드물 — interaction 효과) |

---

## 3. 구현

### 3.1 코드 변경

E1 과 동일 model.py 사용 (분기 토글 flag) — `use_v0=False, use_v1=False` 만 추가.

```python
# Sources/vppm/lstm_ablation/run.py
EXPERIMENTS = {
    "E1": dict(use_v0=False, use_v1=True,  out_subdir="E1_no_v0"),
    "E2": dict(use_v0=False, use_v1=False, out_subdir="E2_no_cameras"),
}
```

모델 구조는 [PLAN_E1_no_v0.md §3.1](./PLAN_E1_no_v0.md#31-코드-변경) 참조 — `n_total = 2 + 0 + 0 + 28 + 8 + 8 + 8 = 54` 자동 계산.

### 3.2 dataset / dataloader

E1 과 동일하게 `load_septet_dataset` 사용. 모델 forward 가 stacks_v0 / stacks_v1 를 모두 무시.

**메모리 절감 옵션**: E2 에서는 v0/v1 캐시 (~150-200 MB/build × 2 = ~300-400 MB) 가 모두 사용되지 않음. 별도 `load_quintet_dataset` (sensor + dscnn + cad + scan + static) 만들면 dataloader 메모리 절감 가능. 1차는 단순 처리.

### 3.3 학습 hp

E0 / E1 과 동일 (dropout 0.1, lr 1e-3, batch 256, max_epochs 5000, early-stop 50, grad_clip 1.0). 공정 비교 위해 변경 없음.

> 파라미터 카운트 ~104k → 6,373 SV 대비 ~6 SV/param. 풀-스택보다 좀 더 여유. 과적합 위험 다소 감소.

### 3.4 실행 명령

```bash
./venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E2 --quick    # smoke
./venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E2            # full run (사용자)
```

도커:
```bash
cd docker/lstm_ablation
docker compose run --rm e2
```

### 3.5 산출물

```
Sources/pipeline_outputs/experiments/lstm_ablation/E2_no_cameras/
├── experiment_meta.json       # use_v0=False, use_v1=False, n_total_feats=54
├── features/
│   └── normalization.json     # 54-차원 재정규화 (v0/v1 통계 모두 제거)
├── models/
│   ├── vppm_lstm_ablation_E2_{YS,UTS,UE,TE}_fold{0-4}.pt
│   └── training_log.json
└── results/
    ├── metrics_raw.json
    ├── metrics_summary.json
    ├── predictions_{YS,UTS,UE,TE}.csv
    ├── correlation_plots.png
    └── scatter_plot_uts.png
```

---

## 4. 판정 기준

### 4.1 1차 판정 (속성별 ΔRMSE)

| ΔRMSE 범위 | 판정 |
|:--|:--|
| Δ ≤ 0 | 카메라 분기 전체가 학습을 방해 — 강한 음의 가치 (가능성 낮음) |
| 0 < Δ < 1σ | **시나리오 A 결정적 입증** — 카메라 분기 전체 노이즈 ⭐ 가장 중요한 결과 후보 |
| 1σ ≤ Δ < 2σ | 경향성 — 카메라 합산 기여 marginal |
| Δ ≥ 2σ | 카메라 분기가 유의 기여 — 시나리오 D |

### 4.2 E1 vs E2 차분 분석

같은 fold 분할이라 fold 별 ΔΔRMSE = ΔE2 − ΔE1 가 깨끗하게 나옴 (동일 sample / 동일 random seed 가정).

```python
# fold 별로 (E1_rmse - E0_rmse) vs (E2_rmse - E0_rmse) 비교
# 모든 fold 에서 ΔE2 - ΔE1 의 부호 / 크기 일관성 확인
```

| ΔΔRMSE = ΔE2 − ΔE1 | 의미 |
|:--|:--|
| ≈ 0 | v1 redundant — v0 가 빠진 후 v1 추가 제거가 효과 없음 (= v0/v1 커버 영역 동일) |
| ≈ ΔE1 | v1 이 v0 와 같은 크기로 독립 기여 (additive) |
| ≫ ΔE1 | v1 이 v0 보다 더 핵심 — v0 빠질 때 v1 이 보완하다가, v1 도 빠지면 무너짐 |
| < 0 | regularization 효과 — v0 만 있을 때 overfitting, v1 도 빼면 일반화 개선 (드물 가능성) |

### 4.3 학습 동학

E0/E1 의 fold_epochs 와 E2 fold_epochs 비교. E2 입력 차원 가장 작아 수렴 빠를 수 있음.

E0 평균 epoch (참고): YS 105 / UTS 119 / UE 107 / TE 113.

---

## 5. 예상 결과 시나리오

### 5.1 시나리오 A (가장 강한 가설 검증)

```
E0 풀-스택       : YS 20.1, UTS 28.5, UE 6.5, TE 8.1
E1 no-v0         : YS 20.3, UTS 28.7, UE 6.5, TE 8.1
E2 no-cameras    : YS 20.5, UTS 29.0, UE 6.6, TE 8.2
ΔE2              : YS +0.4, UTS +0.5, UE +0.1, TE +0.1   (모두 1σ 이내)
```

**해석**: 카메라 분기 전체가 풀-스택의 정체된 누적 개선분만 담당. baseline → LSTM 점프의 출처는 **sensor 시간성 + dscnn / cad / scan 의 spatial-temporal 처리** 에 있음. **카메라 분기는 본 데이터/모델에서 incremental 가치 거의 없음** — minimal 모델(54-feat, ~104k params) 로 회귀 가능.

### 5.2 시나리오 D (반증 — 카메라 핵심)

```
E2 no-cameras    : YS 22.5, UTS 31.5, UE 7.0, TE 8.8
ΔE2              : YS +2.4, UTS +3.0, UE +0.5, TE +0.7   (모두 ≥1σ)
```

**해석**: 카메라 시간성이 풀-스택의 핵심 정보원. baseline → LSTM 점프의 큰 부분이 카메라 LSTM 처리에 기인. 모델 단순화 불가.

### 5.3 시나리오 B (v1 만 핵심)

```
E1 no-v0         : YS 20.2, UTS 28.6, UE 6.5, TE 8.1     (변화 거의 없음)
E2 no-cameras    : YS 21.0, UTS 29.5, UE 6.7, TE 8.3     (변화 유의)
ΔΔ = ΔE2 - ΔE1   : YS +0.7, UTS +0.8 (≈1σ)
```

**해석**: v0 는 redundant, v1 만 핵심 정보 (분말 도포 직후 표면 → recoater streak / soot 시그널). v1 분기에 더 많은 hp budget 할당 (d_embed sweep) 검토.

---

## 6. 후속 실험 분기

| 본 결과 | 다음 실험 |
|:--|:--|
| **시나리오 A 입증** | minimal 모델 (54-feat) 을 새 base 로 두고 sensor / dscnn / cad / scan 도 하나씩 빼는 후속 ablation E3-E6 → 진짜 핵심 분기 1-2개 식별 |
| **시나리오 B (v1 만)** | E1 결과 + v1 d_embed sweep (4/8/16/32) → v1 표현력 한계 |
| **시나리오 D (카메라 핵심)** | dual-cam fusion 강화 (cross-attention, share-CNN) 검토 |

> **시나리오 A 가 맞다면** 본 프로젝트의 모델 단순화 방향 결정 — sensor 시간성 + dscnn LSTM + cad/scan spatial-CNN 만으로 거의 동일 성능 가능. 학습/추론 비용 ↓, 카메라 캐시 (~750 MB) 불필요.

---

## 7. 주의 사항 (E1 외 추가)

1. **v0 / v1 cache 미로드 효과**: dataloader 가 v0/v1 캐시를 여전히 로드하면 메모리는 절감 안 됨. 메모리 critical 한 환경 (작은 GPU) 이라면 `load_quintet_dataset` 분기 추가 검토.

2. **E2 vs vppm_baseline 비교**:
   - vppm_baseline (21-feat 평균 MLP): YS 24.3, UTS 42.9, UE 9.3, TE 11.3
   - **E2 (54-feat 시퀀스, 카메라 빠짐)** 결과가 baseline 보다 크게 좋다면 → "baseline → LSTM 점프의 출처는 카메라가 아니라 sensor/dscnn/cad/scan 의 시간성 처리" 라는 강력 증거.

3. **공정 비교**: E0 (풀-스택) 결과는 seed 고정 없이 학습됨. ΔE2 가 fold std 수준일 가능성이 높으므로 **E0 와 E2 모두 seed 고정 후 재학습** 권장. 미실행 시 fold std 가 noise 인지 신호인지 모호.

4. **E1 + E2 동시 분석 필수**: E2 단독으론 v0/v1 어느 쪽이 핵심인지 불명. **E1 결과와 차분 분석** (§4.2) 으로만 v0 vs v1 구분 가능.

---

## 8. 연관 문서

- 공통 설정 / 가설 / 시나리오 표: [PLAN.md](./PLAN.md)
- E1 (v0 단독 제거): [PLAN_E1_no_v0.md](./PLAN_E1_no_v0.md)
- Base 풀-스택: [Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md)
- v0/v1 캐시 빌더:
  - [Sources/vppm/lstm/crop_stacks.py](../lstm/crop_stacks.py)
  - [Sources/vppm/lstm_dual/crop_stacks_v1.py](../lstm_dual/crop_stacks_v1.py)
- baseline 21-feat 결과 (참조선): [Sources/pipeline_outputs/experiments/vppm_baseline/results/metrics_raw.json](../../pipeline_outputs/experiments/vppm_baseline/results/metrics_raw.json)
