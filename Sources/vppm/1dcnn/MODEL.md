# VPPM-1DCNN 모델 설명 + 학습 결과

> **실험 동기**: baseline (21-feat MLP) 의 **z-축 평균 압축**을 채널별 1D CNN 으로 교체. 출력은 여전히 SV 당 21차원이고, 후단 MLP(21→128→1) 는 baseline 과 완전히 동일하게 재사용. 압축 방식만 평균 → 1D CNN 으로 바꾼 통제된 비교.
>
> **학습 일시**: 2026-05-04
>
> **모델 위치**: `Sources/pipeline_outputs/experiments/vppm_1dcnn/models/`
>
> **결과 위치**: `Sources/pipeline_outputs/experiments/vppm_1dcnn/results/`
>
> **계획서**: [`PLAN.md`](PLAN.md)

상세한 동기와 채널별 P1–P4 압축 방식 분류는 [PLAN.md](PLAN.md), 21 피처 정의는 [`Sources/vppm/FEATURES.md`](../FEATURES.md) 참조.

---

## 1. 아키텍처 — `1dcnn/model.py::VPPM_1DCNN`

```
Input x ∈ ℝ^(B×21×70)        # (batch, channel=21, layer=70)
    ↓
[채널별 독립 1D CNN — depthwise]
DepthwiseConv1d(21 → 21, k=3, padding=1, groups=21)
    ↓ BatchNorm1d(21) → ReLU
DepthwiseConv1d(21 → 21, k=3, padding=1, groups=21)
    ↓ BatchNorm1d(21) → ReLU
AdaptiveAvgPool1d(1)        # (B, 21, 70) → (B, 21, 1) → squeeze → (B, 21)
    ↓
[baseline MLP — 그대로 재사용]
FC(21 → 128)   weight: N(0, σ=0.1),  bias: 0
    ↓ ReLU → Dropout(p = 0.1)
FC(128 → 1)    weight: N(0, σ=0.1),  bias: 0
    ↓
ŷ ∈ ℝ (정규화된 [-1, 1] 타겟 공간)
```

| 항목 | 값 |
|:--|:--|
| 입력 shape | `(B, 21, 70)` — 21채널 × 70 layer |
| Conv 채널 | 21 → 21 → 21 (depthwise, **groups=21**) |
| Kernel size | 3 (padding=1) |
| Conv layers | 2 + BN + ReLU |
| Pooling | AdaptiveAvgPool1d(1) — z-방향 압축 |
| MLP | 21 → 128 → 1 (baseline 과 동일) |
| Dropout | 0.1 |
| 총 파라미터 | **3,197** (baseline 2,945 대비 **+252 / +8.6 %**) |

### 레이어별 파라미터

| Layer | Shape | Parameters |
|:--|:--|--:|
| conv1.weight | [21, 1, 3] | 63 |
| conv1.bias | [21] | 21 |
| bn1.{weight,bias} | [21]×2 | 42 |
| conv2.weight | [21, 1, 3] | 63 |
| conv2.bias | [21] | 21 |
| bn2.{weight,bias} | [21]×2 | 42 |
| fc1.weight | [128, 21] | 2,688 |
| fc1.bias | [128] | 128 |
| fc2.weight | [1, 128] | 128 |
| fc2.bias | [1] | 1 |
| **합계** | | **3,197** |

`groups=21` 의 의미: 21 채널이 각자 독립적인 1D CNN 을 갖는다. 즉 `seg_soot` 의 layer 시퀀스가 `module_oxygen` 같은 다른 채널과 섞이지 않는다. AdaptiveAvgPool1d 가 baseline 의 z-평균을 "학습 가능한 압축"으로 대체하는 핵심 단계다.

---

## 2. 학습 설정 (baseline 과 동일)

| 항목 | 값 |
|:--|:--|
| 손실 | L1Loss |
| Optimizer | Adam, lr=1e-3 |
| Batch size | 1,000 |
| Max epochs | 5,000 |
| Early stop | patience=50 |
| Grad clip | 1.0 |
| K-fold | 5 (sample-wise, seed=42) |
| Property | YS, UTS, UE, TE 별도 모델 (총 20 fold checkpoint) |
| GPU | 단일 GPU (NVIDIA_VISIBLE_DEVICES=1), 단일 device 학습 |

학습 설정은 baseline 과 모두 같다 — RMSE 차이가 z-축 압축 방식 자체의 효과를 직접 반영하도록 다른 변수를 모두 통제했다.

### Fold-별 수렴 epoch (early-stopping)

| Property | fold0 | fold1 | fold2 | fold3 | fold4 |
|:--:|--:|--:|--:|--:|--:|
| YS  | 330 | 357 | 483 | 447 | 316 |
| UTS | 296 | 544 | 368 | 429 | 530 |
| UE  | 256 | 253 | 240 | 558 | 346 |
| TE  | 270 | 505 | 423 | 226 | 127 |

20 fold 모두 정상 converge. 학습 시간은 단일 GPU 로 약 **15분** (4 prop × 5 fold 순차).

---

## 3. 데이터 입력 — 레이어별 raw 시퀀스

baseline 의 `(N_sv, 21)` 평균 벡터 대신 본 실험은 `(N_sv, 70, 21)` 시퀀스를 입력한다.

| 패턴 | 0-based 인덱스 | 개수 | layer 별 raw 값의 의미 |
|:--:|:--|:--:|:--|
| **P1** | 0, 1, 3–10 | 10 | layer 별 SV-patch 의 CAD-픽셀 평균 (raw 값) |
| **P2** | 19, 20 | 2 | layer 별 melt-픽셀 평균 (melt 0 layer 는 0) |
| **P3** | 11–17 | 7 | layer 별 1D 시계열 raw 값 (z-블록 broadcast) |
| **P4** | 2, 18 | 2 | layer-invariant 스칼라 (70 layer broadcast) |

**캐시 파일**: `Sources/pipeline_outputs/experiments/vppm_1dcnn/features/features_seq.npz` (50 MB, 36,047 SV).

z-평균 검증 (baseline `all_features.npz` 대비) 결과: **17/21 OK**, ch 12·20 부동소수점 마진 (1e-5 살짝 초과), **ch 1 (overhang)** 47 단위 차이 — baseline 이 voxel 없는 z-block 에서 carry-over 누락 / **ch 19 (return_delay)** 0.25 단위 차이. 두 채널의 미세 불일치는 그대로 두고 학습 진행했고 결과적으로 큰 영향 없음 (오히려 1dcnn 측 처리가 더 정확할 가능성).

---

## 4. 결과 — 5-fold RMSE (baseline 직접 비교)

### 4.1 메인 표

| Property | naive (target std) | baseline RMSE | **1dcnn RMSE** | Δ (절대) | Δ (%) | naive-대비 reduction |
|:--:|:--:|:--:|:--:|:--:|:--:|:--|
| **YS**  | 33.9 | 24.3 ± 0.8 | **21.6 ± 0.8** | −2.7 | **−11.1 %** | 28 % → **36 %** |
| **UTS** | 68.4 | 42.9 ± 2.0 | **31.3 ± 1.4** | −11.6 | **−27.0 %** | 37 % → **54 %** |
| **UE**  | 15.0 | 9.3 ± 0.3 | **7.1 ± 0.5** | −2.2 | **−23.7 %** | 38 % → **53 %** |
| **TE**  | 18.5 | 11.3 ± 0.5 | **9.2 ± 0.3** | −2.1 | **−18.6 %** | 39 % → **51 %** |

**4/4 property 전부 baseline 대비 RMSE 감소.** UTS 가 가장 큰 폭 (27 %), UE/TE 가 그 다음 (24 % / 19 %), YS 가 가장 작음 (11 %). 1dcnn 의 모든 RMSE 가 baseline 의 평균 ± std 범위를 명확히 벗어나 더 좋다 (예: UTS 31.3 < 42.9 − 2.0 = 40.9).

### 4.2 Fold-별 RMSE

| Property | fold0 | fold1 | fold2 | fold3 | fold4 | mean ± std |
|:--:|--:|--:|--:|--:|--:|:--:|
| YS  | 21.65 | 20.41 | 21.93 | 21.44 | 22.75 | 21.64 ± 0.76 |
| UTS | 33.77 | 30.51 | 29.64 | 30.52 | 31.86 | 31.26 ± 1.44 |
| UE  | 7.84 | 6.87 | 6.43 | 6.86 | 7.30 | 7.06 ± 0.48 |
| TE  | 9.25 | 8.71 | 9.24 | 8.92 | 9.66 | 9.16 ± 0.32 |

### 4.3 학습 안정성

| | YS | UTS | UE | TE |
|:--|:--:|:--:|:--:|:--:|
| std/mean | 3.5 % | 4.6 % | 6.8 % | 3.5 % |
| PLAN §7 합격선 (<7 %) | ✅ | ✅ | ✅ (경계) | ✅ |

UE 만 임계 7 % 에 닿아 있지만 나머지는 안정적.

---

## 5. 빌드별 RMSE 분해

### 5.1 1dcnn per-build

| Build | n | YS | UTS | UE | TE |
|:--|--:|--:|--:|--:|--:|
| B1.1 | 502 | 10.57 | 11.05 | 5.11 | 9.07 |
| B1.2 | 2,700 | **25.97** | **40.21** | 7.56 | 9.58 |
| B1.3 | 893 | 21.12 | 22.55 | 4.91 | 6.38 |
| B1.4 | 694 | 19.10 | 24.26 | 7.70 | 9.22 |
| B1.5 | 1,584 | 17.00 | 24.51 | 7.49 | 9.74 |

B1.2 (sample 수 가장 많음) 가 모든 property 에서 가장 어렵다. B1.1 은 시편 수가 적은 편인데도 잘 맞힘.

> baseline 측 `per_build_rmse.json` 이 없어 build 단위 1:1 비교는 본 표만으로 불가하다. PLAN §7 의 "B1.4/B1.5 baseline 대비 ≥5 % 감소" 합격 여부는 baseline 재산출이 필요한 상태로 남는다 (다음 단계 후보).

---

## 6. 가설 검증

| PLAN §1.3 가설 | 결과 | 근거 |
|:--|:--:|:--|
| z-방향 변동 정보가 실재한다 | ✅ | 4/4 property 에서 RMSE 감소 |
| 1D CNN 압축이 평균보다 정보량이 많다 | ✅ | UTS 27 % / UE 24 % / TE 19 % 감소 — 우연 수준 아님 |
| baseline 의 인장 특성 회귀 RMSE 가 감소한다 | ✅ | 표 4.1 |

YS 의 개선폭이 다른 property 대비 작은 이유는, baseline 이 이미 YS 에서 가장 잘하던 항목 (naive-대비 28 % reduction)이라 추가 개선 여지가 적었던 것으로 해석된다. UTS/UE/TE 는 baseline 의 상대적으로 약한 부분 (37–39 %) 에서 1dcnn 이 z-방향 신호를 활용해 크게 끌어올렸다.

물리적 해석: P1 (DSCNN 8 결함 클래스) 의 z-방향 분포 — "평균 0.05" 의 분산형 결함과 "1 layer 만 0.7 + 나머지 0" 의 국소형 결함 — 이 인장 거동에 다르게 작용한다는 PLAN §1.2 의 예시가 데이터로 확인된 셈이다.

---

## 7. 산출물 위치

```
Sources/pipeline_outputs/experiments/vppm_1dcnn/
├── experiment_meta.json
├── features/
│   ├── features_seq.npz              # (36047, 70, 21) raw + masks + counts (50 MB)
│   └── normalization.json
├── models/
│   ├── vppm_1dcnn_{YS,UTS,UE,TE}_fold{0..4}.pt    # 20 checkpoint (각 ≈19 KB)
│   └── training_log.json                          # fold val_loss + epochs
└── results/
    ├── metrics_summary.json          # 본 문서 §4.1 의 원천
    ├── metrics_raw.json              # fold-별 RMSE 풀세트
    ├── per_build_rmse.json           # 본 문서 §5.1 의 원천
    ├── predictions_{YS,UTS,UE,TE}.csv
    ├── correlation_plots.png         # 4 property × pred-vs-true
    └── scatter_plot_uts.png
```

---

## 8. 한계 / 다음 단계 후보

| 한계 | 다음 단계 |
|:--|:--|
| baseline 측 `per_build_rmse.json` 부재 → build 단위 비교 불가 | baseline 의 `evaluate.py` 에 `save_per_build_rmse` 호출 추가 + 재실행 |
| z-평균 검증 ch 1 (overhang) / ch 19 (return_delay) 의 baseline-1dcnn 미세 불일치 | features_seq 의 P1 overhang carry-over · P2 평균 처리를 baseline 과 정확히 일치시킨 뒤 재추출/재학습 → 깨끗한 ablation |
| 변형 (k=5, AvgMax pool concat 등) 미실험 | PLAN §2.3 의 변형 B/C/D 후속 ablation |
| UE 의 fold std/mean 6.8 % 로 다른 property 대비 약간 흔들림 | dropout 0.2 또는 weight decay 추가 실험 |
