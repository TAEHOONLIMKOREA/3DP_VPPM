# VPPM (Baseline) 모델 설명 + 학습 결과

> **논문**: Scime et al., "A Data-Driven Framework for Direct Local Tensile Property Prediction of Laser Powder Bed Fusion Parts", *Materials* 2023, 16, 7293 — Table 6 / Section 2.11 재현.
>
> **학습 일시**: 2026-04-07
>
> **모델 위치**: `Sources/pipeline_outputs/experiments/vppm_baseline/models/`
>
> **결과 위치**: `Sources/pipeline_outputs/experiments/vppm_baseline/results/`

슈퍼복셀당 21개 핸드크래프트 피처를 받아 2-layer MLP 로 인장 특성(YS/UTS/UE/TE) 을 회귀하는 기본 VPPM 모델. 데이터 파이프라인(슈퍼복셀/21 피처/정규화/K-fold) 은 [`Sources/vppm/README.md`](../README.md) 참조.

---

## 1. 아키텍처 — `common/model.py::VPPM`

```
Input x ∈ ℝ²¹
    ↓
FC(21 → 128)   weight: N(0, σ=0.1),  bias: 0
    ↓
ReLU
    ↓
Dropout(p = 0.1)
    ↓
FC(128 → 1)    weight: N(0, σ=0.1),  bias: 0
    ↓
ŷ ∈ ℝ (정규화된 [-1, 1] 타겟 공간)
```

| 항목 | 값 |
|------|------|
| 입력 차원 | 21 (엔지니어링 피처) |
| 은닉층 | 128 뉴런 |
| 활성화 함수 | ReLU |
| Dropout | 0.1 |
| 출력 차원 | 1 (단일 인장 특성) |
| 총 파라미터 수 | **2,945** |

### 레이어별 파라미터

| Layer | Shape | Parameters |
|-------|-------|-----------|
| fc1.weight | [128, 21] | 2,688 |
| fc1.bias | [128] | 128 |
| fc2.weight | [1, 128] | 128 |
| fc2.bias | [1] | 1 |

파라미터 수: `21·128 + 128 + 128·1 + 1 = 2,945` — 지극히 가벼움.

---

## 2. 학습 설정 — `common/config.py`

4 properties × 5 folds = **총 20개 모델** 학습. 각 fold 의 best 가중치를 `.pt` 로 저장.

| 하이퍼파라미터 | 값 | 비고 |
|:-------------:|:---:|------|
| 손실 함수 | `L1Loss` (MAE) | 이상치 강건, 논문 원문 그대로 |
| 옵티마이저 | Adam | betas=(0.9, 0.999), eps=`1e-4` |
| 학습률 | `1e-3` | 논문(`1e-8`)과 다름 — `[-1,1]` 정규화 공간에서 수렴 가능하도록 조정 |
| 배치 크기 | 1,000 | 전체 샘플이 3만 수준이라 큰 배치 가능 |
| 최대 에포크 | 5,000 | 실제 수렴은 수백 에포크 수준 |
| Early Stopping | patience=50 | 검증 손실 50 에포크 무개선 시 중단, best state 복원 |
| 가중치 초기화 | `N(0, σ=0.1)` | 논문 Section 2.11 |
| 교차검증 | **sample-wise** 5-fold | 같은 시편의 슈퍼복셀이 train/val 에 걸치지 않게 (데이터 누출 방지) |
| 정규화 | Min-Max → `[-1, 1]` | 피처·타겟 모두 동일 |
| 예측 집계 | 샘플별 최소값 | 가장 취약한 슈퍼복셀 기준 (보수적 추정 — 논문 Section 3.1) |

### 학습 루프 핵심

```python
# train.py::train_single_fold
pred = model(x21)                     # (B, 1)
loss = criterion(pred, y)             # L1Loss (MAE)
loss.backward()
optim.step()
# EarlyStopper: val loss 50 에포크 동안 개선 없으면 중단, best state 복원
```

---

## 3. 학습 데이터

### 3.1 빌드별 슈퍼복셀

| 빌드 | 슈퍼복셀 수 | 고유 시편 수 | 특성 |
|:----:|:-----------:|:----------:|------|
| B1.1 | 10,173 | 3,286 | 기준 공정 조건 |
| B1.2 | 10,173 | 3,286 | 다양한 공정 파라미터 (Nominal/Best/LOF/Keyhole) |
| B1.3 | 2,840 | 896 | 오버행 형상 |
| B1.4 | 3,126 | 992 | 스패터/가스 유량 변화 |
| B1.5 | 9,735 | 3,296 | 리코터 손상/분말 공급 부족 |
| **합계** | **36,047** | **11,756** | |

### 3.2 필터링 후 (학습에 사용된 데이터)

- **필터 조건**: NaN 피처 제거 + UTS < 50 MPa 제거 + NaN 타겟 제거
- **유효 슈퍼복셀**: 36,047 → **19,313** (53.6%)

### 3.3 타겟 분포 (필터링 전)

| 속성 | 단위 | 평균 | 표준편차 | 최솟값 | 최댓값 | 내재 측정오차 |
|:----:|:----:|:----:|:------:|:-----:|:-----:|:-----------:|
| YS (항복강도) | MPa | 355.4 | 33.9 | 71.2 | 418.4 | 16.6 |
| UTS (인장강도) | MPa | 530.8 | 68.4 | 74.7 | 609.9 | 15.6 |
| UE (균일연신율) | % | 46.4 | 15.0 | 0.1 | 69.1 | 1.73 |
| TE (총연신율) | % | 59.8 | 18.5 | 4.1 | 94.3 | 2.92 |

### 3.4 정규화 범위

| 속성 | 원본 범위 | 정규화 후 |
|:----:|:---------:|:---------:|
| YS | [71.25, 418.36] MPa | [-1, 1] |
| UTS | [74.72, 609.92] MPa | [-1, 1] |
| UE | [0.08, 69.13] % | [-1, 1] |
| TE | [4.07, 94.28] % | [-1, 1] |

---

## 4. 입력 피처 (21개)

| # | 피처명 | 소스 | 평균 | 표준편차 | 범위 |
|:-:|--------|:----:|:----:|:------:|:----:|
| 1 | distance_from_edge | CAD | 1.92 mm | 1.02 | [0.14, 3.00] |
| 2 | distance_from_overhang | CAD | 70.05 layers | 3.38 | [47.16, 71.00] |
| 3 | build_height | 좌표 | 28.43 mm | 17.83 | [12.25, 71.75] |
| 4 | seg_powder | DSCNN | 0.090 | 0.161 | [0, 0.86] |
| 5 | seg_printed | DSCNN | 0.895 | 0.181 | [0.07, 1.00] |
| 6 | seg_recoater_streaking | DSCNN | 0.008 | 0.047 | [0, 0.70] |
| 7 | seg_edge_swelling | DSCNN | 0.003 | 0.015 | [0, 0.29] |
| 8 | seg_debris | DSCNN | 0.003 | 0.033 | [0, 0.55] |
| 9 | seg_super_elevation | DSCNN | 0.002 | 0.006 | [0, 0.03] |
| 10 | seg_soot | DSCNN | 0.055 | 0.142 | [0, 0.91] |
| 11 | seg_excessive_melting | DSCNN | 0.001 | 0.013 | [0, 0.52] |
| 12 | layer_print_time | 센서 | 95.71 s | 34.38 | [45.42, 155.94] |
| 13 | top_gas_flow_rate | 센서 | 95.09 | 7.42 | [62.57, 99.74] |
| 14 | bottom_gas_flow_rate | 센서 | 39.35 | 3.06 | [24.95, 40.07] |
| 15 | module_oxygen | 센서 | 0.039 % | 0.052 | [0, 0.15] |
| 16 | build_plate_temperature | 센서 | 32.18 °C | 3.76 | [27.0, 39.0] |
| 17 | bottom_flow_temperature | 센서 | 49.25 °C | 6.33 | [41.0, 60.0] |
| 18 | actual_ventilator_flow_rate | 센서 | 39.35 | 3.06 | [24.96, 40.04] |
| 19 | laser_module | 스캔 | 0.359 | 0.480 | [0, 1] |
| 20 | laser_return_delay | 스캔 | 0.173 | 0.076 | [0, 0.5] |
| 21 | laser_stripe_boundaries | 스캔 | 43.02 | 36.60 | [0, 455] |

---

## 5. 추론 및 평가 — `baseline/evaluate.py`

### 파이프라인

1. 각 fold 의 val split 에 대해 저장된 `.pt` 로드 → forward → 정규화된 예측.
2. `denormalize(pred, target_min, target_max)` 로 MPa/% 복원.
3. **per-sample min 집계** (논문 Section 3.1 "보수적 추정"):
   - 한 sample 에 속한 여러 슈퍼복셀 예측 중 **최솟값** 을 그 sample 의 예측으로 사용.
   - 물리적 가정: 인장 파단은 **가장 약한 지점** 에서 시작.
4. RMSE(fold) = √mean((p - t)²) → 5-fold 평균 ± 편차.
5. Naive baseline(전체 평균 예측)과 비교 → **reduction %** 계산.

---

## 6. 학습 결과

### 6.1 Fold별 Validation Loss (L1, 정규화 공간)

| 속성 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **평균** |
|:----:|:------:|:------:|:------:|:------:|:------:|:-------:|
| YS  | 0.0965 | 0.0975 | 0.1016 | 0.0994 | 0.0975 | **0.0985** |
| UTS | 0.0992 | 0.1017 | 0.1075 | 0.1022 | 0.0947 | **0.1010** |
| UE  | 0.1918 | 0.1941 | 0.2050 | 0.1952 | 0.1968 | **0.1966** |
| TE  | 0.2062 | 0.2141 | 0.2174 | 0.2068 | 0.2098 | **0.2108** |

### 6.2 Fold별 학습 에포크 (Early Stopping 적용)

| 속성 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **평균** |
|:----:|:------:|:------:|:------:|:------:|:------:|:-------:|
| YS  | 580   | 167   | 715   | 323   | 487   | **454**   |
| UTS | 403   | 517   | 418   | 549   | 1,120 | **601**   |
| UE  | 1,238 | 1,246 | 745   | 739   | 1,111 | **1,016** |
| TE  | 1,170 | 939   | 874   | 744   | 912   | **928**   |

### 6.3 최종 평가 (원본 스케일 RMSE, per-sample min 집계 후)

| 속성 | VPPM RMSE | Naive RMSE | 개선율 | 측정오차 |
|:----:|:---------:|:----------:|:-----:|:-------:|
| **YS**  | 28.7 ± 0.6 MPa | 33.9 MPa | **15%** | 16.6 MPa |
| **UTS** | 60.7 ± 2.6 MPa | 68.4 MPa | **11%** | 15.6 MPa |
| **UE**  | 12.8 ± 0.3 %   | 15.0 %   | **15%** | 1.73 %   |
| **TE**  | 15.5 ± 0.2 %   | 18.5 %   | **17%** | 2.92 %   |

> - **Naive RMSE**: 전체 평균을 예측값으로 사용한 기준선
> - **측정오차**: 동일 시편 반복 시험 시 내재된 오차 (논문 Section 2.9)
> - 모든 속성에서 Naive 대비 11~17% 개선

### 6.4 결과 분석

**강도 계열 (YS, UTS)**
- Val loss ~0.10 으로 상대적으로 안정적 수렴
- 학습 에포크 평균 454~601회로 빠른 수렴
- YS 는 RMSE 28.7 MPa 로 측정오차(16.6)의 약 1.7배 수준

**연성 계열 (UE, TE)**
- Val loss ~0.20 으로 강도 대비 2배 높은 오차
- 학습 에포크 평균 928~1,016회로 수렴이 느림
- 연성은 본질적으로 미세 결함에 민감하여 예측 난이도가 높음 (논문과 일치하는 경향)

### 6.5 논문 결과와 비교 (Table 10)

| 속성 | 논문 RMSE | 재구현 RMSE | 비고 |
|:----:|:---------:|:----------:|------|
| YS  | 22.2 MPa | 28.7 MPa | 피처 #20-21 미구현 영향 가능 |
| UTS | 37.1 MPa | 60.7 MPa | 정규화·학습률 차이 |
| UE  | 9.57 %   | 12.8 %   | |
| TE  | 12.8 %   | 15.5 %   | |

> 논문 대비 오차 차이의 가능한 원인:
> 1. 학습률 조정 (`1e-8` → `1e-3`) 및 정규화 방식 차이
> 2. 슈퍼복셀 유효성 판단 기준의 미세한 차이

---

## 7. 핵심 설계 결정

| 결정 | 이유 |
|------|------|
| 학습률 `1e-3` (논문은 `1e-8`) | `[-1,1]` 정규화된 타겟에서 `1e-8` 은 수렴 불가. 실용적 값으로 조정 |
| L1 Loss (MAE) | 논문 원문 유지. 인장 측정 이상치에 강건 |
| 샘플 단위 K-Fold | 같은 시편의 슈퍼복셀이 train/val 에 걸치면 데이터 누출 |
| 샘플별 최소값 취합 | 가장 취약한 슈퍼복셀이 파단을 결정한다는 물리적 직관 반영 |
| `FC(21→128→1)` 고정 | 논문 Table 6 그대로. 피처 엔지니어링이 이미 강력해 복잡한 네트워크 불필요 |

---

## 8. 실행 방법

```bash
# 피처 추출부터 학습·평가까지 일괄 (데이터 파이프라인이 이미 돌아있으면 train phase 만)
python -m Sources.vppm.run_pipeline --all

# 개별 단계
python -m Sources.vppm.run_pipeline --phase train      # 학습만 (features.npz 필요)
python -m Sources.vppm.run_pipeline --phase evaluate   # 평가만 (models/*.pt 필요)

# Ablation (예: 피처 11개만)
python -m Sources.vppm.run_pipeline --all --n-feats 11
```

### 추론 코드 예시

```python
import torch
from Sources.vppm.common.model import VPPM
from Sources.vppm.common.dataset import denormalize, load_norm_params

# 모델 로드
model = VPPM(n_feats=21)
model.load_state_dict(torch.load(
    "Sources/pipeline_outputs/experiments/vppm_baseline/models/vppm_UTS_fold0.pt",
    weights_only=True,
))
model.eval()

# 추론 (입력: [-1,1] 정규화된 21차원 피처)
x = torch.randn(1, 21)  # 실제로는 정규화된 피처
with torch.no_grad():
    pred_norm = model(x).item()

# 역정규화
norm = load_norm_params(
    "Sources/pipeline_outputs/experiments/vppm_baseline/features/normalization.json"
)
pred_mpa = denormalize(
    pred_norm,
    norm["target_min"]["ultimate_tensile_strength"],
    norm["target_max"]["ultimate_tensile_strength"],
)
```

---

## 9. 산출물

총 **20개 모델** (4 속성 × 5 folds), 각 ~14KB.

```
Sources/pipeline_outputs/experiments/vppm_baseline/
├── models/
│   ├── vppm_{YS|UTS|UE|TE}_fold{0..4}.pt   # 20 모델
│   └── training_log.json                    # fold별 val loss, 에포크
└── results/
    ├── MODEL.md                  # ← 본 문서 (사본/링크)
    ├── metrics_summary.json      # RMSE 요약 (VPPM vs Naive)
    ├── metrics_raw.json          # full precision 메트릭 (fold별 RMSE 포함)
    ├── predictions_{YS|UTS|UE|TE}.csv  # per-sample GT + 예측 + residual
    ├── correlation_plots.png     # 4 속성 2D 히스토그램 (논문 Figure 17)
    └── scatter_plot_uts.png      # UTS 산점도 (논문 Figure 18)
```

---

## 10. 파일 맵 (모델 관련)

```
Sources/vppm/
├── common/
│   ├── config.py           # 학습 하이퍼파라미터 (HIDDEN_DIM, DROPOUT, LR, ...)
│   ├── dataset.py          # VPPMDataset, create_cv_splits, normalize/denormalize
│   └── model.py            # VPPM (본 문서), VPPM_LSTM
├── baseline/
│   ├── MODEL.md            # ← 본 문서
│   ├── FEATURES.md         # 21 피처 추출 상세
│   ├── features.py         # 21 피처 추출
│   ├── scan_features.py    # 스캔 경로 기반 피처 (#19~21)
│   ├── train.py            # 학습 파이프라인 (EarlyStopper, train_single_fold, train_all)
│   └── evaluate.py         # evaluate_fold, evaluate_all, plot_correlation, save_metrics
├── lstm/
│   └── MODEL.md            # VPPM-LSTM 확장 모델 문서
└── run_pipeline.py         # entry point
```

---

## 11. LSTM 확장과의 차이

| 구분 | VPPM (baseline) | VPPM-LSTM |
|---|---|---|
| 입력 | 21 피처 (scalar) | 21 피처 + (T=70, C=9, 8×8) 이미지 시퀀스 |
| DSCNN 활용 | 스칼라 평균 8개 (피처 4~11) | 스칼라 8개 + 공간 맵 8채널 |
| 모델 | `FC(21→128→1)` | `CNN + Bi-LSTM` 으로 16-dim 임베딩 → `FC(37→128→1)` |
| 파라미터 | 2,945 | ~90k (CNN/LSTM 포함) |
| 추가 데이터 | 없음 | 슈퍼복셀당 70×9×8×8 크롭 이미지 (`image_stacks/stacks_all.h5`) |

자세한 LSTM 구조는 [`lstm/MODEL.md`](../lstm/MODEL.md) 참조. 이미지 시퀀스를 추가하면 전 항목 1~3 포인트 개선됨.
