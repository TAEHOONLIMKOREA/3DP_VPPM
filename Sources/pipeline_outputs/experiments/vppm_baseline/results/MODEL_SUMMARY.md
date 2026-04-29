# VPPM 학습 모델 요약

> **논문**: Scime et al., "A Data-Driven Framework for Direct Local Tensile Property Prediction of Laser Powder Bed Fusion Parts", *Materials* 2023, 16, 7293
>
> **학습 일시**: 2026-04-07
>
> **모델 위치**: `Sources/pipeline_outputs/models/`

---

## 1. 모델 아키텍처

**VPPM (Voxelized Property Prediction Model)** — 2-layer MLP

```
Input(21) ─→ Linear(128) ─→ ReLU ─→ Dropout(0.1) ─→ Linear(1) ─→ Output
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

---

## 2. 학습 설정

| 하이퍼파라미터 | 값 | 비고 |
|:-------------:|:---:|------|
| 손실 함수 | L1 Loss (MAE) | 이상치 강건 |
| 옵티마이저 | Adam | betas=(0.9, 0.999), eps=1e-4 |
| 학습률 | 1e-3 | 논문(1e-8)과 다름 — [-1,1] 정규화 공간에서 수렴 가능하도록 조정 |
| 배치 크기 | 1,000 | |
| 최대 에포크 | 5,000 | |
| Early Stopping | patience=50 | 검증 손실 50 에포크 무개선 시 중단 |
| 가중치 초기화 | N(0, 0.1) | 논문 원문 그대로 |
| 교차검증 | 5-Fold CV | **샘플 단위** 분할 (데이터 누출 방지) |
| 정규화 | Min-Max → [-1, 1] | 피처·타겟 모두 동일 |
| 예측 집계 | 샘플별 최소값 | 가장 취약한 슈퍼복셀 기준 (보수적 추정) |

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
| 16 | build_plate_temperature | 센서 | 32.18 C | 3.76 | [27.0, 39.0] |
| 17 | bottom_flow_temperature | 센서 | 49.25 C | 6.33 | [41.0, 60.0] |
| 18 | actual_ventilator_flow_rate | 센서 | 39.35 | 3.06 | [24.96, 40.04] |
| 19 | laser_module | 스캔 | 0.359 | 0.480 | [0, 1] |
| 20 | laser_return_delay | 스캔 | 0.173 | 0.076 | [0, 0.5] |
| 21 | laser_stripe_boundaries | 스캔 | 43.02 | 36.60 | [0, 455] |

---

## 5. 학습 결과

### 5.1 Fold별 Validation Loss (L1, 정규화 공간)

| 속성 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **평균** |
|:----:|:------:|:------:|:------:|:------:|:------:|:-------:|
| YS | 0.0965 | 0.0975 | 0.1016 | 0.0994 | 0.0975 | **0.0985** |
| UTS | 0.0992 | 0.1017 | 0.1075 | 0.1022 | 0.0947 | **0.1010** |
| UE | 0.1918 | 0.1941 | 0.2050 | 0.1952 | 0.1968 | **0.1966** |
| TE | 0.2062 | 0.2141 | 0.2174 | 0.2068 | 0.2098 | **0.2108** |

### 5.2 Fold별 학습 에포크 (Early Stopping 적용)

| 속성 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **평균** |
|:----:|:------:|:------:|:------:|:------:|:------:|:-------:|
| YS | 580 | 167 | 715 | 323 | 487 | **454** |
| UTS | 403 | 517 | 418 | 549 | 1,120 | **601** |
| UE | 1,238 | 1,246 | 745 | 739 | 1,111 | **1,016** |
| TE | 1,170 | 939 | 874 | 744 | 912 | **928** |

### 5.3 최종 평가 (원본 스케일 RMSE)

| 속성 | VPPM RMSE | Naive RMSE | 개선율 | 측정오차 |
|:----:|:---------:|:----------:|:-----:|:-------:|
| **YS** | 28.7 +/- 0.6 MPa | 33.9 MPa | **15%** | 16.6 MPa |
| **UTS** | 60.7 +/- 2.6 MPa | 68.4 MPa | **11%** | 15.6 MPa |
| **UE** | 12.8 +/- 0.3 % | 15.0 % | **15%** | 1.73 % |
| **TE** | 15.5 +/- 0.2 % | 18.5 % | **17%** | 2.92 % |

> - **Naive RMSE**: 전체 평균을 예측값으로 사용한 기준선
> - **측정오차**: 동일 시편 반복 시험 시 내재된 오차 (논문 Section 2.9)
> - 모든 속성에서 Naive 대비 11~17% 개선

---

## 6. 결과 분석

### 강도 계열 (YS, UTS)
- Val loss ~0.10으로 상대적으로 안정적 수렴
- 학습 에포크 평균 454~601회로 빠른 수렴
- YS는 RMSE 28.7 MPa로 측정오차(16.6)의 약 1.7배 수준

### 연성 계열 (UE, TE)
- Val loss ~0.20으로 강도 대비 2배 높은 오차
- 학습 에포크 평균 928~1,016회로 수렴이 느림
- 연성은 본질적으로 미세 결함에 민감하여 예측 난이도가 높음 (논문과 일치하는 경향)

### 논문 결과와 비교
논문에서 보고한 5개 빌드 통합 학습 결과(Table 10)와의 비교:

| 속성 | 논문 RMSE | 재구현 RMSE | 비고 |
|:----:|:---------:|:----------:|------|
| YS | 22.2 MPa | 28.7 MPa | 피처 #20-21 미구현 영향 가능 |
| UTS | 37.1 MPa | 60.7 MPa | 정규화·학습률 차이 |
| UE | 9.57 % | 12.8 % | |
| TE | 12.8 % | 15.5 % | |

> 논문 대비 오차 차이의 가능한 원인:
> 1. 학습률 조정 (1e-8 → 1e-3) 및 정규화 방식 차이
> 2. 슈퍼복셀 유효성 판단 기준의 미세한 차이

---

## 7. 시각화 결과

| 파일 | 설명 |
|------|------|
| `results/correlation_plots.png` | 4개 속성 예측 vs 실측 2D 히스토그램 (논문 Figure 17) |
| `results/scatter_plot_uts.png` | UTS 예측 vs 실측 산점도 (논문 Figure 18) |

---

## 8. 모델 파일 목록

총 **20개 모델** (4 속성 x 5 folds), 각 ~14KB

```
models/
├── vppm_YS_fold{0-4}.pt      # Yield Strength
├── vppm_UTS_fold{0-4}.pt     # Ultimate Tensile Strength
├── vppm_UE_fold{0-4}.pt      # Uniform Elongation
├── vppm_TE_fold{0-4}.pt      # Total Elongation
└── training_log.json          # 학습 로그
```

### 추론 코드 예시

```python
import torch
from Sources.vppm.model import VPPM
from Sources.vppm.dataset import denormalize, load_norm_params

# 모델 로드
model = VPPM(n_feats=21)
model.load_state_dict(torch.load("Sources/pipeline_outputs/models/vppm_UTS_fold0.pt", weights_only=True))
model.eval()

# 추론 (입력: [-1,1] 정규화된 21차원 피처)
x = torch.randn(1, 21)  # 실제로는 정규화된 피처
with torch.no_grad():
    pred_norm = model(x).item()

# 역정규화
norm = load_norm_params("Sources/pipeline_outputs/features/normalization.json")
pred_mpa = denormalize(pred_norm, norm["target_min"]["ultimate_tensile_strength"],
                                   norm["target_max"]["ultimate_tensile_strength"])
```
