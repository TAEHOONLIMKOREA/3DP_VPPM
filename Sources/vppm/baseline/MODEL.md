# VPPM (Origin) 모델 설명

> **논문**: Scime et al., "A Data-Driven Framework for Direct Local Tensile Property Prediction of Laser Powder Bed Fusion Parts", *Materials* 2023, 16, 7293 — 논문 Table 6, Section 2.11 재현.

슈퍼복셀당 21개 핸드크래프트 피처를 받아 2-layer MLP 로 인장 특성(YS/UTS/UE/TE) 을 회귀하는 기본 VPPM 모델. 데이터 파이프라인(슈퍼복셀/21 피처/정규화/K-fold) 은 `Sources/vppm/README.md` 참조.

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

파라미터 수: `21·128 + 128 + 128·1 + 1 = 2,945` — 지극히 가벼움.

---

## 2. 학습 — `origin/train.py`

4 properties × 5 folds = **총 20개 모델** 학습. 각 fold 의 best 가중치를 `.pt` 로 저장.

### 학습 루프 핵심

```python
# train.py::train_single_fold
pred = model(x21)                     # (B, 1)
loss = criterion(pred, y)             # L1Loss (MAE)
loss.backward()
optim.step()
# EarlyStopper: val loss 50 에포크 동안 개선 없으면 중단, best state 복원
```

### 하이퍼파라미터 (`common/config.py`)

| 항목 | 값 | 근거 |
|---|---|---|
| Loss | `L1Loss` (MAE) | 이상치에 강건, 논문 원문 그대로 |
| Optimizer | Adam | lr=`1e-3`, betas=(0.9, 0.999), eps=`1e-4` |
| Batch size | 1000 | 전체 샘플이 3만 수준이라 큰 배치 가능 |
| Max epochs | 5000 | 실제 수렴은 수백 epoch 수준 |
| Early stop patience | 50 | val loss 개선 없을 때 |
| Weight init | `N(0, σ=0.1)` | 논문 Section 2.11 |
| CV | **sample-wise** 5-fold | 같은 sample 의 여러 슈퍼복셀이 train/val 에 걸치지 않게 |

> **참고**: 논문은 lr=`1e-8` 을 언급하지만 `[-1,1]` 정규화된 타겟에서는 수렴 불가. 실용적 값 `1e-3` 으로 조정 — 결과 RMSE 는 논문과 유사 수준 달성.

---

## 3. 추론 및 평가 — `origin/evaluate.py`

### 파이프라인

1. 각 fold 의 val split 에 대해 저장된 `.pt` 로드 → forward → 정규화된 예측.
2. `denormalize(pred, target_min, target_max)` 로 MPa/%  복원.
3. **per-sample min 집계** (논문 Section 3.1 "보수적 추정"):
   - 한 sample 에 속한 여러 슈퍼복셀 예측 중 **최솟값** 을 그 sample 의 예측으로 사용.
   - 물리적 가정: 인장 파단은 **가장 약한 지점** 에서 시작.
4. RMSE(fold) = √mean((p - t)²) → 5-fold 평균 ± 편차.
5. Naive baseline(전체 평균 예측)과 비교 → **reduction %** 계산.

### 출력

- `results/metrics_summary.json` — RMSE 요약 (VPPM vs Naive)
- `results/metrics_raw.json` — full precision 메트릭 (fold별 RMSE 포함)
- `results/predictions_{YS|UTS|UE|TE}.csv` — per-sample ground truth + 예측 + residual
- `results/correlation_plots.png` — 4 속성 2D 히스토그램 (논문 Figure 17)
- `results/scatter_plot_uts.png` — UTS 산점도 (논문 Figure 18)

---

## 4. 핵심 설계 결정

| 결정 | 이유 |
|------|------|
| 학습률 `1e-3` (논문은 `1e-8`) | `[-1,1]` 정규화된 타겟에서 `1e-8` 은 수렴 불가. 실용적 값으로 조정 |
| L1 Loss (MAE) | 논문 원문 유지. 인장 측정 이상치에 강건 |
| 샘플 단위 K-Fold | 같은 시편의 슈퍼복셀이 train/val 에 걸치면 데이터 누출 |
| 샘플별 최소값 취합 | 가장 취약한 슈퍼복셀이 파단을 결정한다는 물리적 직관 반영 |
| `FC(21→128→1)` 고정 | 논문 Table 6 그대로. 피처 엔지니어링이 이미 강력해 복잡한 네트워크 불필요 |

---

## 5. 실행 방법

```bash
# 피처 추출부터 학습·평가까지 일괄 (데이터 파이프라인이 이미 돌아있으면 train phase 만)
python -m Sources.vppm.run_pipeline --all

# 개별 단계
python -m Sources.vppm.run_pipeline --phase train      # 학습만 (features.npz 필요)
python -m Sources.vppm.run_pipeline --phase evaluate   # 평가만 (models/*.pt 필요)

# Ablation (예: 피처 11개만)
python -m Sources.vppm.run_pipeline --all --n-feats 11
```

---

## 6. 성능 (5-fold CV)

논문 수치와 유사 수준. 정확한 값은 `pipeline_outputs/results/metrics_summary.json` 참조.

| Property | VPPM RMSE | Naive RMSE | Reduction |
|---|---|---|---|
| YS  | ~22 MPa | ~34 MPa | ~35% |
| UTS | ~33 MPa | ~68 MPa | ~52% |
| UE  | ~7.0 %  | ~15 %   | ~53% |
| TE  | ~8.8 %  | ~18 %   | ~51% |

(LSTM 확장 모델의 성능은 `lstm/MODEL.md` 참조. 이미지 시퀀스를 추가하면 전 항목 1~3 포인트 개선됨)

---

## 7. 파일 맵 (모델 관련)

```
Sources/vppm/
├── common/
│   ├── config.py           # 학습 하이퍼파라미터 (HIDDEN_DIM, DROPOUT, LR, ...)
│   ├── dataset.py          # VPPMDataset, create_cv_splits, normalize/denormalize
│   └── model.py            # VPPM (본 문서), VPPM_LSTM
├── origin/
│   ├── MODEL.md            # ← 본 문서
│   ├── features.py         # 21 피처 추출 (데이터 파이프라인 — README.md 참조)
│   ├── train.py            # 학습 파이프라인 (EarlyStopper, train_single_fold, train_all)
│   └── evaluate.py         # evaluate_fold, evaluate_all, plot_correlation, save_metrics
└── run_pipeline.py         # entry point
```

### 산출물

```
Sources/pipeline_outputs/
├── models/
│   ├── vppm_{YS|UTS|UE|TE}_fold{0..4}.pt   # 20 모델
│   └── training_log.json                    # fold별 val loss, 에포크
└── results/
    ├── metrics_summary.json
    ├── metrics_raw.json
    ├── predictions_{YS|UTS|UE|TE}.csv
    ├── correlation_plots.png
    └── scatter_plot_uts.png
```

---

## 8. LSTM 확장과의 차이

| 구분 | VPPM (origin) | VPPM-LSTM |
|---|---|---|
| 입력 | 21 피처 (scalar) | 21 피처 + (T=70, C=9, 8×8) 이미지 시퀀스 |
| DSCNN 활용 | 스칼라 평균 8개 (피처 3~10) | 스칼라 8개 + 공간 맵 8채널 |
| 모델 | `FC(21→128→1)` | `CNN + Bi-LSTM` 으로 16-dim 임베딩 → `FC(37→128→1)` |
| 파라미터 | 2,945 | ~90k (CNN/LSTM 포함) |
| 추가 데이터 | 없음 | 슈퍼복셀당 70×9×8×8 크롭 이미지 (`image_stacks/stacks_all.h5`) |

자세한 LSTM 구조는 `lstm/MODEL.md` 참조.
