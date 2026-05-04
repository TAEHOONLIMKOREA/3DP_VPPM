# VPPM-1DCNN 실험 계획

> **한 줄 요약**: baseline (21-feat MLP) 의 **z-축 평균 압축**을 **채널별 1D CNN** 으로 교체한다. 출력은 여전히 SV 당 21차원이고, 그 후단 MLP(21→128→1) 는 baseline 과 완전히 동일하게 재사용한다.
>
> **참고 실험**: [`baseline`](../baseline/) — 본 실험의 직접 대조군이다. (LSTM 계열 실험은 본 실험과 무관하므로 비교 대상이 아니다.)
>
> **모델 코드 위치**: `Sources/vppm/1dcnn/`
> **결과 출력 위치 (예정)**: `Sources/pipeline_outputs/experiments/vppm_1dcnn/`

---

## 1. 동기

### 1.1 baseline 의 z-축 압축 방식 (현재 구조)

[`Sources/vppm/baseline/features.py`](../baseline/features.py) 는 한 SV 당 70 레이어를 z-축으로 압축해 21차원 스칼라 벡터를 만든다. 압축 방식은 [FEATURES.md §평균 처리 방식별 분류](../FEATURES.md#평균-처리-방식별-분류-1d-cnn-시퀀스화-관점) 의 P1–P4 네 패턴으로 갈라진다.

| 패턴 | 0-based 인덱스 | 개수 | xy 패치 평균 | z 축 압축 |
|:--:|:--|:--:|:--|:--|
| **P1** | 0, 1, 3, 4, 5, 6, 7, 8, 9, 10 | 10 | CAD 픽셀 평균 | **CAD 픽셀 수 가중 평균** |
| **P2** | 19, 20 | 2 | melt 픽셀 평균 | **유효 layer 수 단순 평균** |
| **P3** | 11, 12, 13, 14, 15, 16, 17 | 7 | (없음, 시계열) | **단순 평균** |
| **P4** | 2, 18 | 2 | (없음) | (스칼라 직접 할당, layer-invariant) |

P1+P2+P3 = 19 채널은 본질적으로 (70 layer) 길이의 1D 시퀀스를 평균만으로 1차원 스칼라로 줄이는 작업이다. P4 2채널은 layer 에 의존하지 않는 상수다.

### 1.2 평균이 잃는 것

두 SV 가 평균값은 같지만 z-방향 패턴이 다른 경우를 baseline 은 구분 못 한다.

- 예 1: 모든 layer 에서 약한 키홀(`seg_excessive_melting=0.05`) → 평균 0.05
- 예 2: 1 layer 에서 강한 키홀(0.7), 나머지는 0 → 평균 ≈ 0.01

물리적으로는 둘이 다른 결함 모드(분산형 vs 국소형) 인데, 평균만 보는 baseline 은 같은 입력으로 본다.

### 1.3 본 실험의 가설

> **가설**: P1·P2·P3 19채널의 layer 시퀀스에는 평균만으로는 사라지는 z-방향 변동 정보가 있다. **채널별 1D CNN** 으로 시퀀스를 압축하면 단순 평균보다 더 정보량이 많은 SV-level scalar 를 추출할 수 있고, 결과적으로 baseline 의 인장 특성 회귀 RMSE 가 감소한다.

---

## 2. 핵심 설계 — 평균을 1D CNN 으로 대체

### 2.1 입력 텐서

```
features_seq         : (N_sv, 70, 21)  float32   ← 레이어별 raw 값 (P1/P2/P3 raw, P4 broadcast)
valid_layer_mask     : (N_sv, 70)      bool      ← z-블록 안 layer 마스크 (현재는 거의 모두 True)
cad_count_per_layer  : (N_sv, 70)      int32     ← 각 layer SV xy 패치의 CAD 픽셀 수 (P1 가중 검증용)
melt_count_per_layer : (N_sv, 70)      int32     ← 각 layer 의 melt 픽셀 수 (P2 검증용)
sample_ids           : (N_sv,)         int32
build_ids            : (N_sv,)         int32
voxel_indices        : (N_sv, 3)       int32
targets_*            : (N_sv,)         float32
```

채널 21개 모두 같은 `(N_sv, 70)` shape 으로 정렬해 둔다 — P4 2채널 (#3 build_height, #19 laser_module) 은 70 레이어에 동일 값을 broadcast 해 둔다 (1D CNN 입력 형태 통일용; AdaptiveAvgPool 결과가 평균과 같아져 실제로는 변하지 않음).

### 2.2 모델 구조 — channel-wise (depthwise) 1D CNN

baseline 의 21-feat MLP 와 인터페이스를 동일하게 맞춘다. **출력은 SV 당 21차원** 이다.

```
입력  x        : (B, 21, 70)               # Conv1d 형식 (B, C, L)

[채널별 독립 1D CNN — depthwise]
DepthwiseConv1d(C=21, k=3, padding=1, groups=21)  → (B, 21, 70)
BatchNorm1d(21) → ReLU
DepthwiseConv1d(C=21, k=3, padding=1, groups=21)  → (B, 21, 70)
BatchNorm1d(21) → ReLU
AdaptiveAvgPool1d(1)                              → (B, 21, 1) → squeeze(-1) → (B, 21)

[baseline MLP — 그대로 재사용]
Linear(21 → 128) → ReLU → Dropout(0.1) → Linear(128 → 1)
```

핵심 포인트:
- **groups=21**: 채널 간 가중치 공유 없이 21개 채널이 각자 독립적인 1D CNN 을 갖는다. 각 피처(예: `seg_soot`) 의 layer 시퀀스가 다른 피처(예: `module_oxygen`)와 섞이지 않는다.
- **AdaptiveAvgPool1d(1)**: 70 layer → 1 scalar. 이 단계가 baseline 의 z-평균을 "학습 가능한 압축" 으로 바꾸는 부분.
- **MLP 후단**: baseline `VPPM` 클래스의 `(fc1=Linear(21→128), Dropout(0.1), fc2=Linear(128→1))` 을 그대로 재사용. 1DCNN 블록 직후의 21차원 벡터가 baseline 의 21-feat 와 1:1 자리에 들어간다.
- **가중치 초기화**: MLP 부분(`fc1`/`fc2`) 만 baseline 과 동일한 N(0, 0.1) 로 초기화. Conv/BN 은 PyTorch 기본 (Kaiming-uniform) 을 그대로 사용 (std=0.1 로 줄이면 활성이 죽는 위험).

---

## 3. 데이터 파이프라인

### 3.1 레이어별 raw 피처 추출 (Phase 1)

[`Sources/vppm/baseline/features.py`](../baseline/features.py) 의 4개 그룹 추출 함수를 **z-평균 직전 단계까지 분리** 한다. 평균은 1D CNN 이 학습으로 대신할 자리이므로 캐시에는 저장하지 않는다.

| 채널 그룹 | 본 실험의 헬퍼 함수 (신설) | 출력 shape | 원본 함수 |
|:--|:--|:--|:--|
| P1 (0, 1, 3–10) | `_per_layer_cad_block`, `_per_layer_dscnn_block` | `(n_sv, 70, 2)`, `(n_sv, 70, 8)` + `cad_count_per_layer (n_sv, 70)` | `_extract_cad_features_block`, `_extract_dscnn_features_block` |
| P2 (19, 20) | `_per_layer_scan_block` | `(n_sv, 70, 2)` + `melt_count_per_layer (n_sv, 70)` | `_extract_scan_features_block` |
| P3 (11–17) | `_per_layer_temporal_block` | `(n_sv, 70, 7)` (z-block 통째 broadcast) | `temporal_data[key][l0:l1]` |
| P4 (2, 18) | (헬퍼 없음) | layer-invariant 스칼라를 70 layer 에 broadcast | (그대로 사용) |

각 z-블록(70 layer)을 돌면서 SV 당 `(70, 21)` 텐서를 누적해 `features_seq.npz` 로 저장.

`distance_from_overhang` (#1) 의 vertical-column 누적 상태 (`_prev_cad_layer`, `_last_overhang_layer`) 는 baseline 과 동일하게 instance 변수로 carry-over 한다.

P2 채널은 melt 픽셀이 0인 layer 에서 0 으로 채운다 (baseline 과 동일 정책 — 평균이 1D CNN 으로 옮겨갔지만, NaN→SV 드롭 회피 의도는 유지).

**검증**: 본 캐시의 z-축 평균이 기존 [`Sources/pipeline_outputs/features/all_features.npz`](../../pipeline_outputs/features/all_features.npz) 와 채널별 ±1e-5 이내로 일치해야 한다.
- P1: `cad_count_per_layer` 가중 평균
- P2: `melt_count_per_layer > 0` layer 단순 평균
- P3: 단순 평균
- P4: broadcast 한 70 값이 모두 같으므로 평균 = 자기 자신

`features_seq.py --validate` 옵션으로 채널별 max abs diff 를 출력한다.

### 3.2 정규화

학습 직전에 **채널별 [-1, 1] min-max 정규화** ([`common/dataset.py::normalize`](../common/dataset.py) 사용). min/max 는 (N_sv × 70) 풀어서 채널마다 산출한다. 타겟 정규화는 baseline 과 동일.

NaN SV 드롭 정책도 baseline 과 동일 — 시퀀스에 NaN 이 있으면 SV 단위 드롭. P2 의 0-fallback 은 features_seq 단계에서 이미 처리되어 NaN 이 존재하지 않는다.

### 3.3 K-Fold

[`common/dataset.py::create_cv_splits`](../common/dataset.py) 의 sample-wise 5-fold 그대로 재사용 (seed=42). 같은 시편의 모든 SV 가 같은 fold 에 들어가도록 보장.

---

## 4. 디렉토리 구조

```
Sources/vppm/1dcnn/
├── PLAN.md                       ← 본 문서
├── __init__.py
├── config.py                     ← 1dcnn 전용 하이퍼 (kernel, layers, P1-P4 인덱스)
├── features_seq.py               ← FeatureSequenceExtractor (Phase 1 캐시 생성 + --validate)
├── dataset.py                    ← (N, 21, 70) 로더 + 채널별 정규화
├── model.py                      ← VPPM_1DCNN (depthwise CNN + baseline MLP)
├── train.py                      ← baseline/train.py 미러 (입력만 시퀀스)
├── evaluate.py                   ← RMSE + 빌드별 분해
└── run.py                        ← --phase {features, train, evaluate, all}

Sources/pipeline_outputs/experiments/vppm_1dcnn/
├── features/
│   ├── features_seq.npz          ← (N_sv, 70, 21) raw + masks + counts
│   └── normalization.json
├── models/
│   ├── vppm_1dcnn_{YS,UTS,UE,TE}_fold{0..4}.pt
│   └── training_log.json
└── results/
    ├── metrics_summary.json
    ├── predictions_{YS,UTS,UE,TE}.csv
    └── correlation_plots.png
```

`docker/1dcnn/` 의 Dockerfile / docker-compose.yml / README.md 는 본 PLAN 의 다음 단계에서 [docker-setup](../../../.claude/agents/docker-setup.md) 서브에이전트가 일관 패턴으로 생성한다.

---

## 5. 실행 단계 (Phase)

### Phase 1 — 시퀀스 캐시 생성

`run.py --phase features` 로 시작.

1. `features_seq.py::FeatureSequenceExtractor` 가 5개 빌드 HDF5 를 순회하며 SV 당 `(70, 21)` 텐서를 누적.
2. `valid_layer_mask`, `cad_count_per_layer`, `melt_count_per_layer` 도 함께 저장.
3. 검증 단계: `python -m Sources.vppm.1dcnn.features_seq --validate` 로 z-평균이 기존 baseline `all_features.npz` 와 채널별 ±1e-5 이내인지 확인.

### Phase 2 — 학습

`run.py --phase train` 으로 4 property × 5 fold = 20 모델 학습. baseline 의 [`train.py`](../baseline/train.py) 와 동일한 train loop. 차이는 입력 텐서 shape `(B, 21, 70)` 와 모델 클래스만.

### Phase 3 — 평가

`run.py --phase evaluate` 로 RMSE/R² 산출 + per-build 분해 + correlation plot.

---

## 6. 학습 / 평가 설정

| 항목 | 값 | 비고 |
|:--|:--|:--|
| 입력 | `(B, 21, 70)` | depthwise Conv1d 형식 |
| Conv 채널 | 21 → 21 → 21 (depthwise) | groups=21 |
| Kernel size | 3 (padding=1) | |
| Layers | 2 | + BN + ReLU |
| Pooling | AdaptiveAvgPool1d(1) | (B, 21) |
| MLP | 21 → 128 → 1 | baseline 과 동일 |
| Dropout | 0.1 | baseline 과 동일 |
| 손실 | L1Loss | baseline 과 동일 |
| Optimizer | Adam, lr=1e-3 | baseline 과 동일 |
| Batch size | 256 | baseline 과 동일 |
| Early stop | patience=50, max=5000 epoch | baseline 과 동일 |
| Grad clip | 1.0 | baseline 과 동일 |
| K-fold | 5 (sample-wise, seed=42) | baseline 과 동일 |
| Property | YS, UTS, UE, TE 별도 모델 | baseline 과 동일 |

→ **모델 블록만 다르고 나머지는 모두 baseline 과 동일**. 따라서 RMSE 차이가 1D CNN 압축 자체의 효과를 직접 반영한다.

---

## 7. 성공 기준

| 지표 | 합격선 | 의미 |
|:--|:--|:--|
| **5-fold 평균 RMSE** (4 property 평균) | baseline 대비 ↓ | 1D CNN 압축이 평균보다 정보를 더 잘 보존 |
| **per-property RMSE** | YS/UTS/UE/TE 모두 baseline 의 ±std 안 또는 더 좋음 | 특정 property 만 좋은 게 아니라 일관 개선 |
| **per-build RMSE** (B1.4 / B1.5) | baseline 대비 ≥ 5% 감소 | z-방향 결함 변동이 큰 빌드에서 더 큰 이득이 가설 |
| 학습 안정성 | fold std/mean < 7% | baseline 과 동등한 안정성 |

---

## 8. 위험 / 주의사항

| 위험 | 대응 |
|:--|:--|
| Phase 1 캐시 시간 (HDF5 5빌드 재파싱) | baseline `extract_features` 가 60–90 분/빌드 → 시퀀스 버전은 비슷한 비용. 빌드별 병렬 실행 가능. |
| P3 7채널이 z-블록 내 모든 SV 에서 동일 시계열 | 같은 z-블록 SV 들은 입력의 P3 부분이 완전히 같음 → 1D CNN 이 "동일 입력 → 동일 출력" 을 줄 뿐, 평균과 거의 같아질 수 있음. 신호의 핵심은 P1 (DSCNN 8채널) + P2 (스캔 2채널) 라고 가정. |
| P1 의 CAD-가중 평균이 단순 1D CNN 으로 재현 불가 | `cad_count_per_layer` 를 캐시에 보존해 검증 + 향후 보조 채널 옵션으로 확장 가능. |
| baseline 과 fold split 불일치 | `common/dataset.py::create_cv_splits(seed=42)` 그대로 호출. 평가 단계에서 baseline 과 같은 SV 들에 대해 비교. |

---

## 9. 다음 액션

1. 본 PLAN.md 검토 (사용자 승인) — 특히 §2.2 의 **depthwise k=3 × 2층 → AvgPool → 21차원 → MLP(21→128→1)** 구조가 의도와 맞는지 확인.
2. Phase 1 구현: `features_seq.py` 작성 + 캐시 검증 (z-평균이 기존 `all_features.npz` 와 일치).
3. `docker/1dcnn` compose 셋업 (docker-setup 서브에이전트).
4. Phase 2 학습 → baseline 결과(`Sources/vppm/baseline/MODEL.md`) 와 1:1 비교.
