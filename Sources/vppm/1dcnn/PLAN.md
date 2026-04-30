# VPPM-1DCNN 구현 계획

> **핵심 아이디어**: baseline의 21개 supervoxel 피처는 레이어 축으로 **평균/가중평균**되어 z-방향 변동 정보가 사라진다. 레이어별 raw 피처 시퀀스 `(70, 21)` 을 그대로 두고 **1D CNN으로 인코딩** 하여, 레이어 간 패턴(연속 결함, 점진적 변화, 국소 이상)을 학습한다.
>
> **모델 코드 위치**: `Sources/vppm/1dcnn/` (본 디렉토리)
> **결과 출력 위치 (예정)**: `Sources/pipeline_outputs/experiments/vppm_1dcnn/`
> **참고 실험**: [`vppm_baseline`](../../pipeline_outputs/experiments/vppm_baseline/), [`vppm_lstm`](../../pipeline_outputs/experiments/vppm_lstm/), [`vppm_lstm_dual`](../../pipeline_outputs/experiments/vppm_lstm_dual/)

---

## 1. 동기 — 왜 1D CNN인가

### 1.1 현재 baseline의 정보 손실

[`Sources/vppm/baseline/features.py`](../baseline/features.py) 의 `FeatureExtractor.extract_features()` 는 한 supervoxel(8×8 픽셀 × **70 레이어** 고정)에서 21개 피처를 뽑을 때 **모두 z-방향으로 축약** 한다.

| 인덱스 | 피처군 | 레이어별 raw shape | 현재 aggregation | 정보 손실 정도 |
|:--:|:--|:--|:--|:--:|
| 0–1 | CAD 거리 (edge / overhang) | `(70, 2)` | CAD-가중 z-평균 | 중 |
| 2 | build_height | 스칼라 | z-블록 중심값 | 없음 (z-position 자체) |
| 3–10 | DSCNN 8 클래스 비율 | `(70, 8)` | CAD-가중 z-평균 | **높음** |
| 11–17 | 프린터 센서 7개 | `(70, 7)` | 단순 z-평균 | **높음** |
| 18 | laser_module | 스칼라 | part 단위 binary | 없음 |
| 19–20 | 스캔 (return_delay, stripe_boundaries) | `(70, 2)` | 단순 z-평균 | 중 |

**문제**: 두 supervoxel이 평균값은 같지만 z-방향 패턴이 다른 경우를 baseline은 구분 못 한다.
- 예 1: 모든 레이어에서 약한 키홀(seg_excessive_melting=0.05) → 평균 0.05
- 예 2: 한 레이어에서 강한 키홀(0.7), 나머지는 0 → 평균 0.05
- 두 경우의 인장 거동은 다를 가능성이 큼 (예 2는 국소 결함, 예 1은 분산 결함).

### 1.2 LSTM-Dual의 한계 ([LSTM_RESULTS.md](../../pipeline_outputs/experiments/vppm_lstm/LSTM_RESULTS.md))

vppm_lstm_dual 은 카메라 시퀀스(visible/0, visible/1)에 CNN-LSTM을 적용했지만, **21개 baseline 피처 자체는 여전히 평균값**으로 들어간다. → 카메라 외 신호(센서, DSCNN, 스캔)의 z-방향 변동도 모델이 활용하지 못함.

### 1.3 1D CNN을 쓰는 이유

- **국소 패턴 검출**: kernel=3 이면 "연속 3 레이어 동안 산소 튐 → DSCNN soot 증가 → 센서 온도 상승" 같은 인접 레이어 간 결합 패턴을 학습.
- **고정 길이(70 레이어)** 라서 padding 처리 불필요 → LSTM 보다 단순.
- **Translation invariance**: 결함이 z-방향 어디서 일어나도 패턴 자체를 검출.
- **빠르고 안정적**: 짧은 시퀀스(70)에서 LSTM 대비 학습이 빠르고 grad vanishing 위험 적음.

---

## 2. 데이터 구조 변경

### 2.1 현재 (baseline)

```
features.npz
└── features        : (N_sv, 21) float32      ← 레이어 평균된 값
```

### 2.2 제안 (1dcnn)

```
features_seq.npz
├── features_seq       : (N_sv, 70, 21) float32  ← 레이어별 raw
├── features_agg       : (N_sv, 21) float32      ← 기존 21 (fallback / 추가 입력)
├── valid_layer_mask   : (N_sv, 70) bool         ← 시편 영역 내 유효 레이어
├── sv_indices         : (N_sv, 3) int32         ← (ix, iy, iz)
├── sample_ids         : (N_sv,) int32
├── build_ids          : (N_sv,) int32
└── targets_*          : (N_sv,) float32
```

**예상 용량**: 6,299 시편 × 평균 ~수십 SV × 70 × 21 × 4byte ≈ **수 GB** (관리 가능).

### 2.3 어떤 피처를 시퀀스화하는가

| 인덱스 | 피처 | 시퀀스 포함? | 이유 |
|:--:|:--|:--:|:--|
| 0 | distance_from_edge | ✅ | 레이어별 EDT 결과가 다름 |
| 1 | distance_from_overhang | ✅ | 〃 |
| 2 | build_height | ❌ → 스칼라 유지 | 70 레이어 내 거의 선형 (1D CNN 입력으로 무의미) |
| 3–10 | DSCNN 8 클래스 | ✅ | **z-방향 변동이 핵심 신호** |
| 11–17 | 프린터 센서 7개 | ✅ | 시계열 그 자체 |
| 18 | laser_module | ❌ → 스칼라 유지 | part 단위 상수 |
| 19 | laser_return_delay | ✅ | 레이어별 스캔 패턴 |
| 20 | laser_stripe_boundaries | ✅ | 〃 |

→ **시퀀스 입력**: 19 채널 × 70 레이어 = `(70, 19)`
→ **스칼라 입력**: build_height + laser_module = `(2,)`

---

## 3. 모델 아키텍처

```
입력:
  feats_seq      (B, 19, 70)        [Conv1d 입력 형식: (B, C, L)]
  feats_scalar   (B, 2)              [build_height, laser_module]

1D CNN 인코더:
  Conv1d(19 → 64, kernel=3, padding=1) → BN → ReLU
  Conv1d(64 → 64, kernel=3, padding=1) → BN → ReLU
  Conv1d(64 → 32, kernel=3, padding=1) → BN → ReLU
  AdaptiveAvgPool1d(1) + AdaptiveMaxPool1d(1) → concat → (B, 64)
  Linear(64 → d_embed)               [d_embed ∈ {4, 8, 16}]

결합 MLP:
  concat[cnn_embed (d_embed), feats_scalar (2)] → (B, d_embed+2)
  Linear → 128 → ReLU → Dropout(0.1)
  Linear → 1                         [property별 head]
```

### 3.1 변형 옵션 (ablation)

| 변형 | 설명 | 우선순위 |
|:--|:--|:--:|
| **A: pure 1D CNN** | 위 아키텍처 그대로 | 🔥 main |
| B: + baseline-21 concat | `concat[cnn_embed, feats_agg(21), feats_scalar(2)]` — 평균 피처도 fallback으로 함께 입력 | 🔥 main |
| C: + 카메라 임베딩 (lstm_dual 결합) | `concat[cnn_embed, embed_v0, embed_v1, scalar]` — 풀 hierarchical | ⚪ 후속 |
| D: 1D CNN + LSTM | CNN으로 채널 압축 → LSTM으로 long-range | ⚪ 비교 |
| E: kernel size sweep | k ∈ {3, 5, 7} | ⚪ 보조 |

→ **A 와 B 를 먼저 비교**하여 1D CNN 단독으로 baseline-21을 능가하는지 확인. 만약 A < B 라면 시퀀스 신호가 약하다는 뜻.

### 3.2 Ablation 가설

| 비교 | 가설 | 검증 방법 |
|:--|:--|:--|
| baseline-21 vs 변형 A | 시퀀스 정보가 의미 있다면 A 가 RMSE ↓ | 5-fold RMSE ± std |
| 변형 A vs 변형 B | A가 B와 비슷 → CNN이 평균 정보를 자체 학습. A < B → CNN이 평균을 못 뽑음 (집계 후처리 필요) | 5-fold RMSE 비교 |
| LSTM-Dual vs 변형 C | 카메라 + 21 시퀀스가 카메라 단독보다 좋은가 | RMSE per build (B1.4/B1.5 주목) |

---

## 4. 디렉토리 / 파일 구조

```
Sources/vppm/1dcnn/                        ← 본 모듈 (코드 + PLAN.md)
├── PLAN.md                                ← 본 문서
├── __init__.py
├── config.py                              ← 1dcnn 전용 하이퍼파라미터
├── features_seq.py                        ← 레이어별 피처 추출 (FeatureExtractor 확장)
├── dataset.py                             ← (N, 70, 19) 로더 + 정규화
├── model.py                               ← VPPM_1DCNN 클래스 (변형 A/B/C 지원)
├── train.py                               ← 4 prop × 5 fold 학습 (vppm_lstm_dual/train.py 미러)
├── evaluate.py                            ← RMSE 산출 + plot
└── run.py                                 ← 진입점 (--phase {features, train, evaluate, all})

Sources/pipeline_outputs/experiments/vppm_1dcnn/   ← 산출물 디렉토리
├── features/
│   ├── features_seq.npz                   ← (N, 70, 19) raw 시퀀스
│   └── normalization.json                 ← 채널별 min/max
├── models/
│   ├── vppm_1dcnn_{YS,UTS,UE,TE}_fold{0..4}.pt
│   └── training_log.json
├── results/
│   ├── metrics_raw.json
│   ├── metrics_summary.json
│   ├── predictions_{YS,UTS,UE,TE}.csv
│   ├── per_build_rmse.json                ← 빌드별 RMSE 분해
│   └── plots/
└── experiment_meta.json
```

---

## 5. 구현 단계 (Phase)

### Phase 1 — 데이터 추출 (1–2일)

**목표**: 레이어별 피처 시퀀스 캐시 생성.

**작업**:
1. [`features.py`](../baseline/features.py) 의 4개 그룹 추출 함수를 **레이어 평균 직전 단계까지 분리**:
   - `_extract_cad_features_per_layer(layer)` → `(n_sv, 2)` per layer
   - `_extract_dscnn_features_per_layer(layer)` → `(n_sv, 8)` per layer
   - `_extract_temporal_per_layer(layer)` → `(n_sv, 7)` per layer (시편 영역에 broadcast)
   - `_extract_scan_features_per_layer(layer)` → `(n_sv, 2)` per layer
2. `Sources/vppm/1dcnn/features_seq.py::FeatureSequenceExtractor` 신설:
   - 각 z-블록에서 70 레이어를 반복 호출, `(n_sv, 70, 19)` 텐서 누적
   - 시편 영역 밖 레이어는 `valid_layer_mask=False` 로 표시
3. `Sources/vppm/1dcnn/run.py --phase features` → `features/features_seq.npz` 저장.

**검증**: 시퀀스의 레이어 평균이 기존 [`features.npz`](../../pipeline_outputs/features/all_features.npz) 와 **일치**해야 함 (단순평균 채널: 11–17, 19–20). CAD 가중평균(0–1)·DSCNN(3–10)은 마스크 가중치를 별도 보존 후 비교.

### Phase 2 — 모델 구현 + 학습 (1일)

1. `model.py::VPPM_1DCNN` 작성. `variant ∈ {"A", "B"}` 인자로 두 구조 모두 지원.
2. `dataset.py::build_normalized_dataset_seq()`:
   - 채널별 min-max 정규화 ([-1, 1])
   - `valid_layer_mask=False` 영역은 0 으로 패딩 (또는 채널평균)
   - 정규화 파라미터 JSON 저장
3. `train.py`: [`vppm_lstm_dual/train.py`](../lstm_dual/train.py) 의 `train_all()` 구조 그대로 차용. 데이터 입력만 시퀀스로 교체.
4. K-fold 분할은 [`Sources/vppm/common/dataset.py::create_cv_splits()`](../common/dataset.py) 재사용 (sample-wise 5-fold).

### Phase 3 — 평가 + 비교 (0.5일)

1. `evaluate.py`: 4 property × 5 fold RMSE 산출.
2. **빌드별 RMSE 분해** (B1.1~B1.5) — vppm_lstm_dual 에서 누락되었던 분석을 본 실험에서 표준화.
3. baseline / lstm-single / lstm-dual / 1dcnn-A / 1dcnn-B 5종 비교 표를 `results/comparison.md` 에 기록.
4. correlation plot, fold별 학습 곡선 시각화.

### Phase 4 — 후속 ablation (선택)

- d_embed sweep: {4, 8, 16}
- kernel sweep: {3, 5, 7}
- 변형 C (1dcnn + lstm_dual 카메라 임베딩) 통합 모델
- per-build 분석에서 B1.4/B1.5 (스패터/리코터) 빌드 RMSE 가 baseline 대비 얼마나 줄어드는지 확인

---

## 6. 학습 / 평가 설정

| 항목 | 값 | 비고 |
|:--|:--|:--|
| 시퀀스 입력 채널 | 19 | DSCNN 8 + 센서 7 + CAD 2 + 스캔 2 |
| 시퀀스 길이 | 70 | SV_Z_LAYERS 고정 |
| 스칼라 입력 | 2 | build_height, laser_module |
| d_embed | **8** (default) | 4/16 ablation |
| Conv1d 채널 | 19 → 64 → 64 → 32 | kernel=3, padding=1 |
| Pooling | AdaptiveAvg + AdaptiveMax concat | (B, 64) |
| MLP hidden | 128 | dropout=0.1 |
| 손실 | L1Loss | baseline / lstm 과 동일 |
| Optimizer | Adam, lr=1e-3, wd=0 | |
| Batch size | 256 | |
| Early stop | patience=50, max=5000 epoch | |
| Grad clip | 1.0 | |
| K-fold | 5 (sample-wise) | seed=42 |
| Property | YS, UTS, UE, TE 각각 별도 모델 | |

---

## 7. 성공 기준

| 지표 | 합격선 | 코멘트 |
|:--|:--|:--|
| **변형 A (1DCNN 단독)** vs baseline | 모든 property 에서 평균 RMSE ↓ | 시퀀스 정보가 실재함을 입증 |
| **변형 A** vs LSTM-single (22-feat) | UTS/UE/TE RMSE 가 ±std 안 또는 더 좋음 | baseline 21 시퀀스만으로 카메라 임베딩 수준 도달 |
| **빌드별 RMSE** (B1.4 / B1.5) | baseline 대비 ≥ 10% 감소 | 스패터/리코터 빌드에서 시퀀스 신호가 핵심일 것이라는 가설 검증 |
| 학습 안정성 | 5 fold std/mean < 7% | LSTM-dual 수준 |

**실패 시나리오 대응**:
- A ≈ baseline → 시퀀스 신호가 weak. 변형 B (concat baseline-21) 로 재시도. 그래도 같으면 21 피처 자체의 표현력 한계.
- 학습 불안정 → BatchNorm 을 LayerNorm 으로 교체, dropout 추가.
- 메모리 폭주 → `features_seq.npz` 를 빌드별 분리 + memmap 로딩.

---

## 8. 위험 / 주의사항

| 위험 | 대응 |
|:--|:--|
| Phase 1 캐시 생성 시간 (HDF5 5빌드 재파싱) | 기존 `extract_features` 가 60–90 분 / 빌드 → 시퀀스 버전은 ~1.5× 예상. 빌드별 병렬 처리 가능. |
| CAD 가중평균을 1D CNN에서 재현 못 할 가능성 | `valid_layer_mask` 와 `cad_count_per_layer` 를 별도 채널로 추가 (총 20–21 채널). |
| 센서 7개가 슈퍼복셀 단위로 동일 (xy 균일) | 같은 z-블록 내 모든 SV 가 같은 시계열을 봄 → 정보 추가 효과는 크지 않을 수 있음. **DSCNN/스캔 채널이 주효할 것**. |
| 모델 비교 공정성 | 동일 fold split, 동일 normalization 정책, 동일 early-stop 사용 (모두 [`common/dataset.py`](../common/dataset.py) 경유). |

---

## 9. 일정 요약

| Phase | 내용 | 예상 기간 |
|:--:|:--|:--:|
| 1 | 레이어별 피처 추출 + 캐시 | 1–2 일 |
| 2 | 모델 구현 + 5-fold 학습 (4 prop) | 1 일 |
| 3 | 평가 + 비교표 | 0.5 일 |
| 4 | (선택) ablation / 통합 모델 | +1–2 일 |

**총 ~3–5 일** 안에 baseline-21 vs 1D CNN 비교 결과가 나옴.

---

## 10. 다음 액션

1. 본 PLAN.md 검토 (사용자 승인)
2. Phase 1 시작: `Sources/vppm/1dcnn/features_seq.py` 신설
3. 캐시 생성 → 시퀀스 평균이 기존 21 피처와 일치 검증
4. 모델 학습 → 결과 비교
