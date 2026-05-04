# VPPM-LSTM 모델 설명

> **확장 목표**: Baseline VPPM(21 핸드크래프트 피처 → MLP) 에 **슈퍼복셀별 CNN+LSTM 시계열 임베딩 1개**를 추가해 22(=21+1)-dim 입력으로 학습한다.
>
> **모델 위치**: `Sources/pipeline_outputs/experiments/vppm_lstm/models/`
>
> **결과 위치**: `Sources/pipeline_outputs/experiments/vppm_lstm/results/`

각 슈퍼복셀(SV)은 **xy 1×1 mm × z 3.5 mm (= 70 레이어)** 의 3D 부피이며, 그 SV 가 점유한 z-범위 동안의 **레이어별 카메라 이미지(8×8 크롭)** 를 시간순으로 쌓아 가변 길이 시퀀스를 만든다. 이 시퀀스를 CNN(per-frame) + LSTM 으로 압축해 단일 스칼라(1-dim) 임베딩을 얻고, baseline 의 21 피처와 concat 해 기존 VPPM MLP 로 회귀한다. 데이터 파이프라인 전반은 [`Sources/vppm/README.md`](../README.md), 자세한 설계 결정은 [`PLAN.md`](PLAN.md) 참조.

---

## 1. 전체 아키텍처

```
┌────────────────────────────────────────────────────────────┐
│ Per-SV 입력                                                  │
│   stack:    (T_max=70, H=8, W=8) float16  ─ zero-padded     │
│   length:   T_sv ∈ [1, 70]              ─ 실제 시퀀스 길이    │
│   feat21:   (21,) float32               ─ baseline 피처      │
└────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│ FrameCNN  (per-frame, shared weights)                       │
│   Conv3×3(1→16) → BN → ReLU                                  │
│   Conv3×3(16→32) → BN → ReLU                                 │
│   AdaptiveAvgPool(1) → Linear(32 → 32)                       │
│   ⇒ (T_max, d_cnn=32)                                        │
└────────────────────────────────────────────────────────────┘
        │
        ▼ pack_padded_sequence(lengths, enforce_sorted=False)
┌────────────────────────────────────────────────────────────┐
│ LSTM (forward, 1-layer, hidden=16)                          │
│   packed (T_sv, 32) → h_n[-1] : (16,)                        │
│   Linear(16 → d_embed=1) ⇒ embed (1,)                        │
└────────────────────────────────────────────────────────────┘
        │
        ▼ concat
┌────────────────────────────────────────────────────────────┐
│ 결합 MLP (= baseline VPPM 동일 골격, 입력 22)                  │
│   Linear(22 → 128) → ReLU → Dropout(0.1)                     │
│   Linear(128 → 1) ⇒ ŷ ∈ ℝ ([-1, 1] 정규화 공간)              │
└────────────────────────────────────────────────────────────┘
```

| 단계 | 입력 | 출력 | 파라미터 |
|---|---|---|---|
| FrameCNN | `(B*T, 1, 8, 8)` | `(B*T, 32)` | ~5,900 |
| LSTM (1-layer, fwd) | `(B, T, 32)` + lengths | `h_n[-1] : (B, 16)` | 3,200 |
| Linear (proj) | `(B, 16)` | `(B, 1)` | 17 |
| 결합 MLP | `(B, 22)` | `(B, 1)` | 2,945 |
| **합계** | | | **~12 k** |

> Baseline (2,945) 의 약 4 배. CNN/LSTM 추가 비용은 작지만 SV 당 70×8×8 텐서 forward 가 추가되어 **GPU 학습 권장**.

---

## 2. 입력 데이터

### 2.1 baseline 21 피처
[`baseline/MODEL.md` §4](../baseline/MODEL.md#4-입력-피처-21개) 참조. 변경 없이 그대로 재사용 (`Sources/pipeline_outputs/features/all_features.npz`).

### 2.2 SV 별 카메라 이미지 시퀀스 (신규)

| 항목 | 값 | 설명 |
|---|---|---|
| 채널 | `slices/camera_data/visible/0` | 용융 직후 카메라 (분말 도포 후 채널 `1` 은 미사용) |
| 크롭 크기 | 8 × 8 pixels | `SV_XY_PIXELS ≈ 7.52` → 8 픽셀로 정수화 |
| 최대 길이 | T_max = 70 | SV z-범위(3.5 mm / 0.05 mm) |
| 실제 길이 T_sv | 1 ≤ T_sv ≤ 70 | "유효 레이어" 만 포함 — 가변 길이 |
| dtype | `float16` (uint8 / 255) | 메모리 절약 |

### 2.3 "유효 레이어" 정의

하나의 SV `(ix, iy, iz)` 의 z-범위 `[l0, l1)` (보통 70 레이어) 에서, 다음 조건을 만족하는 레이어 L 만 시퀀스에 포함:

  **`(part_ids[L][r0:r1, c0:c1] > 0).any() == True`**

= 그 레이어에서 SV xy 영역에 CAD/파트가 실제로 존재함.

→ 빌드 시작 전 / 파트 상단 위 / 오버행 등 "파트가 아직 없는" 레이어는 자연스럽게 제외됨.
→ 결과적으로 빌드의 **시간적·공간적 국소 컨텍스트** 만 LSTM 에 전달.

### 2.4 인덱싱 일관성

L1 캐시 (`crop_stacks_B1.x.h5`) 의 SV 순서는 baseline `find_valid_supervoxels()` 가 반환한 `voxel_indices` 와 **1:1 매칭**. `dataset.load_lstm_dataset()` 가 `features.npz` 의 `build_ids` 정렬 순서와 캐시 concat 순서를 검증하므로 SV 인덱싱이 어긋나지 않는다.

---

## 3. 모델 모듈 — `model.py`

### 3.1 `FrameCNN` — 단일 프레임 임베딩

```python
class FrameCNN(nn.Module):
    Conv2d(1 → 16, k=3, pad=1) + BN + ReLU
    Conv2d(16 → 32, k=3, pad=1) + BN + ReLU
    AdaptiveAvgPool2d(1)              # 8×8 → 1×1
    Linear(32 → 32)                   # d_cnn
```

- 8×8 입력에 stride/pool 을 과하게 쓰면 정보가 죽음 → padding 만 사용.
- 모든 프레임이 같은 CNN 가중치를 공유 (shared per-frame encoder).

### 3.2 `VPPM_LSTM` — 결합 모델

```python
class VPPM_LSTM(nn.Module):
    self.cnn         = FrameCNN(d_cnn=32)
    self.lstm        = LSTM(input=32, hidden=16, layers=1, bidirectional=False)
    self.embed_proj  = Linear(d_lstm_out → 1)            # bi 면 32→1, fwd 면 16→1
    self.fc1         = Linear(21 + 1 → 128)
    self.dropout     = Dropout(0.1)
    self.fc2         = Linear(128 → 1)
```

#### Forward 흐름

```python
# stacks: (B, T_max=70, 8, 8), lengths: (B,), feats21: (B, 21)
B, T, H, W = stacks.shape

# (1) per-frame CNN
x = stacks.view(B*T, 1, H, W)
x = cnn(x)                    # (B*T, 32)
x = x.view(B, T, 32)

# (2) 가변 길이 LSTM — pack 으로 padding 무시 + 실제 last hidden 자동 추출
packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
_, (h_n, _) = lstm(packed)
h_last = h_n[-1]              # (B, 16)  ← 각 시퀀스의 실제 마지막 step

# (3) 1-dim 임베딩으로 사영
embed = embed_proj(h_last)    # (B, 1)

# (4) baseline 21 피처와 concat → MLP
x = torch.cat([feats21, embed], dim=1)   # (B, 22)
x = relu(fc1(x))
x = dropout(x)
return fc2(x)                  # (B, 1)
```

> **핵심 트릭**: `pack_padded_sequence` 를 쓰면 LSTM 이 zero-padding 프레임을 무시하고, `h_n[-1]` 이 **각 샘플의 실제 마지막 유효 step hidden** 을 자동으로 가리킨다. 직접 `out[range(B), lengths-1]` 인덱싱할 필요가 없음.
>
> bidirectional 옵션 시 `h_n` 의 마지막 layer forward + backward 두 방향을 concat → `(B, 32)`. `embed_proj` 가 입력 차원을 자동 조정하여 1-dim 임베딩 출력은 동일.

---

## 4. 하이퍼파라미터 (`common/config.py::LSTM_*`)

| 항목 | 값 | 비고 |
|---|---|---|
| `LSTM_CAMERA_CHANNEL` | 0 (용융 직후) | DSCNN 결과맵은 21 피처에 이미 포함 → 원시 시각 채널만 사용 |
| `LSTM_T_MAX` | 70 | SV z-layers (3.5 / 0.05) |
| `LSTM_CROP_H/W` | 8 | SV xy 픽셀 (≈ 7.52 → 8) |
| `LSTM_CNN_CH1/CH2` | 16 / 32 | Conv 두 단계 채널 수 |
| `LSTM_CNN_KERNEL` | 3 | 3×3 (pad=1) |
| `LSTM_D_CNN` | 32 | 프레임당 임베딩 차원 (LSTM 입력) |
| `LSTM_D_HIDDEN` | 16 | LSTM hidden state |
| `LSTM_NUM_LAYERS` | 1 | 단층 |
| `LSTM_BIDIRECTIONAL` | False | forward 만 (시간축 = 빌드 진행 방향) |
| `LSTM_D_EMBED` | 1 | "22 features" 요구사항 충족 |
| `LSTM_LR` | 1e-3 | baseline 동일 |
| `LSTM_BATCH_SIZE` | 256 | (B, 70, 8, 8) 텐서 메모리 고려 |
| `LSTM_MAX_EPOCHS` | 5000 | early-stop 으로 보통 수백 epoch 에서 종료 |
| `LSTM_EARLY_STOP_PATIENCE` | 50 | val loss 50 epoch 무개선 시 중단 |
| `LSTM_GRAD_CLIP` | 1.0 | LSTM 폭발 방지 |
| `LSTM_WEIGHT_DECAY` | 0.0 | baseline 동일 |

> 가중치 초기화: 결합 MLP 와 `embed_proj` 만 `N(0, σ=0.1)` (baseline 과 동일). CNN/LSTM 은 PyTorch 기본 (Xavier-uniform 류) — 너무 작은 std 면 LSTM 활성이 죽어 학습 안됨.

---

## 5. 학습 전략

| 결정 | 선택 | 이유 |
|---|---|---|
| 학습 방식 | **End-to-end joint** | CNN+LSTM+MLP 를 한 옵티마이저로 통합 학습. 사용자 요구에 부합 |
| 손실 | `L1Loss` (MAE) | baseline 동일, 측정 이상치 강건 |
| 옵티마이저 | Adam (β=0.9/0.999, eps=1e-4) | baseline 동일 |
| 그래디언트 클립 | norm ≤ 1.0 | LSTM 학습 안정성 |
| K-fold | **sample-wise** 5-fold | 같은 시편 SV 가 train/val 에 걸치면 데이터 누출 |
| 결합 방식 | **concat** (`feat21 ⊕ embed1`) | 사용자 발언 "더한다" 를 concat 으로 해석 |

> **2-stage 폴백**: joint 학습이 불안정하면 (1) CNN+LSTM+임시Linear 로 임베딩만 사전 학습 → (2) 임베딩을 `.npz` 로 저장 → (3) baseline 의 22-feat MLP 만 재학습. 코드는 동일 모듈 재사용 가능. 자세한 내용은 [`PLAN.md` §6](PLAN.md#6-학습-전략) 참조.

---

## 6. 평가 — `evaluate.py`

baseline `evaluate_fold/save_metrics/plot_*` 함수를 그대로 import 해 사용. 차이는 모델 forward 시그니처뿐.

1. 각 fold 의 val split 에 대해 `vppm_lstm_{short}_fold{k}.pt` 로드 → forward → 정규화된 예측.
2. `denormalize(pred, t_min, t_max)` 로 MPa/% 복원.
3. **per-sample min 집계** — 한 시편의 여러 SV 예측 중 최솟값을 그 시편의 예측으로 사용 (논문 Section 3.1, "가장 약한 지점이 파단을 결정").
4. RMSE(fold) 계산 → 5-fold 평균 ± 표준편차.
5. Naive baseline (전체 평균 예측) 대비 reduction% 계산.
6. `correlation_plots.png`, `scatter_plot_uts.png`, `predictions_*.csv`, `metrics_summary.json` 저장.

---

## 7. 산출물

```
Sources/pipeline_outputs/experiments/vppm_lstm/
├── cache/
│   ├── crop_stacks_B1.{1..5}.h5    # SV 별 (70, 8, 8) padded float16 + lengths
│   └── (총 ~150 MB gzip 후, 36k SV 기준)
├── models/
│   ├── vppm_lstm_{YS|UTS|UE|TE}_fold{0..4}.pt   # 20 모델
│   └── training_log.json                          # fold별 val loss, epochs
├── features/
│   └── normalization.json          # 22-dim 기준 (단, embed 는 학습 중 산출이라 21-dim 까지 정규화 파라미터)
├── results/
│   ├── metrics_summary.json
│   ├── metrics_raw.json
│   ├── predictions_{YS|UTS|UE|TE}.csv
│   ├── correlation_plots.png       # 4 속성 2D 히스토그램 (논문 Figure 17 형식)
│   └── scatter_plot_uts.png
└── experiment_meta.json            # 학습 시점의 config 스냅샷
```

---

## 8. 실행 방법

```bash
# 0) 사전조건: baseline features 가 이미 추출되어 있어야 함
ls Sources/pipeline_outputs/features/all_features.npz   # 존재 확인

# 1) 일괄 실행 (cache → train → evaluate)
python -m Sources.vppm.lstm.run --all

# 2) 단계별 실행
python -m Sources.vppm.lstm.run --phase cache       # L1 캐시 빌드 (~30 분)
python -m Sources.vppm.lstm.run --phase train       # 학습 (4 props × 5 folds, GPU 권장)
python -m Sources.vppm.lstm.run --phase evaluate    # 평가만 (models/*.pt 필요)

# 3) Smoke test (epochs=20, patience=10)
python -m Sources.vppm.lstm.run --all --quick

# 4) Docker 실행 — 백그라운드, SSH 끊겨도 계속
docker compose -f docker/lstm/docker-compose.yml up -d --build
```

### 추론 코드 예시

```python
import torch
from Sources.vppm.lstm.model import VPPM_LSTM
from Sources.vppm.common.dataset import denormalize, load_norm_params

model = VPPM_LSTM()
model.load_state_dict(torch.load(
    "Sources/pipeline_outputs/experiments/vppm_lstm/models/vppm_lstm_UTS_fold0.pt",
    weights_only=True,
))
model.eval()

# 입력: 정규화된 feat21 + padded stack + length
feats21 = torch.randn(1, 21)
stack = torch.randn(1, 70, 8, 8)
length = torch.tensor([42])         # 실제 시퀀스 길이 (cpu int64)

with torch.no_grad():
    pred_norm = model(feats21, stack, length).item()

norm = load_norm_params(
    "Sources/pipeline_outputs/experiments/vppm_lstm/features/normalization.json"
)
pred_mpa = denormalize(
    pred_norm,
    norm["target_min"]["ultimate_tensile_strength"],
    norm["target_max"]["ultimate_tensile_strength"],
)
```

---

## 9. baseline 과의 비교

| 구분 | VPPM (baseline) | VPPM-LSTM |
|---|---|---|
| 입력 | 21 피처 (scalar) | 21 피처 + (T≤70, 8×8) 가변 길이 이미지 시퀀스 |
| DSCNN 활용 | 스칼라 평균 8개 (피처 4–11) | 스칼라 8개 + 원시 카메라 이미지 (시간축 정보 추가) |
| 모델 | `FC(21→128→1)` | `CNN + LSTM` 으로 1-dim 임베딩 → `FC(22→128→1)` |
| 파라미터 | 2,945 | ~12,000 (CNN+LSTM 포함) |
| 추가 산출물 | 없음 | `cache/crop_stacks_B1.x.h5` (~150 MB gzip) |
| 학습 디바이스 | CPU 가능 | **GPU 권장** ((B, 70, 8, 8) 텐서 forward) |

자세한 baseline 결과는 [`baseline/MODEL.md`](../baseline/MODEL.md) 참조.

---

## 10. 파일 맵

```
Sources/vppm/lstm/
├── PLAN.md             # 설계 결정 / Open Questions
├── MODEL.md            # ← 본 문서 (모델 설명)
├── FLOW.md             # 코드 동작 흐름 설명
├── __init__.py
├── crop_stacks.py      # Phase L1: HDF5 → SV별 크롭 시퀀스 캐시 빌드
├── dataset.py          # VPPMLSTMDataset, load/build, collate_fn
├── model.py            # FrameCNN, VPPM_LSTM (본 문서의 §3)
├── train.py            # train_single_fold, train_all (4 props × 5 folds)
├── evaluate.py         # _evaluate_fold, evaluate_all
└── run.py              # CLI 진입점 — phase=cache|train|evaluate|all

docker/lstm/
├── Dockerfile          # LSTM 전용 (entrypoint 가 ORNL/features/output 검증)
├── docker-compose.yml  # cache → train → evaluate 순차 실행
├── entrypoint.sh
└── .env                # UID_GID, NVIDIA_VISIBLE_DEVICES, LSTM_PHASE, LSTM_EXTRA
```

---

## 11. 한계 / 향후 개선

- **단일 카메라 채널만 사용**: `visible/1` (분말 도포 후) 도 함께 쓰면 입력 채널이 (T, 2, 8, 8) 이 되어 더 풍부 — 캐시 크기 2 배.
- **임베딩 차원 1**: 사용자 요구 "22 features" 충족용. 표현력을 높이려면 `--d-embed 16` 으로 37-feat MLP 가능.
- **forward LSTM**: 빌드 진행 방향(시간축) 자연스럽지만, bidirectional 이 임베딩 표현력에 유리할 수 있음.
- **유효 레이어 기준**: 현재 "한 픽셀이라도 part_ids > 0" — 더 엄격한 비율 임계값 (예: ≥ 10%) 으로 노이즈 줄일 여지.
