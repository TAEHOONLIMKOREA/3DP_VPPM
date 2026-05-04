# VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-Sensor-1 실험 계획

> **한 줄 요약**: [`lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4`](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md) 의 sensor 분기를 **per-field 1D-CNN(7×4=28-dim) → 단일 multi-channel LSTM(d_embed_s=1)** 으로 교체. 다른 6개 분기(카메라 v0/v1 d=16, DSCNN d=8, CAD spatial-CNN+LSTM d=8, Scan spatial-CNN+LSTM d=8, 정적 2-feat) 와 캐시·하이퍼파라미터는 모두 동일하게 유지하는 **controlled 비교 실험**. MLP 입력 차원은 86 → **59**.

- **실험 이름**: `lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1`
- **결과 위치 (예정)**: `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/`
- **1차 비교 대상**: [`vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4`](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/) — sensor 시간 인코더만 1D-CNN ↔ LSTM 으로 swap (단일 변수 통제)
- **2차 비교 대상**: [`vppm_lstm_dual_img_4_sensor_7_dscnn_8`](../lstm_dual_img_4_sensor_7_dscnn_8/) — 기존 sensor LSTM(d=7) 대비 d 축소 효과
- **3차 비교 대상**: [`vppm_lstm_dual_4`](../lstm_dual_4/) — image-only LSTM (sensor 평균)
- **학습 일시 (계획)**: 2026-05-05 ~

---

## 1. 동기 — sensor 시간 인코더 1D-CNN ↔ LSTM 비교

기존 fullstack 실험 ([`lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4`](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md#11-주의-사항)) 의 sensor 분기는 **per-field 1D-CNN + AdaptiveAvgPool** 구조였다. 이 선택의 장단:

| 측면 | per-field 1D-CNN (기존) | 단일 multi-channel LSTM (본 실험) |
|:--|:--|:--|
| 채널간 상호작용 | 없음 (필드별 독립) | 있음 (LSTM hidden 이 7채널 동시 인코딩) |
| 시간축 압축 | AdaptiveAvgPool — 패딩 0 평균 섞임 (패딩 30%면 신호 ~0.7× 희석) | packed LSTM — 패딩 영역 미참조, 시간 종단까지 stateful |
| 표현력 | 7 필드 × d_per_field=4 = **28 dim** | d_embed_s=1 — 단일 스칼라 |
| 파라미터 | ~15k (필드별 conv ×7) | ~0.7k (단일 LSTM, in=7, hid=16, proj 16→1) |

→ 두 인코더 차이는 **(a) 채널간 상호작용 유무 + (b) 패딩 처리 + (c) 표현력 차원** 세 축. 본 실험은 fullstack 86 모델의 sensor 분기만 교체해 위 차이가 최종 RMSE 에 어떤 방향/크기로 영향을 주는지 확인.

### 왜 d_embed_s = 1 인가

- **명명 규약 일치**: 다른 분기 (`dscnn_8` = d_embed_d=8, `cad_8` = d_embed_c=8, `scan_8` = d_embed_sc=8, `sensor_7` = d_embed_s=7) 와 동일하게 suffix 가 d_embed 값. `sensor_1` ⇒ **d_embed_s = 1**.
- **압축 한계 테스트**: 28 → 7 → 1 로 sensor 표현 차원을 단계적으로 축소하면서 정보 임계점을 찾는 ablation. 만약 d=1 으로도 RMSE 가 기존 실험과 거의 같다면 sensor 의 시간 정보는 인장 특성 예측에 미미하다는 의미 (averaged 21-feat 으로 충분).
- **파라미터 절감**: fullstack 대비 sensor 분기 (~15k) + MLP 입력 (86→59 → fc1 22k→15k) 합쳐 약 22k 절감 → 6,373 SV / ~110k 파라미터로 SV/param 비율 ~58 (fullstack ~50) 약간 개선. 과적합 위험 소폭 완화.

### 누적 실험 변천

| 실험 | sensor 처리 | sensor 출력 dim | 채널간 |
|:--|:--|:--:|:--:|
| `vppm` (21-feat MLP) | 70-layer 단순평균 (스칼라) | 7 (값 자체) | — |
| `vppm_lstm_dual_4` | 위와 동일 (이미지만 LSTM) | 7 | — |
| `vppm_lstm_dual_img_4_sensor_7` | **단일 multi-ch LSTM**, d=7 | 7 | ✅ |
| `vppm_lstm_dual_img_4_sensor_7_dscnn_8` | 위와 동일 | 7 | ✅ |
| `vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4` | **per-field 1D-CNN**, 7×4=28 | 28 | ❌ |
| **본 실험** | **단일 multi-ch LSTM**, **d=1** | **1** | ✅ |

---

## 2. 데이터 흐름 — 7 분기 통합 (sensor 분기만 LSTM 으로 교체)

```
stack_v0    (B, T, 1, 8, 8)     ──[CNN+LSTM+proj(d=16)]────────────> embed_v0   (B, 16)
stack_v1    (B, T, 1, 8, 8)     ──[CNN+LSTM+proj(d=16)]────────────> embed_v1   (B, 16)
sensors     (B, T, 7)           ──[multi-ch LSTM+proj(d=1)]────────> embed_s    (B, 1)   ★ 변경
dscnn       (B, T, 8)           ──[LSTM+proj(d=8)]─────────────────> embed_d    (B, 8)
cad_patch   (B, T, 2, 8, 8)     ──[CNN(in=2)+LSTM+proj(d=8)]───────> embed_c    (B, 8)
scan_patch  (B, T, 2, 8, 8)     ──[CNN(in=2)+LSTM+proj(d=8)]───────> embed_sc   (B, 8)
feat_static (B, 2)              ── (build_height, laser_module) ──> feat2      (B, 2)

   feat2 ⊕ embed_v0 ⊕ embed_v1 ⊕ embed_s ⊕ embed_d ⊕ embed_c ⊕ embed_sc
   = 2 + 16 + 16 + 1 + 8 + 8 + 8 = 59
                     │
                     ▼
          MLP(59 → 256 → 128 → 64 → 1)
```

### 표기

| 표기 | 의미 | 값 |
|:--:|:--|:--|
| **B** | Batch | 256 (`config.LSTM_BATCH_SIZE`) |
| **T** | Time steps — SV 활성 layer 수 (가변) | ≤ 70 |
| 7 (sensors) | sensor 채널 수 | `len(TEMPORAL_FEATURES)` |
| 8 (dscnn) | DSCNN 클래스 수 | `len(DSCNN_FEATURE_MAP)` |
| 1 (embed_s) | sensor LSTM proj 출력 | **신규 d_embed_s = 1** |

> **시퀀스 길이 공통**: 7 분기 중 시퀀스 입력 6종 (v0/v1/sensor/dscnn/cad_patch/scan_patch) 의 `lengths` 는 모두 동일. fullstack 과 동일 규칙 — 본 실험은 캐시 재사용으로 `lengths` 일치가 자동 보장됨.

---

## 3. 캐시 전략 — **신규 빌드 0**

| 캐시 | 출처 | 재사용 여부 |
|:--|:--|:--:|
| `crop_stacks_{B}.h5` (v0) | [`Sources/vppm/lstm/crop_stacks.py`](../lstm/crop_stacks.py) | ✅ |
| `crop_stacks_v1_{B}.h5` (v1) | [`Sources/vppm/lstm_dual/crop_stacks_v1.py`](../lstm_dual/crop_stacks_v1.py) | ✅ |
| `sensor_stacks_{B}.h5` | [`Sources/vppm/lstm_dual_img_4_sensor_7/cache_sensor.py`](../lstm_dual_img_4_sensor_7/cache_sensor.py) | ✅ |
| `dscnn_stacks_{B}.h5` | [`Sources/vppm/lstm_dual_img_4_sensor_7_dscnn_8/cache_dscnn.py`](../lstm_dual_img_4_sensor_7_dscnn_8/cache_dscnn.py) | ✅ |
| `cad_patch_stacks_{B}.h5` | [`lstm_dual_img_16_..._sensor_4/cache_cad_patch.py`](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache_cad_patch.py) | ✅ |
| `scan_patch_stacks_{B}.h5` | [`lstm_dual_img_16_..._sensor_4/cache_scan_patch.py`](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache_scan_patch.py) | ✅ |

→ **모든 캐시가 기존 실험 산출물 재사용**. 본 실험은 신규 캐시 빌드 없이 model.py + dataset.py + train/evaluate/run 만 추가하면 됨.

> 본 실험을 돌리기 전 사전조건: 기존 fullstack(`lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4`) 의 cad_patch / scan_patch 캐시가 5빌드 모두 존재해야 함. compose 의 사전조건 체크에 명시.

---

## 4. 모델 변경 (`model.py`)

### 4.1 sensor 분기 — `_SensorLSTMBranch` 재사용

기존 [`Sources/vppm/lstm_dual_img_4_sensor_7/model.py`](../lstm_dual_img_4_sensor_7/model.py#L23-L53) 의 `_SensorLSTMBranch` 를 그대로 import. 차이는 생성자 인자 (`d_embed=1`).

```python
from ..lstm_dual_img_4_sensor_7.model import _SensorLSTMBranch
```

> 단일 multi-channel LSTM. `pack_padded_sequence` → 마지막 hidden → `Linear(d_hidden, d_embed=1)`. fullstack 의 `_PerFieldConv1DBranch` (필드별 7개 1D-CNN + AdaptiveAvgPool) 와 정확히 1:1 swap.

### 4.2 다른 분기 — fullstack 그대로 재사용

```python
from ..lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.model import (
    _LSTMBranch,           # 카메라 v0/v1 + CAD-patch + Scan-patch (in_channels 일반화)
    _GroupLSTMBranch,      # DSCNN
    FrameCNN,              # in_channels 파라미터화 버전
)
```

### 4.3 메인 모델

```python
class VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_Sensor_1(nn.Module):
    def __init__(self,
                 # 카메라 (d_embed 16, in_channels=1)
                 d_cnn=config.LSTM_D_CNN,
                 d_hidden_cam=config.LSTM_FULL59_D_HIDDEN_CAM,        # 16
                 d_embed_v0=config.LSTM_FULL59_D_EMBED_V0,            # 16
                 d_embed_v1=config.LSTM_FULL59_D_EMBED_V1,            # 16
                 # sensor — 단일 multi-channel LSTM (d=1)
                 n_sensor_ch=config.LSTM_FULL59_N_SENSOR_CH,          # 7
                 d_hidden_s=config.LSTM_FULL59_D_HIDDEN_S,            # 16
                 d_embed_s=config.LSTM_FULL59_D_EMBED_S,              # 1
                 # DSCNN — 다채널 스칼라 시퀀스 LSTM
                 n_dscnn_ch=config.LSTM_FULL59_N_DSCNN_CH,            # 8
                 d_hidden_d=config.LSTM_FULL59_D_HIDDEN_D,            # 16
                 d_embed_d=config.LSTM_FULL59_D_EMBED_D,              # 8
                 # CAD spatial-CNN+LSTM
                 n_cad_ch=config.LSTM_FULL59_N_CAD_CH,                # 2
                 d_cnn_c=config.LSTM_FULL59_D_CNN_C,                  # 32
                 d_hidden_c=config.LSTM_FULL59_D_HIDDEN_C,            # 16
                 d_embed_c=config.LSTM_FULL59_D_EMBED_C,              # 8
                 # Scan spatial-CNN+LSTM
                 n_scan_ch=config.LSTM_FULL59_N_SCAN_CH,              # 2
                 d_cnn_sc=config.LSTM_FULL59_D_CNN_SC,                # 32
                 d_hidden_sc=config.LSTM_FULL59_D_HIDDEN_SC,          # 16
                 d_embed_sc=config.LSTM_FULL59_D_EMBED_SC,            # 8
                 # 결합 MLP — 59 → 256 → 128 → 64 → 1
                 mlp_hidden=config.LSTM_FULL59_MLP_HIDDEN,            # (256, 128, 64)
                 dropout=config.DROPOUT_RATE):
        super().__init__()

        # 카메라 v0/v1 — spatial-CNN+LSTM (in_channels=1, d=16)
        self.branch_v0 = _LSTMBranch(in_channels=1, d_cnn=d_cnn,
                                     d_hidden=d_hidden_cam, d_embed=d_embed_v0)
        self.branch_v1 = _LSTMBranch(in_channels=1, d_cnn=d_cnn,
                                     d_hidden=d_hidden_cam, d_embed=d_embed_v1)
        # Sensor — 단일 multi-channel LSTM (★ 변경: 1D-CNN → LSTM)
        self.branch_sensor = _SensorLSTMBranch(
            n_channels=n_sensor_ch,
            d_hidden=d_hidden_s, d_embed=d_embed_s,
            num_layers=config.LSTM_NUM_LAYERS,
            bidirectional=config.LSTM_BIDIRECTIONAL,
        )
        # DSCNN — 스칼라 시퀀스 LSTM
        self.branch_dscnn = _GroupLSTMBranch(n_dscnn_ch, d_hidden_d, d_embed_d)
        # CAD — spatial-CNN+LSTM (in_channels=2)
        self.branch_cad = _LSTMBranch(in_channels=n_cad_ch, d_cnn=d_cnn_c,
                                       d_hidden=d_hidden_c, d_embed=d_embed_c)
        # Scan — spatial-CNN+LSTM (in_channels=2)
        self.branch_scan = _LSTMBranch(in_channels=n_scan_ch, d_cnn=d_cnn_sc,
                                        d_hidden=d_hidden_sc, d_embed=d_embed_sc)

        n_static = 2                                                            # build_height + laser_module
        n_total = (n_static
                   + d_embed_v0 + d_embed_v1
                   + d_embed_s                                                  # 1 (★ fullstack 28 → 1)
                   + d_embed_d + d_embed_c + d_embed_sc)
        # = 2 + 16 + 16 + 1 + 8 + 8 + 8 = 59
        h1, h2, h3 = mlp_hidden                                                 # (256, 128, 64)
        self.fc1 = nn.Linear(n_total, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, 1)
        self.dropout = nn.Dropout(dropout)
        self._init_mlp_weights()

    def _init_mlp_weights(self):
        projs = [self.branch_v0.proj, self.branch_v1.proj,
                 self.branch_sensor.proj,        # ★ 단일 proj (per-field 7개 → 1개)
                 self.branch_dscnn.proj, self.branch_cad.proj, self.branch_scan.proj]
        for m in (self.fc1, self.fc2, self.fc3, self.fc4, *projs):
            nn.init.normal_(m.weight, mean=0.0, std=config.WEIGHT_INIT_STD)
            nn.init.zeros_(m.bias)

    def forward(self,
                feats_static: torch.Tensor,                  # (B, 2)
                stacks_v0: torch.Tensor, stacks_v1: torch.Tensor,    # (B, T, 8, 8) 또는 (B, T, 1, 8, 8)
                sensors: torch.Tensor,                       # (B, T, 7)
                dscnn: torch.Tensor,                         # (B, T, 8)
                cad_patch: torch.Tensor,                     # (B, T, 2, 8, 8)
                scan_patch: torch.Tensor,                    # (B, T, 2, 8, 8)
                lengths: torch.Tensor) -> torch.Tensor:
        e_v0 = self.branch_v0(stacks_v0, lengths)            # (B, 16)
        e_v1 = self.branch_v1(stacks_v1, lengths)            # (B, 16)
        e_s  = self.branch_sensor(sensors, lengths)          # (B, 1)  ★
        e_d  = self.branch_dscnn(dscnn, lengths)             # (B, 8)
        e_c  = self.branch_cad(cad_patch, lengths)           # (B, 8)
        e_sc = self.branch_scan(scan_patch, lengths)         # (B, 8)
        x = torch.cat([feats_static, e_v0, e_v1, e_s, e_d, e_c, e_sc], dim=1)   # (B, 59)
        x = F.relu(self.fc1(x)); x = self.dropout(x)                            # → 256
        x = F.relu(self.fc2(x)); x = self.dropout(x)                            # → 128
        x = F.relu(self.fc3(x)); x = self.dropout(x)                            # → 64
        return self.fc4(x)                                                      # → 1
```

> forward 시그니처는 fullstack 과 **완전히 동일**. dataset.py 의 `__getitem__` / `collate_fn` 도 변경 불필요 — 본 실험에서는 fullstack 의 `dataset.py` 를 그대로 import 해서 사용하면 됨.

---

## 5. 데이터셋 (`dataset.py`)

본 실험은 **fullstack 의 `load_septet_dataset` / `build_normalized_dataset` / `VPPMLSTMSeptetDataset` 를 그대로 import 해서 사용**. 입력 형태가 동일 (sensors raw 시퀀스 + 다른 6 입력) 하므로 코드 중복 회피.

```python
from ..lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.dataset import (
    load_septet_dataset,
    build_normalized_dataset,
    VPPMLSTMSeptetDataset,
)
```

> 정규화 통계도 동일 — sensors 는 패딩 0 제외 per-channel min-max [-1, 1]. 캐시 attrs (cad inversion/mask, scan inversion=False) 검증 로직 그대로.

---

## 6. config.py 추가

```python
# ============================================================
# VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-Sensor-1
# (Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/PLAN.md)
# 짧은 별칭 prefix: LSTM_FULL59_  (정식 이름이 너무 길어서 59-feat MLP 입력 차원 기반)
# fullstack(LSTM_FULL86_) 과 거의 동일. sensor 분기만 1D-CNN → 단일 multi-ch LSTM(d=1)
# ============================================================
LSTM_FULL59_D_HIDDEN_CAM = 16
LSTM_FULL59_D_EMBED_V0   = 16
LSTM_FULL59_D_EMBED_V1   = 16

# Sensor — 단일 multi-channel LSTM (★ fullstack 의 per-field 1D-CNN 대체)
LSTM_FULL59_N_SENSOR_CH = len(TEMPORAL_FEATURES)                               # 7
LSTM_FULL59_D_HIDDEN_S  = 16                                                   # sensor LSTM hidden
LSTM_FULL59_D_EMBED_S   = 1                                                    # sensor proj 출력 (★ 변경: 28 → 1)

LSTM_FULL59_N_DSCNN_CH = len(DSCNN_FEATURE_MAP)                                # 8
LSTM_FULL59_D_HIDDEN_D = 16
LSTM_FULL59_D_EMBED_D  = 8

LSTM_FULL59_N_CAD_CH      = 2
LSTM_FULL59_CAD_PATCH_H   = LSTM_CROP_H                                        # 8
LSTM_FULL59_CAD_PATCH_W   = LSTM_CROP_W
LSTM_FULL59_D_CNN_C       = 32
LSTM_FULL59_D_HIDDEN_C    = 16
LSTM_FULL59_D_EMBED_C     = 8

LSTM_FULL59_N_SCAN_CH     = 2
LSTM_FULL59_SCAN_PATCH_H  = LSTM_CROP_H
LSTM_FULL59_SCAN_PATCH_W  = LSTM_CROP_W
LSTM_FULL59_D_CNN_SC      = 32
LSTM_FULL59_D_HIDDEN_SC   = 16
LSTM_FULL59_D_EMBED_SC    = 8

LSTM_FULL59_STATIC_IDX = [2, 18]                                               # build_height, laser_module
LSTM_FULL59_MLP_HIDDEN = (256, 128, 64)                                        # 59 → 256 → 128 → 64 → 1

LSTM_FULL59_EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / "vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1"
LSTM_FULL59_CACHE_DIR      = LSTM_FULL59_EXPERIMENT_DIR / "cache"              # 본 실험은 신규 캐시 없음 (디렉터리는 생성, 비어있음)
LSTM_FULL59_MODELS_DIR     = LSTM_FULL59_EXPERIMENT_DIR / "models"
LSTM_FULL59_RESULTS_DIR    = LSTM_FULL59_EXPERIMENT_DIR / "results"
LSTM_FULL59_FEATURES_DIR   = LSTM_FULL59_EXPERIMENT_DIR / "features"

# 모든 캐시는 기존 디렉터리 재사용 — 신규 빌드 없음
LSTM_FULL59_CACHE_V0_DIR     = LSTM_CACHE_DIR
LSTM_FULL59_CACHE_V1_DIR     = LSTM_DUAL_CACHE_DIR
LSTM_FULL59_CACHE_SENSOR_DIR = LSTM_DUAL_IMG_4_SENSOR_7_CACHE_DIR
LSTM_FULL59_CACHE_DSCNN_DIR  = LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_DIR
LSTM_FULL59_CACHE_CAD_DIR    = LSTM_FULL86_CACHE_CAD_DIR                       # fullstack 의 cad_patch 캐시 재사용
LSTM_FULL59_CACHE_SCAN_DIR   = LSTM_FULL86_CACHE_SCAN_DIR                      # fullstack 의 scan_patch 캐시 재사용
```

---

## 7. 변경 사항 요약 (fullstack `_1dcnn_sensor_4` 대비)

| 항목 | fullstack `_1dcnn_sensor_4` | **본 실험 `_sensor_1`** |
|:--:|:--:|:--:|
| 카메라 v0/v1 d_embed | 16 / 16 | 동일 |
| **Sensor 분기** | **per-field 1D-CNN ×7, d_per_field=4 (28-dim)** | **단일 multi-ch LSTM, d_embed_s=1 (1-dim)** |
| Sensor 채널간 상호작용 | ❌ (필드별 독립) | ✅ (LSTM hidden 이 7-ch 동시 인코딩) |
| Sensor 패딩 처리 | AdaptiveAvgPool — 패딩 0 평균 섞임 | packed LSTM — 패딩 미참조 |
| DSCNN 분기 | 8-ch LSTM (d=8) | 동일 |
| CAD 분기 | spatial-CNN+LSTM (in=2, d=8) | 동일 |
| Scan 분기 | spatial-CNN+LSTM (in=2, d=8) | 동일 |
| Static feat | 2 (#3, #19) | 동일 |
| MLP 입력 차원 | **86** (2+16+16+28+8+8+8) | **59** (2+16+16+**1**+8+8+8) |
| MLP 구조 | 86→256→128→64→1 | **59**→256→128→64→1 |
| 신규 캐시 | cad_patch, scan_patch (2종) | **없음** (모두 재사용) |
| 학습 hp | 1e-3 / 256 / 50 | 동일 (controlled 비교) |

### 파라미터 카운트 추정 (fullstack 대비)

| 분기 | fullstack | **본 실험** | 차이 |
|:--|:--:|:--:|:--:|
| 카메라 FrameCNN ×2 + LSTM ×2 | ~26k | ~26k | — |
| Sensor 분기 | ~15k (필드별 1D-CNN ×7) | **~0.7k** (단일 LSTM in=7, hid=16, proj 16→1) | **−14.3k** |
| DSCNN LSTM | ~1.5k | ~1.5k | — |
| CAD spatial-CNN+LSTM | ~13k | ~13k | — |
| Scan spatial-CNN+LSTM | ~13k | ~13k | — |
| MLP (Nin → 256 → 128 → 64 → 1) | ~58k (Nin=86) | **~51k** (Nin=59, fc1: 86×256=22k → 59×256=15k) | **−7k** |
| **합계** | **~125-135k** | **~105-115k** | **−20k (~15% 절감)** |

> 6,373 SV / ~110k 파라미터 ≈ **~58 SV/param** (fullstack ~50). 과적합 위험 소폭 완화. 단, sensor 압축이 너무 심해 underfit 위험 있음 — 시나리오별 RMSE 비교로 판단.

---

## 8. 디렉터리 / 파일 구조

```
Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/
├── PLAN.md                     (이 파일)
├── __init__.py
├── dataset.py                  # 거의 비어있음 — fullstack dataset 재export
├── model.py                    # VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_Sensor_1 (fullstack 모듈 import + sensor branch 만 교체)
├── train.py                    # forward 시그니처는 fullstack 과 동일 — 거의 그대로 재사용 가능
├── evaluate.py                 # 동일
└── run.py                      # 진입점 (캐시 빌드 단계 없음 — train + evaluate 만)

Sources/vppm/common/config.py   # LSTM_FULL59_* 상수 추가

Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/
├── cache/                       # 비어있음 (모든 캐시 외부 재사용)
├── models/                      # 4 props × 5 folds = 20 .pt
├── results/
│   ├── metrics_raw.json
│   ├── metrics_summary.json
│   ├── predictions_{YS,UTS,UE,TE}.csv
│   ├── correlation_plots.png
│   └── scatter_plot_uts.png
├── features/normalization.json # static(2) + sensor(7) + dscnn(8) + cad_patch(2) + scan_patch(2) — fullstack 와 동일 통계
└── experiment_meta.json
```

> `dataset.py` / `train.py` / `evaluate.py` 는 fullstack 의 동일 파일을 거의 그대로 import. 코드 중복 최소화하되, run.py 의 출력 경로 / config prefix 만 `LSTM_FULL59_*` 로 변경.

---

## 9. 실행 단계

| Phase | 작업 | 산출물 | 예상 시간 |
|:--:|:--|:--|:--:|
| **S0** | `config.py` 에 `LSTM_FULL59_*` 추가, 디렉터리 생성 | (코드) | 10분 |
| **S1** | `model.py` — fullstack `_LSTMBranch` / `_GroupLSTMBranch` import + `_SensorLSTMBranch` import + 메인 모델 (sensor 분기만 교체) | model.py | 30분 |
| **S2** | `dataset.py` — fullstack dataset 재export | dataset.py | 5분 |
| **S3** | `train.py` / `evaluate.py` — fullstack 코드 거의 그대로 (config prefix 만 LSTM_FULL59_) | train.py, evaluate.py | 20분 |
| **S4** | `run.py` 진입점 + `experiment_meta.json` (캐시 빌드 단계 제거) | run.py | 15분 |
| **S5** | smoke test (`--quick` 1 fold × YS, epochs=10) — forward/backward + sensor branch d_embed=1 동작 검증 | smoke 로그 | 10분 |
| **S6** | `docker/lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/` 환경 — `docker-setup` 서브에이전트 (사전조건: fullstack cad_patch/scan_patch 캐시 존재) | Dockerfile/compose | 20분 |
| **S7** | **풀런 — 사용자 실행** (4 props × 5 folds, ~3.5-4.5h GPU 가정) | metrics_*.json, plots | ~3.5-4.5h |
| **S8** | `RESULTS.md` — sensor 1D-CNN(28) ↔ LSTM(1) 비교, 빌드별 RMSE 분해 | RESULTS.md | 1.5h |

> **fullstack 대비 신규 작업**: model.py 의 sensor 분기 교체 + config prefix 추가만. dataset/train/evaluate 는 거의 그대로 재사용. 캐시 빌드 0건. ⇒ 구현 1-2시간 + 풀런 ~4시간.

---

## 10. 가설별 기대치

| 시나리오 | 예상 결과 | 해석 |
|:--|:--|:--|
| **A. sensor 시간 정보 임계점 충족** | RMSE ≈ fullstack `_1dcnn_sensor_4` (±1%) | sensor 표현 1-dim 으로도 인장 특성 예측에 충분 (28-dim 은 과도). 채널간 상호작용이 1D-CNN 의 표현력 손실을 보상. |
| **B. sensor 표현 부족 (underfit)** | RMSE ↑ +3~8 % (특히 UE/TE) | d=1 압축이 너무 심함. sensor 의 시간 패턴 정보가 1차원으로 안 압축됨. d=4 또는 d=7 로 ablation 필요. |
| **C. LSTM 이 1D-CNN 보다 우세** | RMSE ↓ −1~3 % | packed 로 패딩 미참조 + 채널간 상호작용 살림. 압축 손실보다 두 이점이 큼. |
| **D. 과적합 완화** | val 성능 ↑, train↔val gap ↓ vs fullstack | 파라미터 ~20k 절감 + sensor 표현 강한 정규화 효과. 6,373 SV 데이터 대비 적합 capacity. |

### 빌드별 예측

| 빌드 | sensor 시간 패턴 중요도 | 예상 영향 (vs fullstack) |
|:--|:--|:--|
| **B1.1** (기준) | 낮음 | 거의 동일 |
| **B1.2** (Keyhole) | 중간 (laser power drift) | ±1 % |
| **B1.3** (오버행) | 낮음 (CAD/scan 이 주역) | 거의 동일 |
| **B1.4** (스패터/가스) | **높음** (가스 유량 dip, 산소 spike) | sensor 표현 부족하면 +3-5 % RMSE |
| **B1.5** (리코터) | 중간 (recoater state ↔ sensor 영향) | ±1-2 % |

→ **B1.4 RMSE 분해** 가 본 실험의 핵심 차별 검증 포인트 (sensor 시간 정보 압축 한계).

---

## 11. 주의 사항

1. **d_embed_s=1 의미**: sensor LSTM 의 proj 출력이 단일 스칼라. fullstack 의 28-dim 표현 대비 28× 압축. 실질적으로 sensor 분기가 "전체 시퀀스의 1-bit summary" 역할. underfit 위험 있음.

2. **명명 규약 일관성**: 다른 분기 (`dscnn_8`, `cad_8`, `scan_8`) 와 동일하게 suffix = d_embed. `_sensor_1` ⇒ d_embed_s=1. (fullstack 의 `_1dcnn_sensor_4` 는 architecture prefix `1dcnn_` + d_per_field=4 라 패턴이 약간 다름)

3. **fullstack 모듈 import 의존성**: 본 실험 `model.py` / `dataset.py` 가 fullstack 모듈을 import. fullstack 코드 변경 시 본 실험도 영향. 향후 fullstack 리팩토링 시 import 경로 검증 필요.

4. **캐시 사전조건**: 본 실험을 돌리려면 다음 캐시가 모두 존재해야 함 (compose 사전조건 체크에 명시):
   - `vppm_lstm/cache/crop_stacks_B1.{1..5}.h5` (v0)
   - `vppm_lstm_dual/cache/crop_stacks_v1_B1.{1..5}.h5` (v1)
   - `vppm_lstm_dual_img_4_sensor_7/cache/sensor_stacks_B1.{1..5}.h5`
   - `vppm_lstm_dual_img_4_sensor_7_dscnn_8/cache/dscnn_stacks_B1.{1..5}.h5`
   - `vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache/cad_patch_stacks_B1.{1..5}.h5`
   - `vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache/scan_patch_stacks_B1.{1..5}.h5`

5. **정규화 통계**: fullstack 과 동일 (sensors per-channel min-max, 패딩 0 제외). 별도 `normalization.json` 을 본 실험 디렉터리에 새로 저장하되 값은 fullstack 과 동일해야 함 (sanity check).

6. **forward 시그니처**: fullstack 과 완전히 동일 (8 인자). 단지 sensor branch 의 출력 차원만 28 → 1. → train/evaluate 코드 거의 그대로 재사용 가능.

7. **bidirectional / num_layers**: sensor LSTM 은 `LSTM_NUM_LAYERS=1` / `LSTM_BIDIRECTIONAL=False` 그대로. 양방향 켜면 d_lstm_out=32 → proj 32→1 만 변화 — 영향 미미.

---

## 12. 후속 실험 분기 (본 실험 결과 의존)

| 결과 시나리오 | 다음 실험 |
|:--|:--|
| **A. 1-dim 으로 충분** | sensor 분기 완전 제거 ablation (d=0) — sensor 시간 정보가 인장 특성 예측에 무의미한지 검증 |
| **B. underfit** | d_embed_s sweep — `_sensor_4` (per-field LSTM, d=4×7=28), `_sensor_7` (단일 LSTM d=7) 추가 ablation |
| **C. LSTM 우세** | 다른 분기도 1D-CNN ↔ LSTM swap 비교 (DSCNN 1D-CNN vs LSTM, scan-temporal 1D-CNN vs LSTM) |
| **D. 과적합 완화** | dropout 축소 (0.1 → 0.05) + MLP 폭 확장 (256 → 384) 으로 capacity 증대 재실험 |

---

## 13. 참조

- **직접 베이스 (sensor 분기 swap)**: [`Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/`](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/) (PLAN/cache_cad_patch/cache_scan_patch/dataset/model)
- **Sensor LSTM 분기 원본**: [`Sources/vppm/lstm_dual_img_4_sensor_7/model.py`](../lstm_dual_img_4_sensor_7/model.py) (`_SensorLSTMBranch` — 본 실험 재사용)
- **카메라 분기**: [`Sources/vppm/lstm_dual/model.py`](../lstm_dual/model.py) (`_LSTMBranch` — fullstack 에서 in_channels 일반화 버전 재사용)
- **캐시들**: [`crop_stacks.py`](../lstm/crop_stacks.py), [`crop_stacks_v1.py`](../lstm_dual/crop_stacks_v1.py), [`cache_sensor.py`](../lstm_dual_img_4_sensor_7/cache_sensor.py), [`cache_dscnn.py`](../lstm_dual_img_4_sensor_7_dscnn_8/cache_dscnn.py), [`cache_cad_patch.py`](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache_cad_patch.py), [`cache_scan_patch.py`](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache_scan_patch.py)
- **21-feat 패턴 분류**: [`Sources/vppm/FEATURES.md` § 평균 처리 방식별 분류](../FEATURES.md#평균-처리-방식별-분류-1d-cnn-시퀀스화-관점)
- **결과 해석 표준**: [`Sources/pipeline_outputs/experiments/vppm_lstm/LSTM_RESULTS.md`](../../pipeline_outputs/experiments/vppm_lstm/LSTM_RESULTS.md)
