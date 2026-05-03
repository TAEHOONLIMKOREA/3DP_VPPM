# VPPM-LSTM-Dual-Img-4-Sensor-7-DSCNN-8 실험 계획

> **한 줄 요약**: `lstm_dual_img_4_sensor_7` 의 3분기(camera v0/v1 + sensor) 구조에 **4번째 분기로 8채널 DSCNN segmentation LSTM** 을 추가. 기존 21 baseline 의 G1(DSCNN) 8-feat 가중평균 + G2(sensor) 7-feat 산술평균을 **둘 다 제거** 하고, 시간 패턴이 살아있는 LSTM 임베딩으로 **모두 대체**. MLP 입력 차원은 dual_4(29)·sensor_7(29) 와 동일하게 6 + 4 + 4 + 7 + 8 = **29** 로 맞춰 controlled 비교.

- **실험 이름**: `lstm_dual_img_4_sensor_7_dscnn_8` ("dual image at d_embed=4 + sensor at d_embed=7 + DSCNN at d_embed=8")
- **결과 위치 (예정)**: `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7_dscnn_8/`
- **1차 비교 대상**: `vppm_lstm_dual_img_4_sensor_7` — 단일 변수 (DSCNN 표현 방식만 변경)
- **2차 비교 대상**: `vppm_lstm_dual_4` — 시간성 정보원 모두 LSTM 화한 누적 효과 (sensor + DSCNN)
- **학습 일시 (계획)**: 2026-05-01 ~

---

## 1. 동기 — 시간성 정보원 전체를 LSTM 으로 통합

### 현재까지의 누적 결과

| 실험 | 시간성 정보 처리 | UTS / TE 변화 (vs dual_4) |
|:--|:--|:--:|
| `lstm_dual_4` | 카메라 v0/v1 LSTM, **sensor·DSCNN 평균** | (기준) 29.7 / 8.5 |
| `lstm_dual_img_4_sensor_7` | 카메라 LSTM + **sensor LSTM**, DSCNN 평균 | -2.1 % / **-3.5 %** |
| **`lstm_dual_img_4_sensor_7_dscnn_8`** (본 실험) | 카메라 LSTM + sensor LSTM + **DSCNN LSTM** | ? |

→ sensor_7 가 시간 패턴 LSTM 화의 가치를 입증한 시점. **남은 시간성 정보원은 DSCNN segmentation 1개**. 이번 실험으로 "시간성 정보 전체를 LSTM 화" 가 마무리됨.

### 왜 DSCNN 도 LSTM 인가

baseline 21 피처 중 G1(DSCNN segmentation) 8개는 [features.py:218-258](../baseline/features.py#L218-L258) 에서 z-블록 70 layer 의 **CAD-mask 가중평균** 으로 SV 에 들어감:

```python
for layer in range(l0, l1):                        # 70 layer
    seg = f["slices/segmentation_results/{cls_id}"][layer]
    seg_smoothed = gaussian_filter(seg, sigma=...)
    patch_cad = part_layer[r0:r1, c0:c1] > 0
    accum[vi, ci] += seg_smoothed[r0:r1, c0:c1][patch_cad].mean() * n_cad
features[v, 3:11] = accum / counts                  # ← 단일 8-stat 스칼라
```

→ **시간 패턴 손실**. 70 layer 동안의:

| DSCNN 클래스 | 손실되는 시간 패턴 |
|:--|:--|
| **Recoater Streaking** (B1.5 핵심) | 특정 layer 에서 **갑자기 발생** 후 위 layer 에 누적 영향. 평균은 "전체 약함" vs "1 layer 강함" 구분 불가 |
| **Excessive Melting** (B1.2 Keyhole) | 연속 layer 에서 **패턴화** — 시계열 길이/연속성이 결함 강도 신호 |
| **Debris** (B1.4 스패터) | **누적**되는 경향 — 후반 layer 가중치가 평균 대비 큼 |
| **Edge Swelling / Super-Elevation** | 형상 의존적이라 **z 위치별 패턴** (낮은 z 에서 발생 → 위쪽 layer 에 영향) |
| **Soot** | 가스 흐름 변동에 따른 **layer-burst** 패턴 가능 |
| Powder / Printed (정상 2 채널) | 시간 패턴 의미 적음 — 평균과 LSTM 차이 작을 것 |

DSCNN 은 이미 **결함 분류된 semantic 신호** 라 인장 특성과 더 직접적 인과관계 → 시간 패턴 가치가 sensor 보다 클 가능성. 다만 **카메라 LSTM 과 정보 중복** (DSCNN 은 visible/0/1 에서 추출된 segmentation) 변수가 있음 — `lstm_dual_4` 에서 visible/1 추가가 평탄했던 패턴이 재현될 수 있음.

`vppm_lstm_dual_img_4_sensor_7` 결과와 본 실험의 차이로 **"DSCNN segmentation 도 LSTM 으로 처리하면 sensor LSTM 위에 추가 이득이 있는가"** 를 검증.

---

## 2. 데이터 흐름 — 4 분기 통합

### 기존 sensor_7 (3 분기)

```
stack_v0  (B, T, 8, 8) ──[CNN+LSTM+proj]──> embed_v0 (B, 4)
stack_v1  (B, T, 8, 8) ──[CNN+LSTM+proj]──> embed_v1 (B, 4)
sensor_seq(B, T, 7)    ──[LSTM+proj]──>     embed_s  (B, 7)
feat14 (G3+G1+G4)     ─────────────────────────────────────┐
                                                           ├─> concat (B, 29) ──MLP──> (B, 1)
embed_v0 ──────────────────────────────────────────────────┤
embed_v1 ──────────────────────────────────────────────────┤
embed_s  ──────────────────────────────────────────────────┘
```

### 신규 sensor_7_dscnn_8 (4 분기)

```
stack_v0  (B, T, 8, 8) ──[CNN+LSTM+proj]──> embed_v0 (B, 4)
stack_v1  (B, T, 8, 8) ──[CNN+LSTM+proj]──> embed_v1 (B, 4)
sensor_seq(B, T, 7)    ──[LSTM+proj]──>     embed_s  (B, 7)
dscnn_seq (B, T, 8)    ──[LSTM+proj]──>     embed_d  (B, 8)   ★ 신규 분기
feat6 (G3+G4)         ─────────────────────────────────────┐
                                                           ├─> concat (B, 29) ──MLP──> (B, 1)
embed_v0 ──────────────────────────────────────────────────┤
embed_v1 ──────────────────────────────────────────────────┤
embed_s  ──────────────────────────────────────────────────┤
embed_d  ──────────────────────────────────────────────────┘
```

| 표기 | 의미 | 값 |
|:--:|:--|:--|
| **B** | Batch — 한 번에 처리하는 SV 개수 | 256 (`config.LSTM_BATCH_SIZE`) |
| **T** | Time steps — SV 의 활성 layer 수 (가변, 최대 70) | ≤ 70 (`LSTM_T_MAX = SV_Z_LAYERS`) |
| 8×8 | 카메라 SV 크롭 픽셀 | `LSTM_CROP_{H,W}` |
| 7 | sensor 채널 수 (TEMPORAL_FEATURES) | 고정 |
| 8 | DSCNN 클래스 수 (paper 8-class) | 고정 (`DSCNN_FEATURE_MAP`) |

- **`feat6`** = 21 baseline feat 중 G2(sensor 7) + G1(DSCNN 8) 둘 다 빼고 남은 **6개** (CAD 3 + Scan 3). 시간성 정보가 모두 LSTM 분기로 이동했으므로 baseline 에는 SV 단위 정적 신호 (CAD 형상 + 스캔 통계) 만 남음.
- **`embed_d`** = 8-channel DSCNN LSTM 의 last hidden(d_hidden_d=16) → proj 8-dim. **차원을 8 로 맞춰** 기존 DSCNN 평균 8개와 같은 폭 → MLP 입력 총 차원이 sensor_7 와 동일(29) 한 controlled 실험.
- **시퀀스 길이**: 카메라 v0/v1 / sensor / DSCNN 4 분기 모두 **공통 `lengths`** (이 SV 가 활성이었던 layer 만 카운트). 같은 시간축에 정렬되어야 함.
- **CNN 불필요 (DSCNN 분기)**: DSCNN 은 SV 영역 평균 (per-layer 8-scalar) 만 캐시하므로 CNN 통과 없이 직접 LSTM 입력. sensor 분기와 같은 단순 LSTM 구조.

---

## 3. DSCNN Sequence 캐시

> sensor cache (`sensor_stacks_*.h5`) 는 **sensor_7 실험의 cache 디렉터리를 그대로 재사용** — 별도 빌드 불필요. 본 PLAN 의 캐시 작업은 DSCNN 만.

### 입력 / 출력

- **입력**: `slices/segmentation_results/{cls_id}` (DSCNN_FEATURE_MAP 의 8개 hdf5 class id `[0, 1, 3, 5, 6, 7, 8, 10]`) → (num_layers, H, W) per-layer segmentation map
- **출력 캐시**: `experiments/vppm_lstm_dual_img_4_sensor_7_dscnn_8/cache/dscnn_stacks_{B1.x}.h5`
  - `dscnn`: (N, T_max=70, 8) **float32** — 패딩된 시퀀스, **SV crop ∩ CAD 영역 평균** (baseline `_extract_dscnn_features_block` 의 layer-별 결과를 누적하지 않고 그대로 저장)
  - `lengths`: (N,) int16 — **카메라 v0 캐시 / sensor 캐시 의 lengths 와 비트 단위 동일** (검증 단계에서 assert)
  - `sv_indices`: (N, 3) int32 — `(ix, iy, iz)`
  - `sample_ids`: (N,) int32

### 빌드 절차 (`cache_dscnn.py`)

```python
for build_id in builds:
    grid = SuperVoxelGrid.from_hdf5(hdf5)
    valid = find_valid_supervoxels(grid, hdf5)
    dscnn_class_ids = [v[0] for v in config.DSCNN_FEATURE_MAP.values()]   # 8 ids
    sigma_px = config.GAUSSIAN_STD_PIXELS
    
    part_ds = f["slices/part_ids"]
    
    for sv_i, (ix, iy, iz) in enumerate(valid["voxel_indices"]):
        l0, l1 = grid.get_layer_range(iz)
        r0, r1, c0, c1 = grid.get_pixel_range(int(ix), int(iy))
        
        # 카메라 캐시와 동일한 valid_mask 재계산
        block_part = part_ds[l0:l1, r0:r1, c0:c1]                          # (Tb, h, w)
        valid_mask = (block_part > 0).reshape(block_part.shape[0], -1).any(axis=1)
        if not valid_mask.any():
            valid_mask[block_part.shape[0] // 2] = True                     # lstm/crop_stacks.py:94 안전장치
        
        # 활성 layer 별로 8 채널 segmentation 평균 추출
        layers_in_block = np.arange(l0, l1)[valid_mask]
        seq = np.zeros((len(layers_in_block), 8), dtype=np.float32)
        
        for ti, layer in enumerate(layers_in_block):
            part_layer = part_ds[layer]
            patch_cad = part_layer[r0:r1, c0:c1] > 0
            n_cad = patch_cad.sum()
            if n_cad == 0:
                continue                                                    # 0 으로 둠 (드물게 layer 내 CAD 부재)
            for ci, cls_id in enumerate(dscnn_class_ids):
                seg_key = f"slices/segmentation_results/{cls_id}"
                if seg_key not in f:
                    continue
                seg = f[seg_key][layer].astype(np.float32)
                seg = gaussian_filter(seg, sigma=sigma_px)
                seq[ti, ci] = seg[r0:r1, c0:c1][patch_cad].mean()
        
        T_sv = seq.shape[0]
        dscnn[sv_i, :T_sv] = seq
        dscnn[sv_i, T_sv:] = 0                                              # zero-pad
        lengths[sv_i] = T_sv
```

> **valid_mask 일치 검증**: 카메라 v0 캐시의 `lengths[sv_i]`, sensor 캐시의 `lengths[sv_i]`, 본 캐시의 `lengths[sv_i]` 가 모든 SV 에 대해 동일해야 함. `verify_dscnn_v0_consistency` 헬퍼로 빌드별 assert.

> **I/O 비용**: layer 당 8 클래스 × (H × W) float32 segmentation map 로딩 + 가우시안 블러. 빌드당 ~10–20 분 추정 (sensor 캐시 ~5분 대비 무거움). 캐시 1회 생성 후 재사용.

### Layer-별 Gaussian 블러 캐시 vs 즉석 계산

baseline 은 **block 단위로 한 번에** 8 채널 × 70 layer 가우시안 블러를 계산. 본 캐시도 동일하게 SV 별 layer-loop 안에서 layer 단위 블러 → 빌드당 한 번만 수행 (sigma_px 고정).

> 메모리 절약 위해 layer-단위 블러 결과는 캐시하지 않고 즉석 계산. 캐시 파일은 SV-단위 (T_sv, 8) 시퀀스만 저장.

### 정규화 — runtime per-channel min-max

baseline 의 `normalize()` 와 동일한 [-1, 1] **min-max** 를 DSCNN 시퀀스에도 적용:

```python
# build_normalized_dataset 에서:
dscnn_min = dscnn.reshape(-1, 8).min(axis=0)        # (8,)
dscnn_max = dscnn.reshape(-1, 8).max(axis=0)        # 패딩 0 포함
dscnn_norm = 2 * (dscnn - dscnn_min) / (dscnn_max - dscnn_min + 1e-8) - 1
# 패딩 영역(원래 0) 도 함께 정규화되지만, LSTM 은 lengths 로 packed 되어 패딩 영역을 절대 보지 않음.
```

- **저장**: `experiments/vppm_lstm_dual_img_4_sensor_7_dscnn_8/features/normalization.json` 에 `dscnn_min`, `dscnn_max` 추가 (기존 21-feat → **6-feat** 의 `feature_min/max` 와 별도 키, sensor 7ch min/max 도 함께).
- 캐시 자체는 **raw 보존** → 정규화 스킴이 바뀌어도 캐시 재빌드 불필요.

> DSCNN segmentation 값은 [0, 1] 범위 (확률) 라 min-max 가 채널별로 차이가 작을 수 있음. 그래도 sensor 와 일관성 유지 위해 동일 스킴 적용.

---

## 4. 모델 변경 (`model.py`)

### 새 분기: `_DSCNNLSTMBranch`

```python
class _DSCNNLSTMBranch(nn.Module):
    """8-channel DSCNN segmentation 시퀀스 → LSTM → proj(d_embed_d).
    sensor_7 의 _SensorLSTMBranch 와 동일 구조 — 채널 수와 d_embed 만 다름.
    """
    def __init__(self, n_channels=8, d_hidden=16, d_embed=8,
                 num_layers=1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=d_hidden,
            num_layers=num_layers, batch_first=True, bidirectional=bidirectional,
        )
        d_lstm_out = d_hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(d_lstm_out, d_embed)
    
    def forward(self, dscnn, lengths):
        # dscnn: (B, T, 8) float32, lengths: (B,) int64
        packed = pack_padded_sequence(dscnn, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1] if not self.bidirectional else torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.proj(h_last)   # (B, d_embed)
```

> sensor_7 의 `_SensorLSTMBranch` 와 코드 본질이 동일 → 향후 공통 모듈로 묶을 여지 있음 (이번 실험에서는 별도 클래스로 두고 통합은 미루기).

### `VPPM_LSTM_Dual_Img_4_Sensor_7_DSCNN_8`

```python
class VPPM_LSTM_Dual_Img_4_Sensor_7_DSCNN_8(nn.Module):
    """카메라 dual(v0+v1) + sensor + DSCNN 4분기 LSTM. G1·G2 평균 제거 → 6-feat baseline 사용."""
    def __init__(self,
                 d_cnn=config.LSTM_D_CNN,
                 d_hidden=config.LSTM_D_HIDDEN,
                 d_embed_v0=config.LSTM_DUAL_4_D_EMBED_V0,                  # = 4
                 d_embed_v1=config.LSTM_DUAL_4_D_EMBED_V1,                  # = 4
                 d_embed_s=config.LSTM_DUAL_IMG_4_SENSOR_7_D_EMBED_S,       # = 7
                 d_hidden_s=config.LSTM_DUAL_IMG_4_SENSOR_7_D_HIDDEN_S,     # = 16
                 n_sensor_channels=config.LSTM_DUAL_IMG_4_SENSOR_7_N_CHANNELS,            # = 7
                 d_embed_d=config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_D_EMBED_D,            # = 8
                 d_hidden_d=config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_D_HIDDEN_D,          # = 16
                 n_dscnn_channels=config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_N_CHANNELS,    # = 8
                 num_layers=config.LSTM_NUM_LAYERS,
                 bidirectional=config.LSTM_BIDIRECTIONAL,
                 hidden_dim=config.HIDDEN_DIM,
                 dropout=config.DROPOUT_RATE):
        super().__init__()
        # 카메라 두 분기 — lstm_dual.model._LSTMBranch 재사용
        self.branch_v0 = _LSTMBranch(d_cnn, d_hidden, d_embed_v0, num_layers, bidirectional)
        self.branch_v1 = _LSTMBranch(d_cnn, d_hidden, d_embed_v1, num_layers, bidirectional)
        # sensor 분기 — lstm_dual_img_4_sensor_7.model._SensorLSTMBranch 재사용
        self.branch_sensor = _SensorLSTMBranch(
            n_channels=n_sensor_channels,
            d_hidden=d_hidden_s, d_embed=d_embed_s,
            num_layers=num_layers, bidirectional=bidirectional,
        )
        # DSCNN 분기 — 신규
        self.branch_dscnn = _DSCNNLSTMBranch(
            n_channels=n_dscnn_channels,
            d_hidden=d_hidden_d, d_embed=d_embed_d,
            num_layers=num_layers, bidirectional=bidirectional,
        )
        # MLP — 입력 = 6 (G3+G4) + d_embed_v0 + d_embed_v1 + d_embed_s + d_embed_d = 29
        n_baseline_kept = (config.N_FEATURES
                          - len(config.FEATURE_GROUPS["sensor"])
                          - len(config.FEATURE_GROUPS["dscnn"]))               # 21 - 7 - 8 = 6
        n_total = n_baseline_kept + d_embed_v0 + d_embed_v1 + d_embed_s + d_embed_d
        self.fc1 = nn.Linear(n_total, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self._init_mlp_weights()
    
    def _init_mlp_weights(self):
        for m in (self.fc1, self.fc2,
                  self.branch_v0.proj, self.branch_v1.proj,
                  self.branch_sensor.proj, self.branch_dscnn.proj):
            nn.init.normal_(m.weight, mean=0.0, std=config.WEIGHT_INIT_STD)
            nn.init.zeros_(m.bias)
    
    def forward(self, feats6, stacks_v0, stacks_v1, sensors, dscnn, lengths):
        # feats6:    (B, 6)   — G1·G2 제거된 baseline (CAD 3 + Scan 3)
        # stacks_v*: (B, T, 8, 8)
        # sensors:   (B, T, 7)
        # dscnn:     (B, T, 8)
        # lengths:   (B,) int64 — 네 분기 모두 공통
        embed_v0 = self.branch_v0(stacks_v0, lengths)
        embed_v1 = self.branch_v1(stacks_v1, lengths)
        embed_s  = self.branch_sensor(sensors, lengths)
        embed_d  = self.branch_dscnn(dscnn, lengths)
        x = torch.cat([feats6, embed_v0, embed_v1, embed_s, embed_d], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```

### Feature 6 추출

`build_normalized_dataset` 에서 21-feat 로드 후 G1 + G2 인덱스 제거:

```python
sensor_idx = config.FEATURE_GROUPS["sensor"]                       # [11..17]
dscnn_idx  = config.FEATURE_GROUPS["dscnn"]                        # [3..10]
remove_idx = set(sensor_idx) | set(dscnn_idx)
keep_idx = [i for i in range(config.N_FEATURES) if i not in remove_idx]  # 6 indices: [0,1,2,18,19,20]
features_6 = features_21[:, keep_idx]
# 정규화 통계는 6-feat 기준으로 다시 계산 (G3+G4 만의 min/max)
f_min = features_6.min(axis=0); f_max = features_6.max(axis=0)
features_6_norm = normalize(features_6, f_min, f_max)
```

---

## 5. 변경 사항 요약 (sensor_7 대비)

| 항목 | sensor_7 (기존) | **sensor_7_dscnn_8 (신규)** |
|:--:|:--:|:--:|
| 카메라 v0 LSTM 분기 | 있음 (d_embed=4) | 동일 |
| 카메라 v1 LSTM 분기 | 있음 (d_embed=4) | 동일 |
| Sensor LSTM 분기 | 있음 (d_embed=7) | 동일 |
| **DSCNN LSTM 분기** | **없음** (G1 8-feat 가중평균) | **있음** (d_embed_d=8) |
| Baseline feat 차원 | 14 (G3+G1+G4) | **6** (G3+G4 — G1 제거) |
| MLP 입력 차원 | 14 + 4 + 4 + 7 = **29** | 6 + 4 + 4 + 7 + 8 = **29** ✓ 동일 |
| Sensor 캐시 | 신규 (`sensor_stacks_*.h5`) | **재사용** (sensor_7 디렉터리에서 import) |
| DSCNN 캐시 | 없음 | 신규 — `dscnn_stacks_{B1.x}.h5` |
| 학습 hp (lr/batch/patience) | 1e-3 / 256 / 50 | 동일 (controlled) |

> **단일 변수 통제**: MLP 입력 차원이 같음 → 모델 capacity 변화 없이 DSCNN 표현 방식만 가중평균 ↔ LSTM 으로 교체. **sensor_7 → 본 실험 RMSE 차이가 정확히 "DSCNN LSTM 효과"** 가 됨.

### dual_4 대비 누적 변화

| 항목 | dual_4 | sensor_7 | **sensor_7_dscnn_8** |
|:--:|:--:|:--:|:--:|
| 시간성 정보원 LSTM 화 | 카메라만 | 카메라 + 센서 | **카메라 + 센서 + DSCNN (전부)** |
| Baseline feat 차원 | 21 | 14 | **6** |
| MLP 입력 차원 | 29 | 29 | 29 ✓ |
| 분기 수 | 2 | 3 | **4** |

---

## 6. 가설별 기대치

| 시나리오 | 예상 결과 | 해석 |
|:--|:--|:--|
| **A. DSCNN 시간 패턴이 의미 있음** | sensor_7 대비 RMSE −2~−5 % (UTS / TE 중심) | 결함 발생 layer / 누적 패턴이 미세조직에 영향. sensor LSTM 위에 추가 신호 제공. |
| **B. 카메라 LSTM 과 정보 중복** | sensor_7 대비 ~동일 (std 범위 안) | DSCNN 은 visible/0/1 에서 추출되므로 카메라 LSTM 이 이미 같은 시간 패턴을 봄. 추가 분기는 redundant. `lstm_dual_4` 의 visible/1 평탄 패턴 재현. |
| **C. 과적합 시작** | val 성능 ↓, train↔val gap ↑ | 분기 추가로 파라미터 ~수천 늘어남. 6,373 SV 로는 여유 있어 가능성 낮음. |
| **D. Sensor + DSCNN 시너지** | sensor_7 의 UE 후퇴(+1.7 %) 가 사라지고 정상화 | 결함 신호와 sensor 신호가 상호 보완적이면 fold 0 outlier 해소 가능 |

### 빌드별 예측

| 빌드 | 핵심 결함 채널 | DSCNN LSTM 효과 예상 |
|:--|:--|:--|
| **B1.5** (리코터 손상) | Recoater Streaking spike | 🔥 **가장 큰 개선 후보** — 평균이 spike 를 가장 못 잡음. sensor LSTM 으로도 부분만 잡힐 가능성 |
| **B1.4** (스패터/가스) | Debris 누적 | 🔥 후반 layer 가중치 차이 → LSTM 우위. sensor (가스 유량) + DSCNN (debris 분류) 시너지 후보 |
| **B1.2** (Keyhole) | Excessive Melting 연속 패턴 | ⚪ 평균도 어느 정도 잡음. 차이 적을 수 있음 |
| **B1.3** (오버행) | Edge Swelling z-위치 의존 | ⚪ 형상 의존이라 평균 vs LSTM 차이 작음 |
| **B1.1** (기준) | 결함 적음 | ❌ 차이 거의 없을 것 |

→ **B1.5 / B1.4 RMSE 분해** 가 핵심 검증. 종합 RMSE 가 평탄해도 두 빌드에서 큰 개선이 잡히면 시나리오 A 부분 확정.

### sensor_7 결과로부터의 정량 추론

| 속성 | dual_4 (기준) | sensor_7 (실측) | **sensor_7_dscnn_8 예상** | 근거 |
|:--:|:--:|:--:|:--:|:--|
| YS  | 20.7 | 20.6 (-0.4 %) | 20.5 ± 0.9 (-0~1 %) | 측정한계 1.24× — 여유 적음 |
| UTS | 29.7 | 29.1 (-2.1 %) | **28.0–28.8 (-1~3 %)** | 결함 직접 신호. 카메라 중복 변수 존재 |
| UE  | 6.6  | 6.7 (+1.7 %)  | **6.4–6.6 (-1~3 %)** | 결함 직접 신호로 fold 0 outlier 해소 가능성 |
| TE  | 8.5  | 8.2 (-3.5 %)  | **7.9–8.1 (-1~4 %)** | sensor 와 비슷한 메커니즘. DSCNN 가산 여부가 관건 |

> **명확한 검증 포인트**: sensor_7 대비 차이가 std (≈ 0.2~0.3) 보다 크면 시나리오 A, 작으면 B. UE 의 fold 0 outlier 가 사라지면 시나리오 D 추가 지지.

---

## 7. 디렉터리 / 파일 구조

```
Sources/vppm/lstm_dual_img_4_sensor_7_dscnn_8/
├── PLAN.md                       (이 파일)
├── __init__.py
├── cache_dscnn.py                # DSCNN 시퀀스 캐시 빌드 (sensor 캐시는 재사용)
├── dataset.py                    # 5개 입력 (feat6, sv0, sv1, sensors, dscnn) 로드/정규화
├── model.py                      # VPPM_LSTM_Dual_Img_4_Sensor_7_DSCNN_8
├── train.py                      # forward 시그니처 변경 반영 (5 → 6 입력)
├── evaluate.py                   # 동일
└── run.py                        # 진입점

Sources/vppm/common/config.py     # 신규 상수 추가 (sensor_7 의 상수는 그대로 재사용)
  + LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_D_EMBED_D = 8
  + LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_D_HIDDEN_D = 16
  + LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_N_CHANNELS = 8    # = len(DSCNN_FEATURE_MAP)
  + LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / "vppm_lstm_dual_img_4_sensor_7_dscnn_8"
  + LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_DIR     = ... / "cache"
  + LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_MODELS_DIR    = ... / "models"
  + LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_RESULTS_DIR   = ... / "results"
  + LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_FEATURES_DIR  = ... / "features"

Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7_dscnn_8/
├── cache/
│   ├── dscnn_stacks_B1.1.h5       # ~ N × 70 × 8 × 4B = 1–10 MB / 빌드 (DSCNN 만)
│   └── ...                        # sensor_stacks_*.h5 는 sensor_7 디렉터리에서 import (symlink 또는 path 참조)
├── models/                          # 4 props × 5 folds = 20 .pt
├── results/
│   ├── metrics_raw.json
│   ├── metrics_summary.json
│   ├── predictions_{YS,UTS,UE,TE}.csv
│   ├── correlation_plots.png
│   └── scatter_plot_uts.png
├── features/normalization.json      # G3+G4 6-feat min/max + sensor 7ch min/max + dscnn 8ch min/max
└── experiment_meta.json
```

---

## 8. 실행 단계 (Phase)

| Phase | 작업 | 산출물 | 예상 시간 |
|:--:|:--|:--|:--:|
| **S0** | `config.py` 에 신규 상수 추가, 디렉터리 생성 | (코드만) | 5분 |
| **S1** | `cache_dscnn.py` 구현 + 5빌드 캐시 빌드 (v0 / sensor lengths 와 일치 검증) | `dscnn_stacks_*.h5` × 5 | 빌드당 ~10–20분 |
| **S2** | `dataset.py` — load_pent_dataset (5 입력) / build_normalized_dataset (G1+G2 제거 + dscnn min-max) | dataset.py | 30분 |
| **S3** | `model.py` — `_DSCNNLSTMBranch` + `VPPM_LSTM_Dual_Img_4_Sensor_7_DSCNN_8` | model.py | 30분 |
| **S4** | `train.py` / `evaluate.py` — forward 호출 시그니처 변경 (5 → 6 입력) | train.py, evaluate.py | 30분 |
| **S5** | `run.py` 진입점 + `experiment_meta.json` 저장 | run.py | 20분 |
| **S6** | smoke test (`--quick` 1 fold × YS only, epochs=20) — forward/backward 통과 확인 | smoke 로그 | 10분 |
| **S7** | docker-compose 환경 (`docker/lstm_dual_img_4_sensor_7_dscnn_8/`) — `docker-setup` 서브에이전트 | Dockerfile/compose | 20분 |
| **S8** | **풀런 — 사용자 실행** (4 props × 5 folds, ~3.5h GPU 가정) | metrics_*.json, plots | ~3.5h |
| **S9** | `RESULTS.md` — 가설 A/B/C/D 중 결론, dual_4 / sensor_7 / 본 실험의 누적 비교 + 빌드별 RMSE 분해 표 | RESULTS.md | 1h |

> sensor_7 대비 추가 작업은 거의 없음 — DSCNN 캐시 빌드 + 모델 분기 1개 추가. 모델/학습 코드는 sensor_7 에서 거의 그대로 이식 가능.

---

## 9. 측정오차 한계와의 관계

기존 baseline 시점의 측정오차 (16.6 / 15.6 / 1.73 / 2.92):
- YS RMSE 가 가장 측정한계에 근접 (sensor_7 기준 1.24×). 추가 개선 여지 작음.
- **UE / TE** 가 측정한계 대비 가장 여유 (sensor_7 기준 각 3.88× / 2.79×). DSCNN 결함 시간 패턴이 거기서 잡히면 가장 큰 절대치 개선.
- **UTS** 도 sensor_7 에서 −2.1 % 잡힌 것을 감안하면 추가 개선 여지 있음.

본 실험의 1차 관전 포인트는 **UTS / TE / UE** (특히 sensor_7 의 UE fold 0 outlier 가 DSCNN 분기 추가로 해소되는지가 별도 관심사).

---

## 10. 후속 실험 분기 (본 실험 결과 의존)

| 결과 시나리오 | 다음 실험 |
|:--|:--|
| **A. UTS / TE 개선 (≥ 2 %)** | DSCNN 채널별 ablation — 8 채널 중 어느 것이 핵심 신호인지 분리. `dscnn_subablation` 기존 그룹 활용 가능 |
| **B. 평탄 (모든 속성 std 안)** | DSCNN 시간 패턴이 카메라 LSTM 과 redundant 확정. visible-only 정보원의 한계 재확인. **scan path geometry / multi-target / per-build fine-tuning** 등 다른 정보원으로 이동 |
| **C. 빌드별 분해에서 B1.4/B1.5 만 큰 개선** | per-build fine-tuning 또는 빌드별 weighting 도입. 결함 빌드용 sub-model 검토 |
| **D. UE outlier 해소 (시너지)** | Multi-target joint training 검토 — 4 속성 cross-regularization 으로 추가 안정성 확보 가능성 |

---

## 11. 참조

- 기존 dual_4 코드: [Sources/vppm/lstm_dual_4/](../lstm_dual_4/) (model/dataset/train/evaluate/run)
- **sensor_7 (직접 베이스)**: [Sources/vppm/lstm_dual_img_4_sensor_7/](../lstm_dual_img_4_sensor_7/) (PLAN/cache_sensor/dataset/model)
- sensor_7 결과 README: [vppm_lstm_dual_img_4_sensor_7/results/README.md](../../pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7/results/README.md)
- 카메라 LSTM 캐시 빌드: [Sources/vppm/lstm/crop_stacks.py](../lstm/crop_stacks.py)
- v1 캐시 + 일치 검증: [Sources/vppm/lstm_dual/crop_stacks_v1.py](../lstm_dual/crop_stacks_v1.py)
- baseline G1(DSCNN) 처리: [Sources/vppm/baseline/features.py:218-258](../baseline/features.py#L218-L258)
- baseline G2(sensor) 처리: [Sources/vppm/baseline/features.py:104-108](../baseline/features.py#L104-L108)
- `DSCNN_FEATURE_MAP` 정의: [Sources/vppm/common/config.py:53-63](../common/config.py#L53-L63)
- `FEATURE_GROUPS["dscnn"]` / `["sensor"]` 정의: [Sources/vppm/common/config.py:128-129](../common/config.py#L128-L129)
- DSCNN 채널별 ablation: `vppm/common/config.py:151-165` (dscnn_subablation 그룹 — 후속 실험 시 활용)
- 결과 해석 표준: [Sources/pipeline_outputs/experiments/vppm_lstm/LSTM_RESULTS.md](../../pipeline_outputs/experiments/vppm_lstm/LSTM_RESULTS.md)
