# VPPM-LSTM-Dual-Img-4-Sensor-7 실험 계획

> **한 줄 요약**: `lstm_dual_4` 의 두 카메라 LSTM 분기에 **3번째 분기로 7채널 sensor LSTM** 을 추가. 기존 21 baseline 의 G2(temporal sensor) 7-feat 단순평균을 **제거** 하고, 7-channel 센서 시퀀스를 LSTM 의 last hidden(d_embed_s=7) 으로 **대체** — 시간 패턴이 살아있는 표현을 MLP 에 공급. MLP 입력 차원은 dual_4(29) 와 동일하게 14 + 4 + 4 + 7 = **29** 로 맞춰 controlled 비교.

- **실험 이름**: `lstm_dual_img_4_sensor_7` ("dual image at d_embed=4 + sensor at d_embed=7")
- **결과 위치 (예정)**: `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7/`
- **비교 대상**: `vppm_lstm_dual_4` (camera dual + 21 baseline)
- **학습 일시 (계획)**: 2026-04-30 ~

---

## 1. 동기 — 왜 sensor 도 LSTM 인가

baseline 21 피처 중 G2(센서) 7개는 [features.py:104-108](../baseline/features.py#L104-L108) 에서 z-블록 70 layer 의 **단순 산술 평균** 으로 SV 에 들어감:

```python
for ti, key in enumerate(TEMPORAL_FEATURES):
    vals = f["temporal/{key}"][l0:l1]    # (70,) 1D 시계열
    feature[v, 11+ti] = vals.mean()       # ← 단일 스칼라
```

→ **시간 패턴 손실**. 70 layer 동안의:
- 산소 농도 spike (출력 중 챔버 누설)
- 빌드 플레이트 온도 drift (열 누적)
- 가스 유량 dip (recoater 동작 영향)

같은 *동적 신호* 가 평균에 묻혀버림. 카메라에 LSTM 을 적용해 시간 패턴을 살린 것과 같은 동기 — 센서는 SV 입력의 7/21 = **33%** 를 차지하는데 그게 다 단순 평균이라 손실 폭이 더 큼.

`vppm_lstm_dual_4` 결과와 본 실험의 차이로 **"sensor 도 LSTM 으로 처리하면 추가 이득이 있는가"** 를 검증.

---

## 2. 데이터 흐름 — 새 분기 추가

### 기존 lstm_dual_4

```
stack_v0 (B, T, 8, 8) ──[CNN+LSTM+proj]──> embed_v0 (B, 4)
stack_v1 (B, T, 8, 8) ──[CNN+LSTM+proj]──> embed_v1 (B, 4)
feat21               ─────────────────────────────────────┐
                                                          ├─> concat (B, 29) ──MLP──> (B, 1)
embed_v0 ─────────────────────────────────────────────────┤
embed_v1 ─────────────────────────────────────────────────┘
```

### 신규 lstm_dual_img_4_sensor_7

```
stack_v0 (B, T, 8, 8) ──[CNN+LSTM+proj]──> embed_v0 (B, 4)
stack_v1 (B, T, 8, 8) ──[CNN+LSTM+proj]──> embed_v1 (B, 4)
sensor_seq (B, T, 7)  ──[LSTM+proj]──>     embed_s  (B, 7)   ★ 신규 분기
feat14 (G3+G1+G4)    ─────────────────────────────────────┐
                                                          ├─> concat (B, 29) ──MLP──> (B, 1)
embed_v0 ─────────────────────────────────────────────────┤
embed_v1 ─────────────────────────────────────────────────┤
embed_s  ─────────────────────────────────────────────────┘
```

| 표기 | 의미 | 값 |
|:--:|:--|:--|
| **B** | Batch — 한 번에 처리하는 SV 개수 | 256 (`config.LSTM_BATCH_SIZE`) |
| **T** | Time steps — SV 의 활성 layer 수 (가변, 최대 70) | ≤ 70 (`LSTM_T_MAX = SV_Z_LAYERS`) |
| 8×8 | 카메라 SV 크롭 픽셀 | `LSTM_CROP_{H,W}` |
| 7 | sensor 채널 수 (TEMPORAL_FEATURES) | 고정 |

- **`feat14`** = 21 baseline feat 중 G2(센서) 7개를 **빼고 남은 14개** (CAD 3 + DSCNN 8 + Scan 3).
- **`embed_s`** = 7-channel sensor LSTM 의 last hidden(d_hidden_s=16) → proj 7-dim. **차원을 7 로 맞춰** 기존 sensor 단순 평균 7개와 같은 폭 → MLP 입력 총 차원이 lstm_dual_4 와 동일(29) 한 controlled 실험.
- **시퀀스 길이**: 카메라 v0/v1 의 `lengths` 와 **동일** 사용 (이 SV 가 활성이었던 layer 만 카운트). → sensor 도 카메라와 같은 시간축에 정렬.

---

## 3. Sensor Sequence 캐시

### 입력 / 출력

- **입력**: `temporal/{key}` (k = TEMPORAL_FEATURES 7 개) → (num_layers,) 1D 빌드 전체 시계열
- **출력 캐시**: `experiments/vppm_lstm_dual_img_4_sensor_7/cache/sensor_stacks_{B1.x}.h5`
  - `sensors`: (N, T_max=70, 7) **float32** — 패딩된 시퀀스, **raw 값** (정규화 안 함)
  - `lengths`: (N,) int16 — **카메라 v0 캐시의 lengths 와 비트 단위 동일** (검증 단계에서 assert)
  - `sv_indices`: (N, 3) int32 — `(ix, iy, iz)`
  - `sample_ids`: (N,) int32

### 빌드 절차 (`cache_sensor.py`)

```python
for build_id in builds:
    grid = SuperVoxelGrid.from_hdf5(hdf5)
    valid = find_valid_supervoxels(grid, hdf5)
    
    # temporal/* 일괄 로드 (각 (num_layers,) 작음 → 메모리 무시 가능)
    temporals = stack([f[f"temporal/{k}"][:] for k in TEMPORAL_FEATURES])  # (7, num_layers)
    part_ds = f["slices/part_ids"]
    
    for sv_i, (ix, iy, iz) in enumerate(valid["voxel_indices"]):
        l0, l1 = grid.get_layer_range(iz)
        r0, r1, c0, c1 = grid.get_pixel_range(int(ix), int(iy))
        
        # 카메라 캐시와 동일한 valid_mask 재계산 (옵션 B 선택 — I/O 중복이지만 코드 변경 최소)
        block_part = part_ds[l0:l1, r0:r1, c0:c1]                              # (Tb, h, w)
        valid_mask = (block_part > 0).reshape(block_part.shape[0], -1).any(axis=1)
        if not valid_mask.any():
            valid_mask[block_part.shape[0] // 2] = True                         # 안전장치 (lstm/crop_stacks.py:94)
        
        # 해당 valid layer 에서만 sensor 값 추출
        layers_in_block = np.arange(l0, l1)[valid_mask]
        seq = temporals[:, layers_in_block].T                                   # (T_sv, 7)
        T_sv = seq.shape[0]
        
        sensors[sv_i, :T_sv] = seq
        sensors[sv_i, T_sv:] = 0                                                # zero-pad
        lengths[sv_i] = T_sv
```

> **valid_mask 일치 검증**: 카메라 v0 캐시의 `lengths[sv_i]` 와 본 캐시의 `lengths[sv_i]` 가 모든 SV 에 대해 동일해야 함. `verify_sensor_v0_consistency` 헬퍼로 빌드별 assert.

### 정규화 — runtime per-channel min-max

baseline 의 `normalize()` 와 동일한 [-1, 1] **min-max** 를 sensor 시퀀스에도 적용:

```python
# build_normalized_dataset 에서:
sensor_min = sensors.reshape(-1, 7).min(axis=0)        # (7,) — 패딩 0 도 포함되지만,
sensor_max = sensors.reshape(-1, 7).max(axis=0)        # 실 sensor 값 범위가 0 을 포함하므로 무방
sensors_norm = 2 * (sensors - sensor_min) / (sensor_max - sensor_min + 1e-8) - 1
# 패딩 영역(원래 0) 도 함께 정규화되지만, LSTM 은 lengths 로 packed 되어 패딩 영역을 절대 보지 않음.
```

- **저장**: `experiments/vppm_lstm_dual_img_4_sensor_7/features/normalization.json` 에 `sensor_min`, `sensor_max` 추가 (기존 21-feat → **14-feat** 의 `feature_min/max` 와 별도 키).
- 캐시 자체는 **raw 보존** → 정규화 스킴이 바뀌어도 캐시 재빌드 불필요.

---

## 4. 모델 변경 (`model.py`)

### 새 분기: `_SensorLSTMBranch`

```python
class _SensorLSTMBranch(nn.Module):
    """7-channel sensor 시퀀스 → LSTM → proj(d_embed_s)."""
    def __init__(self, n_channels=7, d_hidden=16, d_embed=7,
                 num_layers=1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=d_hidden,
            num_layers=num_layers, batch_first=True, bidirectional=bidirectional,
        )
        d_lstm_out = d_hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(d_lstm_out, d_embed)
    
    def forward(self, sensors, lengths):
        # sensors: (B, T, 7) float32, lengths: (B,) int64
        packed = pack_padded_sequence(sensors, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1] if not self.bidirectional else torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.proj(h_last)   # (B, d_embed)
```

### `VPPM_LSTM_Dual_Img_4_Sensor_7`

```python
class VPPM_LSTM_Dual_Img_4_Sensor_7(nn.Module):
    """카메라 dual(v0+v1) + sensor 3분기 LSTM. G2 평균 제거 → 14-feat baseline 사용."""
    def __init__(self,
                 d_cnn=config.LSTM_D_CNN,
                 d_hidden=config.LSTM_D_HIDDEN,
                 d_embed_v0=config.LSTM_DUAL_4_D_EMBED_V0,           # = 4
                 d_embed_v1=config.LSTM_DUAL_4_D_EMBED_V1,           # = 4
                 d_embed_s=config.LSTM_DUAL_IMG_4_SENSOR_7_D_EMBED_S, # = 7
                 d_hidden_s=config.LSTM_DUAL_IMG_4_SENSOR_7_D_HIDDEN_S,# = 16
                 n_sensor_channels=config.LSTM_DUAL_IMG_4_SENSOR_7_N_CHANNELS,  # = 7
                 num_layers=config.LSTM_NUM_LAYERS,
                 bidirectional=config.LSTM_BIDIRECTIONAL,
                 hidden_dim=config.HIDDEN_DIM,
                 dropout=config.DROPOUT_RATE):
        super().__init__()
        # 카메라 두 분기 — lstm_dual.model._LSTMBranch 재사용
        self.branch_v0 = _LSTMBranch(d_cnn, d_hidden, d_embed_v0, num_layers, bidirectional)
        self.branch_v1 = _LSTMBranch(d_cnn, d_hidden, d_embed_v1, num_layers, bidirectional)
        # sensor 분기 — 신규
        self.branch_sensor = _SensorLSTMBranch(
            n_channels=n_sensor_channels,
            d_hidden=d_hidden_s, d_embed=d_embed_s,
            num_layers=num_layers, bidirectional=bidirectional,
        )
        # MLP — 입력 = 14 (G3+G1+G4) + d_embed_v0 + d_embed_v1 + d_embed_s = 29
        n_baseline_kept = config.N_FEATURES - len(config.FEATURE_GROUPS["sensor"])  # 21 - 7 = 14
        n_total = n_baseline_kept + d_embed_v0 + d_embed_v1 + d_embed_s
        self.fc1 = nn.Linear(n_total, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self._init_mlp_weights()
    
    def _init_mlp_weights(self):
        for m in (self.fc1, self.fc2,
                  self.branch_v0.proj, self.branch_v1.proj, self.branch_sensor.proj):
            nn.init.normal_(m.weight, mean=0.0, std=config.WEIGHT_INIT_STD)
            nn.init.zeros_(m.bias)
    
    def forward(self, feats14, stacks_v0, stacks_v1, sensors, lengths):
        # feats14:   (B, 14)  — G2 제거된 baseline
        # stacks_v*: (B, T, 8, 8)
        # sensors:   (B, T, 7)
        # lengths:   (B,) int64 — 세 분기 모두 공통
        embed_v0 = self.branch_v0(stacks_v0, lengths)
        embed_v1 = self.branch_v1(stacks_v1, lengths)
        embed_s  = self.branch_sensor(sensors, lengths)
        x = torch.cat([feats14, embed_v0, embed_v1, embed_s], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```

### Feature 14 추출

`build_normalized_dataset` 에서 21-feat 로드 후 G2 인덱스 제거:

```python
sensor_idx = config.FEATURE_GROUPS["sensor"]                       # [11..17]
keep_idx = [i for i in range(config.N_FEATURES) if i not in sensor_idx]  # 14 indices
features_14 = features_21[:, keep_idx]
# 정규화 통계는 14-feat 기준으로 다시 계산 (G3+G1+G4 만의 min/max)
f_min = features_14.min(axis=0); f_max = features_14.max(axis=0)
features_14_norm = normalize(features_14, f_min, f_max)
```

---

## 5. 변경 사항 요약 (lstm_dual_4 대비)

| 항목 | dual_4 (기존) | **dual_img_4_sensor_7 (신규)** |
|:--:|:--:|:--:|
| 카메라 v0 LSTM 분기 | 있음 (d_embed=4) | 동일 |
| 카메라 v1 LSTM 분기 | 있음 (d_embed=4) | 동일 |
| **Sensor LSTM 분기** | **없음** (G2 7-feat 단순평균) | **있음** (d_embed_s=7) |
| Baseline feat 차원 | 21 (G3+G1+G2+G4) | **14** (G3+G1+G4 — G2 제거) |
| MLP 입력 차원 | 21 + 4 + 4 = **29** | 14 + 4 + 4 + 7 = **29** ✓ 동일 |
| Sensor 캐시 | 없음 | 신규 — `sensor_stacks_{B1.x}.h5` |
| 학습 hp (lr/batch/patience) | 1e-3 / 256 / 50 | 동일 (controlled) |

> **단일 변수 통제**: MLP 입력 차원이 같음 → 모델 capacity 변화 없이 sensor 표현 방식만 평균 ↔ LSTM 으로 교체.

---

## 6. 가설별 기대치

| 시나리오 | 예상 결과 | 해석 |
|:--|:--|:--|
| **A. 시간 패턴이 의미 있음** | RMSE -3~-10% (특히 UE/TE) | layer 간 sensor drift / spike 가 미세조직에 영향. 평균은 정보 손실원이었음. |
| **B. 평균이 충분** | RMSE ~동일 (std 범위 안) | 70-layer 짧은 윈도우에서는 평균이 손실 정보 대부분을 담음. 추가 분기는 noise. |
| **C. 과적합** | val 성능 ↓, train↔val gap ↑ | 분기 추가로 파라미터 ~수천 늘어남. 6,373 SV 로는 여유 있어 가능성 낮음. |

**B1.4(스패터/가스 변동) / B1.5(리코터 손상)** 빌드 분해 결과가 핵심:
- 가스 유량/플레이트 온도 변동이 큰 빌드에서만 개선 → 시나리오 A 확정
- 모든 빌드에서 평탄 → B (sensor 시간 패턴이 인장 특성 예측에 별로 안 중요)

---

## 7. 디렉터리 / 파일 구조

```
Sources/vppm/lstm_dual_img_4_sensor_7/
├── PLAN.md                       (이 파일)
├── __init__.py
├── cache_sensor.py               # sensor 시퀀스 캐시 빌드
├── dataset.py                    # 4개 입력 (feat14, sv0, sv1, sensors) 로드/정규화
├── model.py                      # VPPM_LSTM_Dual_Img_4_Sensor_7
├── train.py                      # forward 시그니처 변경 반영
├── evaluate.py                   # 동일
└── run.py                        # 진입점

Sources/vppm/common/config.py     # 신규 상수 추가
  + LSTM_DUAL_IMG_4_SENSOR_7_D_EMBED_S = 7
  + LSTM_DUAL_IMG_4_SENSOR_7_D_HIDDEN_S = 16
  + LSTM_DUAL_IMG_4_SENSOR_7_N_CHANNELS = 7    # = len(TEMPORAL_FEATURES)
  + LSTM_DUAL_IMG_4_SENSOR_7_EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / "vppm_lstm_dual_img_4_sensor_7"
  + LSTM_DUAL_IMG_4_SENSOR_7_CACHE_DIR     = ... / "cache"
  + LSTM_DUAL_IMG_4_SENSOR_7_MODELS_DIR    = ... / "models"
  + LSTM_DUAL_IMG_4_SENSOR_7_RESULTS_DIR   = ... / "results"
  + LSTM_DUAL_IMG_4_SENSOR_7_FEATURES_DIR  = ... / "features"

Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7/
├── cache/
│   ├── sensor_stacks_B1.1.h5       # ~ N × 70 × 7 × 4B = 1–10 MB / 빌드
│   └── ...
├── models/                          # 4 props × 5 folds = 20 .pt
├── results/
│   ├── metrics_raw.json
│   ├── metrics_summary.json
│   ├── predictions_{YS,UTS,UE,TE}.csv
│   ├── correlation_plots.png
│   └── scatter_plot_uts.png
├── features/normalization.json      # G3+G1+G4 14-feat min/max + sensor 7ch min/max
└── experiment_meta.json
```

---

## 8. 실행 단계 (Phase)

| Phase | 작업 | 산출물 | 예상 시간 |
|:--:|:--|:--|:--:|
| **S0** | `config.py` 에 신규 상수 추가, 디렉터리 생성 | (코드만) | 5분 |
| **S1** | `cache_sensor.py` 구현 + 5빌드 캐시 빌드 (v0 lengths 와 일치 검증) | `sensor_stacks_*.h5` × 5 | 빌드당 ~5분 (HDF5 I/O) |
| **S2** | `dataset.py` — load_quad_dataset / build_normalized_dataset (G2 제거 + sensor min-max) | dataset.py | 30분 |
| **S3** | `model.py` — `_SensorLSTMBranch` + `VPPM_LSTM_Dual_Img_4_Sensor_7` | model.py | 30분 |
| **S4** | `train.py` / `evaluate.py` — forward 호출 시그니처 변경 (4 → 5 입력) | train.py, evaluate.py | 30분 |
| **S5** | `run.py` 진입점 + `experiment_meta.json` 저장 | run.py | 20분 |
| **S6** | smoke test (`--quick` 1 fold × YS only, epochs=20) — forward/backward 통과 확인 | smoke 로그 | 10분 |
| **S7** | docker-compose 환경 (`docker/lstm_dual_img_4_sensor_7/`) — `docker-setup` 서브에이전트 | Dockerfile/compose | 20분 |
| **S8** | **풀런 — 사용자 실행** (4 props × 5 folds, ~3h GPU 가정) | metrics_*.json, plots | ~3h |
| **S9** | `RESULTS.md` — 가설 A/B/C 중 결론, dual_4 와 빌드별 RMSE 비교 표 | RESULTS.md | 1h |

---

## 9. 측정오차 한계와의 관계

기존 baseline 시점의 측정오차 (16.6 / 15.6 / 1.73 / 2.92):
- YS RMSE 가 가장 측정한계에 근접. 추가 개선 여지 작음.
- **UE / TE** 가 측정한계 대비 가장 여유 (각 3.76× / 2.88×) → sensor 시간 패턴이 거기서 잡히면 가장 큰 절대치 개선.

본 실험의 1차 관전 포인트는 **UE / TE**.

---

## 10. 참조

- 기존 dual_4 코드: [Sources/vppm/lstm_dual_4/](../lstm_dual_4/) (model/dataset/train/evaluate/run)
- 카메라 LSTM 캐시 빌드: [Sources/vppm/lstm/crop_stacks.py](../lstm/crop_stacks.py)
- v1 캐시 + 일치 검증: [Sources/vppm/lstm_dual/crop_stacks_v1.py](../lstm_dual/crop_stacks_v1.py)
- baseline G2 처리: [Sources/vppm/baseline/features.py:104-108](../baseline/features.py#L104-L108)
- `FEATURE_GROUPS["sensor"]` 정의: [Sources/vppm/common/config.py:126-148](../common/config.py#L126-L148)
- 결과 해석 표준: [Sources/pipeline_outputs/experiments/vppm_lstm/LSTM_RESULTS.md](../../pipeline_outputs/experiments/vppm_lstm/LSTM_RESULTS.md)
