# VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-1DCNN-Sensor-4 실험 계획

> **한 줄 요약**: baseline 21-feat 중 **시간성 정보를 가진 모든 피처(P1+P2+P3 = 19개)** 를 layer-시퀀스 인코더(LSTM 또는 1D-CNN)로 교체. 카메라 v0/v1 임베딩은 d_embed=**16** 으로 확장하고, sensor 는 필드별 1D-CNN(필드당 4-dim, 총 28) 으로, DSCNN 은 8-ch LSTM(8-dim) 으로, P1(`distance_from_edge`/`distance_from_overhang`) · P2(`laser_return_delay`/`laser_stripe_boundaries`) 는 그룹별 LSTM(d=8) 으로 처리. 정적 스칼라(P4: `build_height`, `laser_module`) 만 baseline 에 남김. 최종 MLP 입력 차원 = **86**.

- **실험 이름**: `lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4`
- **결과 위치 (예정)**: `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/`
- **1차 비교 대상**: `vppm_lstm_dual_img_4_sensor_7_dscnn_8` — 시간성 처리 깊이 변화 (ablation: 카메라 4→16 / sensor LSTM→1D-CNN per-field / scan·CAD 평균→LSTM)
- **2차 비교 대상**: `vppm_lstm_dual_4` (image-only LSTM), baseline 21-feat MLP
- **학습 일시 (계획)**: 2026-05-01 ~

---

## 1. 동기 — 모든 시간성 정보원의 LSTM/1D-CNN 화

baseline 21-feat 명세([`Sources/vppm/FEATURES.md`](../FEATURES.md#평균-처리-방식별-분류-1d-cnn-시퀀스화-관점)) 에서 정의한 **평균 처리 패턴(P1-P4)** 기준 분류:

| 패턴 | 피처 (#) | baseline 처리 | 본 실험 처리 | 순서정보 |
|:--:|:--|:--|:--|:--:|
| **P1** | #1 distance_edge, #2 distance_overhang, #4-11 DSCNN 8ch | layer 단위 픽셀 평균 → CAD-가중 z-평균 (스칼라 1) | (#1,#2) **CAD spatial-CNN+LSTM(d=8)** — 8×8 패치 보존 + inversion + cad_mask 픽셀곱 + (#4-11) **DSCNN LSTM(d=8)** (스칼라 시퀀스 그대로) | ✅ + spatial 보존 |
| **P2** | #20 return_delay, #21 stripe_boundaries | melt 픽셀 평균 → 유효 layer 단순평균 (스칼라 1) | **Scan spatial-CNN+LSTM(d=8)** — 8×8 패치 보존 (raw + 미용융 픽셀=0) | ✅ + spatial 보존 |
| **P3** | #12-18 sensor 7ch | 70 layer 단순 평균 (스칼라 1) | **per-field 1D-CNN** (필드당 4-dim, 총 28) | ✅ 보존 |
| **P4** | #3 build_height, #19 laser_module | 픽셀/layer 미참조 (직접 할당) | **그대로 유지** (정적 스칼라 2개) | — |

→ 시간성 정보가 있는 19개(P1-P3) 를 모두 시퀀스 인코더로 교체. baseline 에는 P4 의 2 스칼라만 남음.

### 누적 실험 변천

| 실험 | 시간성 처리 깊이 |
|:--|:--|
| `vppm` (21-feat MLP) | 모두 평균 (스칼라) |
| `vppm_lstm_dual_4` | 카메라 v0/v1 LSTM(d=4), 그 외 평균 |
| `vppm_lstm_dual_img_4_sensor_7` | + sensor LSTM(d=7) |
| `vppm_lstm_dual_img_4_sensor_7_dscnn_8` | + DSCNN LSTM(d=8) |
| **본 실험** | **+ CAD/scan spatial-CNN+LSTM(d=8, 8×8 패치 보존), 카메라 d=16, sensor 필드별 1D-CNN(4-dim)** |

### 왜 이 변경들인가

| 변경 | 근거 |
|:--|:--|
| **카메라 d_embed 1 → 4 → 16** | 8×8 SV 크롭은 픽셀 64개의 spatial 정보 + T 의 temporal 정보. d=16 까지는 정보 압축으로 충분. d_hidden=16 과 일치시켜 proj 가 identity 에 가깝게 출발 가능 (d_hidden 도 16 유지). |
| **sensor 필드별 1D-CNN(4-dim)** | sensor 7채널은 공정 의미가 서로 다름 (유량 vs 온도 vs 산소 vs 시간). 채널을 섞는 multi-channel LSTM 보다 채널별 독립 처리가 해석성·표현력 면에서 유리. 1D-CNN 은 LSTM 보다 짧은 시퀀스(≤70)에서 패턴 추출에 빠름. 필드당 4-dim = level + slope + curvature + spike 카운터 등 2차 통계까지 표현 가능. d=3 (early/mid/late 만) 대비 1차 미분 신호를 더 잘 살림. 다른 분기(cam=16, dscnn=8, cad=8, scan=8)와 4 의 배수로 align. |
| **CAD spatial-CNN+LSTM (#1, #2) → d=8** | `distance_from_edge`/`distance_from_overhang` 은 (a) z-축 변화 (오버행 위로 누적되는 영향) (b) **layer 안 spatial 분포** (SV 의 한쪽은 edge, 반대쪽은 interior) 둘 다 담음. baseline 의 "layer-당 8×8 → mean → 스칼라" 는 (a) 와 (b) 둘 다 잃음. 카메라 분기와 동일한 spatial-CNN+LSTM 으로 8×8 패치를 layer 마다 그대로 보존. **inversion** (`saturation - raw`) 으로 컨벤션 통일 + **cad_mask 픽셀곱** 으로 분말 영역 = 0 (nominal). |
| **Scan spatial-CNN+LSTM (#20, #21) → d=8** | `return_delay`/`stripe_boundaries` 도 같은 이유로 spatial 정보 가치. baseline 이 이미 미용융 픽셀에 0 (의도적) / NaN→0 처리 → 곱셈 마스킹 불필요. raw map 의 8×8 패치 그대로. |

---

## 2. 데이터 흐름 — 7 분기 통합

```
stack_v0    (B, T, 1, 8, 8)     ──[CNN+LSTM+proj(d=16)]────────────> embed_v0   (B, 16)
stack_v1    (B, T, 1, 8, 8)     ──[CNN+LSTM+proj(d=16)]────────────> embed_v1   (B, 16)
sensors     (B, T, 7)           ──[per-field 1D-CNN, 7×4]──────────> embed_s    (B, 28)
dscnn       (B, T, 8)           ──[LSTM+proj(d=8)]─────────────────> embed_d    (B, 8)
cad_patch   (B, T, 2, 8, 8)     ──[CNN(in=2)+LSTM+proj(d=8)]───────> embed_c    (B, 8)
scan_patch  (B, T, 2, 8, 8)     ──[CNN(in=2)+LSTM+proj(d=8)]───────> embed_sc   (B, 8)
feat_static (B, 2)              ── (build_height, laser_module) ──> feat2      (B, 2)

   feat2 ⊕ embed_v0 ⊕ embed_v1 ⊕ embed_s ⊕ embed_d ⊕ embed_c ⊕ embed_sc
   = 2 + 16 + 16 + 28 + 8 + 8 + 8 = 86
                     │
                     ▼
          MLP(86 → 256 → 128 → 64 → 1)
```

### 표기

| 표기 | 의미 | 값 |
|:--:|:--|:--|
| **B** | Batch — 한 번에 처리하는 SV 개수 | 256 (`config.LSTM_BATCH_SIZE`) |
| **T** | Time steps — SV 활성 layer 수 (가변, 최대 70) | ≤ 70 (`LSTM_T_MAX = SV_Z_LAYERS`) |
| 8×8 | 카메라 / CAD / Scan SV 크롭 픽셀 | `LSTM_CROP_{H,W}` |
| 7 | sensor 채널 수 | `len(TEMPORAL_FEATURES)` |
| 8 | DSCNN 클래스 수 | `len(DSCNN_FEATURE_MAP)` |
| 2 (cad_patch) | ch0=`edge_proximity` (inverted+masked), ch1=`overhang_proximity` (inverted+masked) — layer 별 8×8 패치 | 신규 캐시 |
| 2 (scan_patch) | ch0=`return_delay` (raw, NaN→0), ch1=`stripe_boundaries` (raw) — layer 별 8×8 패치 | 신규 캐시 |
| 2 (feat_static) | `build_height` (idx 2), `laser_module` (idx 18) | baseline 21-feat 에서 추출 |

### 시퀀스 길이 공통

7 분기 중 시퀀스 입력 6종 (v0/v1/sensor/dscnn/cad_patch/scan_patch) 의 **`lengths` 는 모두 동일**. 같은 `valid_mask = part_ids > 0 in SV xy` 규칙으로 계산 — 캐시 빌드 시 v0 캐시의 `lengths` 와 비트 단위 일치 검증 (`verify_*_v0_consistency`).

> scan_patch 의 raw 출처(melt 픽셀)는 v0 의 valid_mask(CAD) 와 다름. 그러나 본 실험에서는 **카메라 v0 의 lengths 기준으로 시퀀스를 정렬**하고, scan_patch 는 해당 layer 의 melt 픽셀 맵 (없으면 0 패치) 을 그대로 채워 lengths 를 일치시킨다. 이렇게 하면 packed LSTM 이 모두 같은 lengths 를 공유 — 정렬·패딩 코드 단순화.

---

## 3. 캐시 전략

| 캐시 | 출처 | 재사용 여부 | 신규 빌드 |
|:--|:--|:--:|:--:|
| `crop_stacks_{B}.h5` (v0) | `Sources/vppm/lstm/crop_stacks.py` | ✅ | — |
| `crop_stacks_v1_{B}.h5` (v1) | `Sources/vppm/lstm_dual/crop_stacks_v1.py` | ✅ | — |
| `sensor_stacks_{B}.h5` | `Sources/vppm/lstm_dual_img_4_sensor_7/cache_sensor.py` | ✅ | — |
| `dscnn_stacks_{B}.h5` | `Sources/vppm/lstm_dual_img_4_sensor_7_dscnn_8/cache_dscnn.py` | ✅ | — |
| **`cad_patch_stacks_{B}.h5`** | 신규 — `cache_cad_patch.py` | — | ✅ |
| **`scan_patch_stacks_{B}.h5`** | 신규 — `cache_scan_patch.py` | — | ✅ |

### 3.1 신규 캐시 — `cad_patch_stacks_{B}.h5`

baseline `_extract_cad_features_block`([`features.py:156-216`](../baseline/features.py#L156-L216)) 의 layer 단위 픽셀 맵을 **평균하지 않고** SV 8×8 패치 그대로 저장. **inversion** + **cad_mask 픽셀곱** 적용.

**입력**: `slices/part_ids` (per-layer CAD 마스크).

**출력 dataset**:
- `cad_patch` : `(N, T_max=70, 2, h, w) float16`, zero-padded — `h=w=8` (대부분), 가장자리 SV 는 ≤8
  - 채널 0: `edge_proximity` = (3.0 − distance_from_edge) × cad_mask (mm)
  - 채널 1: `overhang_proximity` = (71 − distance_from_overhang) × cad_mask (layers)
  - **inversion 후**: 0 = nominal (interior / no recent overhang), saturation = signal max
  - **cad_mask 픽셀곱 후**: 분말 영역 픽셀값 = 0 (= nominal, 다른 시퀀스 입력과 컨벤션 일관)
- `lengths` : `(N,) int16` — v0 캐시와 동일
- `sv_indices` : `(N, 3) int32`
- `sample_ids` : `(N,) int32`
- attrs:
  - `inversion_applied = True`
  - `mask_applied = True`
  - `edge_saturation_mm = 3.0`
  - `overhang_saturation_layers = 71`
  - `channel_names = ["edge_proximity", "overhang_proximity"]`
  - `convention = "0 = nominal, large = signal (matches DSCNN/scan/sensor)"`

**빌드 절차**:

```python
# extract_features 의 P1 처리 로직을 layer 단위로 풀어쓴 버전
# 빌드별로 _last_overhang_layer (H,W) 상태를 z-축 carry-over (baseline 과 동일)
EDGE_SAT = config.DIST_EDGE_SATURATION_MM           # 3.0
OH_SAT   = config.DIST_OVERHANG_SATURATION_LAYERS   # 71

last_overhang = np.full((H, W), -np.inf, dtype=np.float32)
prev_cad = None

for layer in range(num_layers):
    cad_mask_layer = (part_ds[layer] > 0)               # bool, 픽셀곱용
    if cad_mask_layer.any():
        dist_edge = distance_transform_edt(cad_mask_layer) * pixel_size_mm
        dist_edge = np.minimum(dist_edge, EDGE_SAT)
        dist_edge = gaussian_filter(dist_edge, sigma=sigma_px)
    else:
        dist_edge = np.zeros((H, W), dtype=np.float32)

    if prev_cad is not None:
        overhang = cad_mask_layer & ~prev_cad
        if overhang.any():
            last_overhang[overhang] = float(layer)
    dist_oh = float(layer) - last_overhang
    dist_oh = np.minimum(dist_oh, OH_SAT)
    dist_oh = gaussian_filter(dist_oh, sigma=sigma_px)
    prev_cad = cad_mask_layer.copy()

    # === inversion + 픽셀곱 (layer 전체 맵 단위로 한 번에) ===
    edge_prox = (EDGE_SAT - dist_edge) * cad_mask_layer.astype(np.float32)
    oh_prox   = (OH_SAT   - dist_oh)   * cad_mask_layer.astype(np.float32)

    # 이 layer 가 활성인 SV 들에 대해 8×8 패치 잘라서 저장 (평균 X)
    for sv with valid_mask[off]==True at this layer:
        seq[ti, 0, :, :] = edge_prox[r0:r1, c0:c1]      # (h, w) 그대로
        seq[ti, 1, :, :] = oh_prox  [r0:r1, c0:c1]
```

> **상태 carry-over 주의**: `last_overhang` 은 빌드 전체에 걸쳐 유지되므로 **layer 순서대로 처리** 해야 함. SV 순서가 아니라 layer-major 루프 사용. ([`baseline/features.py:71-79`](../baseline/features.py#L71-L79) 와 동일 규칙)

> **패치 크기 일관**: 카메라 v0 캐시와 동일하게 `int(round(7.52))=8` px 정사각형, 가장자리 SV 는 image bound 로 clip 되어 < 8 가능. `lstm/crop_stacks.py` 의 zero-pad 로직 그대로 차용 — 부족분은 0 (= nominal) 으로 채움.

**예상 비용**: 빌드당 ~20-35분 (mean 대신 패치 저장 + inversion + 픽셀곱 약간 추가). 디스크 비용은 빌드당 ~5 MB (float16, N≈2k × 70 × 2 × 64 × 2B).

### 3.2 신규 캐시 — `scan_patch_stacks_{B}.h5`

baseline `_extract_scan_features_block`([`features.py:260-308`](../baseline/features.py#L260-L308)) 의 layer 단위 픽셀 맵을 **평균하지 않고** SV 8×8 패치 그대로 저장. **inversion 불필요** (raw 이미 "0=nominal, large=signal" 컨벤션). **마스킹 불필요** — baseline 의 [`scan_features.py`](../baseline/scan_features.py) 가 이미 미용융 픽셀에 0 (stripe_boundaries) / NaN→0 (return_delay 는 본 캐시에서 변환) 을 채움.

**입력**: `scans/{layer}` (레이저 경로 5열 raw).

**출력 dataset**:
- `scan_patch` : `(N, T_max=70, 2, h, w) float16`, zero-padded — `h=w=8` (대부분)
  - 채널 0: `return_delay` raw 8×8 패치 (s, saturation 0.75, **NaN→0 변환**)
  - 채널 1: `stripe_boundaries` raw 8×8 패치 (a.u., Sobel RMS, baseline 이 이미 미용융=0)
  - 둘 다 미용융 픽셀 = 0 (= nominal). 컨벤션 자연 일치.
- `lengths` : `(N,) int16` — v0 캐시와 동일 (CAD valid_mask 기준; melt-only 마스크 아님)
- `sv_indices`, `sample_ids` : v0 와 동일
- attrs:
  - `inversion_applied = False`
  - `mask_applied = False`  (baseline 픽셀 맵 자체에 0 채워져 있어 곱셈 불필요)
  - `return_delay_saturation_s = 0.75`
  - `channel_names = ["return_delay", "stripe_boundaries"]`
  - `convention = "0 = no melt (nominal), large = signal"`

**빌드 절차**:

```python
for layer in range(num_layers):
    if f"scans/{layer}" not in hdf5_file:
        # 이 layer 는 scan 데이터 부재 → seq[ti, :, :, :] = 0 (= nominal) 으로 둠
        continue
    scans = f[f"scans/{layer}"][...]
    mt = build_melt_time_map(scans, (H, W), pixel_size_mm)            # (H, W) NaN at non-melt
    rd_map = compute_return_delay_map(mt, kernel_px=8, sat_s=0.75)    # NaN at non-melt
    sb_map = compute_stripe_boundaries_map(mt)                         # 0 at non-melt (baseline 이 이미 처리)

    # === NaN → 0 변환 (CNN 입력으로 NaN 흘려보내면 폭발) ===
    rd_map = np.nan_to_num(rd_map, nan=0.0, copy=False)                # 0 = no melt = nominal

    # 이 layer 가 활성인 SV 들에 대해 8×8 패치 잘라서 저장 (평균 X, 마스킹 X)
    for sv with valid_mask[off]==True at this layer:
        seq[ti, 0, :, :] = rd_map[r0:r1, c0:c1]                        # (h, w) 그대로
        seq[ti, 1, :, :] = sb_map[r0:r1, c0:c1]
```

> **마스킹 안 하는 이유 (사용자 결정)**: baseline `scan_features.py` 가 미용융 픽셀에 명시적 0 을 부여 ([§G4 #21](../FEATURES.md#21-laser_stripe_boundaries-au--sobel-rms)). return_delay 도 NaN→0 변환 후 동일 컨벤션. 추가 곱셈 없이도 0 = nominal 일관 보장.

> **활성 layer 인데 melt 0 인 경우**: 시퀀스에 0 패치가 흘러감 — baseline 의 "melt 픽셀 0개 → 0 채움" 의도와 동일 처리.

> **lengths 일치**: scan_patch 의 valid 시점은 melt 픽셀 유무지만, 캐시는 v0 의 CAD valid_mask 기준 lengths 로 정렬해 저장 (활성 layer 인데 melt 0 인 곳은 패치 전체 0 으로 채움). 모든 분기의 lengths 를 동일하게 만들어 packed LSTM 통합 처리 가능.

**예상 비용**: 빌드당 ~15-25분. `build_melt_time_map` + 패치 저장 추가 비용. 디스크 비용은 빌드당 ~5 MB (float16).

### 3.3 일치 검증

cache_cad_patch / cache_scan_patch 둘 다 v0 캐시와 lengths/sv_indices/sample_ids 비트 단위 일치 + 패치 크기 (8×8) + attrs (`inversion_applied`, `mask_applied`) 빌드 후 검증:
- `verify_cad_patch_v0_consistency`
- `verify_scan_patch_v0_consistency`

추가 sanity check (cad 만):
- `cad_patch[..., 0]` 값 범위 ∈ [0, 3.0]
- `cad_patch[..., 1]` 값 범위 ∈ [0, 71]
- 분말 영역 픽셀 (cad_mask=0 위치) 값 = 0 (랜덤 샘플 100개 SV)

---

## 4. 모델 변경 (`model.py`)

### 4.1 신규 분기: `_PerFieldConv1DBranch`

```python
class _PerFieldConv1DBranch(nn.Module):
    """필드별 독립 1D-CNN. (B, T, n_fields) → (B, n_fields * d_per_field).

    각 필드를 1채널 1D-CNN 으로 통과 → AdaptiveAvgPool → flatten → linear.
    필드간 weight sharing 없음 (필드별 ModuleList).
    """

    def __init__(self, n_fields: int, d_per_field: int = 3,
                 hidden_ch: int = 16, kernel_size: int = 5):
        super().__init__()
        self.n_fields = n_fields
        self.d_per_field = d_per_field
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Conv1d(hidden_ch, hidden_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(d_per_field),  # (B, hidden_ch, d_per_field)
            )
            for _ in range(n_fields)
        ])
        self.proj = nn.ModuleList([
            nn.Linear(hidden_ch * d_per_field, d_per_field)  # (B, d_per_field)
            for _ in range(n_fields)
        ])

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_fields) — 패딩 영역은 0 (zero-pad)
        # lengths: (B,) — AdaptiveAvgPool 사용 → lengths 미사용 (1차 구현)
        outs = []
        for fi in range(self.n_fields):
            xf = x[:, :, fi].unsqueeze(1)               # (B, 1, T)
            yf = self.convs[fi](xf)                     # (B, hidden_ch, d_per_field)
            yf = yf.flatten(1)                          # (B, hidden_ch * d_per_field)
            yf = self.proj[fi](yf)                      # (B, d_per_field)
            outs.append(yf)
        return torch.cat(outs, dim=1)                   # (B, n_fields * d_per_field)
```

> **AdaptiveAvgPool1d 채택 (사용자 결정)**: 시간축 고정 길이 출력으로 압축. lengths 무시 → 패딩(0) 영역도 평균에 섞임. T_sv 중간값 ~50 가정 시 패딩 비율 ~30%, 신호 희석 정도는 0.7× 약화 수준. 결과가 LSTM-sensor_7 대비 평탄하면 lengths-aware mean 으로 업그레이드 검토 ([§11.5](#11-주의-사항)).
>
> **Hyperparam 기본**: `hidden_ch=16`, `kernel_size=5`, `d_per_field=4` — sensor_7 의 d_embed=7 (필드 7개 × 1) 대비 본 실험은 7×4=28 (4배 표현력). d=3 (early/mid/late 만) 대비 1차 미분 신호까지 살림.

### 4.2 분기 — 다채널 시퀀스 LSTM (`_GroupLSTMBranch`, DSCNN 전용)

```python
class _GroupLSTMBranch(nn.Module):
    """다채널 스칼라 시퀀스 → LSTM → proj. **DSCNN 전용** (CAD/Scan 은 spatial 분기 사용)."""

    def __init__(self, n_channels: int, d_hidden: int, d_embed: int,
                 num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=d_hidden,
            num_layers=num_layers, batch_first=True, bidirectional=bidirectional,
        )
        d_lstm_out = d_hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(d_lstm_out, d_embed)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_channels), lengths: (B,) int64 (cpu)
        packed = pack_padded_sequence(x, lengths.detach().cpu(),
                                       batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return self.proj(h_last)
```

> DSCNN 은 baseline 캐시 (`dscnn_stacks_*.h5`) 가 이미 (T, 8) 스칼라 시퀀스라 LSTM 단독 처리. 본 실험은 DSCNN 의 spatial 정보 까지 살리진 않음 (별도 후속 실험).

### 4.3 분기 — Spatial-CNN + LSTM (카메라 / CAD / Scan 공용)

카메라 v0/v1 분기 (`_LSTMBranch`) 를 `in_channels` 파라미터로 일반화해 CAD / Scan 도 동일 구조 재사용.

```python
class FrameCNN(nn.Module):
    """SV 8×8 패치 → d_cnn 임베딩. in_channels 파라미터로 카메라(1ch)/CAD(2ch)/Scan(2ch) 공용.

    기존 `lstm/model.py:FrameCNN` 의 in_channels=1 하드코딩을 본 실험에서 일반화.
    카메라 v0/v1 호출은 in_channels=1 디폴트로 동작 변화 없음.
    """

    def __init__(self, d_cnn: int = 32, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, config.LSTM_CNN_CH1, 3, padding=1)
        self.conv2 = nn.Conv2d(config.LSTM_CNN_CH1, config.LSTM_CNN_CH2, 3, padding=1)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.proj  = nn.Linear(config.LSTM_CNN_CH2, d_cnn)

    def forward(self, x):
        # x: (B*T, in_channels, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.gap(x).flatten(1)        # (B*T, ch2)
        return self.proj(x)               # (B*T, d_cnn)


class _LSTMBranch(nn.Module):
    """spatial 패치 시퀀스 → per-frame CNN → LSTM → proj.

    재사용 케이스:
      - 카메라 v0:  in_channels=1, d_embed=16
      - 카메라 v1:  in_channels=1, d_embed=16
      - CAD-patch:  in_channels=2, d_embed=8
      - Scan-patch: in_channels=2, d_embed=8
    """

    def __init__(self, in_channels: int, d_cnn: int, d_hidden: int, d_embed: int,
                 num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.bidirectional = bidirectional
        self.cnn = FrameCNN(d_cnn=d_cnn, in_channels=in_channels)
        self.lstm = nn.LSTM(d_cnn, d_hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        d_lstm_out = d_hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(d_lstm_out, d_embed)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) — 카메라(C=1) / CAD(C=2) / Scan(C=2)
        # 호환: 카메라 캐시는 (B, T, H, W) 로 들어오면 unsqueeze(2) 로 channel dim 추가
        if x.dim() == 4:                              # 카메라 v0/v1 backward compat
            x = x.unsqueeze(2)                        # (B, T, 1, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.cnn(x)                               # (B*T, d_cnn)
        x = x.reshape(B, T, -1)                       # (B, T, d_cnn)
        packed = pack_padded_sequence(x, lengths.detach().cpu(),
                                       batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return self.proj(h_last)                      # (B, d_embed)
```

> 기존 `lstm_dual.model._LSTMBranch` 를 본 실험에서 in_channels 일반화한 버전으로 import (또는 본 실험 model.py 안에 신규 정의). 카메라 분기 호출은 동작 변화 없음.

### 4.4 메인 모델

```python
class VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_1DCNN_Sensor_4(nn.Module):
    def __init__(self,
                 # 카메라 (d_embed 16, in_channels=1)
                 d_cnn=config.LSTM_D_CNN,
                 d_hidden_cam=16,
                 d_embed_v0=16, d_embed_v1=16,
                 # sensor per-field 1D-CNN
                 n_sensor_fields=7, d_per_sensor_field=4,
                 sensor_hidden_ch=16, sensor_kernel=5,
                 # DSCNN LSTM (스칼라 시퀀스)
                 n_dscnn_ch=8, d_hidden_d=16, d_embed_d=8,
                 # CAD spatial-CNN+LSTM (in_channels=2, d=8)
                 n_cad_ch=2,   d_cnn_c=32,   d_hidden_c=16,  d_embed_c=8,
                 # Scan spatial-CNN+LSTM (in_channels=2, d=8)
                 n_scan_ch=2,  d_cnn_sc=32,  d_hidden_sc=16, d_embed_sc=8,
                 # 결합 MLP — 86 → 256 → 128 → 64 → 1
                 mlp_hidden=(256, 128, 64), dropout=config.DROPOUT_RATE):
        super().__init__()

        # 카메라 v0/v1 — spatial-CNN+LSTM (in_channels=1)
        self.branch_v0 = _LSTMBranch(in_channels=1, d_cnn=d_cnn, d_hidden=d_hidden_cam,
                                     d_embed=d_embed_v0)
        self.branch_v1 = _LSTMBranch(in_channels=1, d_cnn=d_cnn, d_hidden=d_hidden_cam,
                                     d_embed=d_embed_v1)
        # Sensor — per-field 1D-CNN
        self.branch_sensor = _PerFieldConv1DBranch(
            n_sensor_fields, d_per_sensor_field, sensor_hidden_ch, sensor_kernel,
        )
        # DSCNN — 스칼라 시퀀스 LSTM
        self.branch_dscnn = _GroupLSTMBranch(n_dscnn_ch, d_hidden_d, d_embed_d)
        # CAD — spatial-CNN+LSTM (in_channels=2, edge_proximity + overhang_proximity)
        self.branch_cad = _LSTMBranch(in_channels=n_cad_ch, d_cnn=d_cnn_c,
                                       d_hidden=d_hidden_c, d_embed=d_embed_c)
        # Scan — spatial-CNN+LSTM (in_channels=2, return_delay + stripe_boundaries)
        self.branch_scan = _LSTMBranch(in_channels=n_scan_ch, d_cnn=d_cnn_sc,
                                        d_hidden=d_hidden_sc, d_embed=d_embed_sc)

        n_static = 2                                                            # build_height + laser_module
        n_total = (n_static
                   + d_embed_v0 + d_embed_v1
                   + n_sensor_fields * d_per_sensor_field                       # 28
                   + d_embed_d + d_embed_c + d_embed_sc)                        # 8 + 8 + 8 = 24
        # = 2 + 16 + 16 + 28 + 8 + 8 + 8 = 86
        # MLP: 86 → 256 → 128 → 64 → 1 (4 fc layer)
        h1, h2, h3 = mlp_hidden                                                 # (256, 128, 64)
        self.fc1 = nn.Linear(n_total, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, 1)
        self.dropout = nn.Dropout(dropout)
        self._init_mlp_weights()

    def _init_mlp_weights(self):
        projs = [self.branch_v0.proj, self.branch_v1.proj,
                 self.branch_dscnn.proj, self.branch_cad.proj, self.branch_scan.proj]
        # _PerFieldConv1DBranch 의 필드별 proj 도 포함
        for proj in self.branch_sensor.proj:
            projs.append(proj)
        for m in (self.fc1, self.fc2, self.fc3, self.fc4, *projs):
            nn.init.normal_(m.weight, mean=0.0, std=config.WEIGHT_INIT_STD)
            nn.init.zeros_(m.bias)

    def forward(self,
                feats_static: torch.Tensor,                  # (B, 2)
                stacks_v0: torch.Tensor, stacks_v1: torch.Tensor,    # (B, T, 8, 8) - 호환: (B, T, 1, 8, 8) 도 가능
                sensors: torch.Tensor,                       # (B, T, 7)
                dscnn: torch.Tensor,                         # (B, T, 8)
                cad_patch: torch.Tensor,                     # (B, T, 2, 8, 8)
                scan_patch: torch.Tensor,                    # (B, T, 2, 8, 8)
                lengths: torch.Tensor) -> torch.Tensor:
        e_v0 = self.branch_v0(stacks_v0, lengths)            # (B, 16)
        e_v1 = self.branch_v1(stacks_v1, lengths)            # (B, 16)
        e_s  = self.branch_sensor(sensors, lengths)          # (B, 28)
        e_d  = self.branch_dscnn(dscnn, lengths)             # (B, 8)
        e_c  = self.branch_cad(cad_patch, lengths)           # (B, 8)
        e_sc = self.branch_scan(scan_patch, lengths)         # (B, 8)
        x = torch.cat([feats_static, e_v0, e_v1, e_s, e_d, e_c, e_sc], dim=1)   # (B, 86)
        x = F.relu(self.fc1(x)); x = self.dropout(x)                            # → 256
        x = F.relu(self.fc2(x)); x = self.dropout(x)                            # → 128
        x = F.relu(self.fc3(x)); x = self.dropout(x)                            # → 64
        return self.fc4(x)                                                      # → 1
```

### 4.5 정적 피처 추출

```python
# build_normalized_dataset 안에서:
# 21-feat 중 P4 두 개만 추출
static_idx = [2, 18]                                                            # build_height, laser_module
feats_static = features_21[:, static_idx]                                       # (N, 2)
```

`build_height` 는 [-1, 1] 정규화, `laser_module` 은 binary {0, 1} 그대로 (정규화 시 동일 매핑).

---

## 5. 데이터셋 (`dataset.py`)

`load_septet_dataset` (7-입력 결합):

```python
def load_septet_dataset(...) -> dict:
    """returns {features: (N,21), stacks_v0/v1, sensors, dscnn, cad_patch, scan_patch, lengths, ...}

    cad_patch:  (N, 70, 2, 8, 8) float16 — channel 0=edge_proximity, 1=overhang_proximity
    scan_patch: (N, 70, 2, 8, 8) float16 — channel 0=return_delay,   1=stripe_boundaries
    """
    # 6 캐시 로드 (v0/v1/sensor/dscnn/cad_patch/scan_patch) + 21-feat npz
    # lengths/sv_indices/sample_ids 6-way 일치 검증
    # cad_patch attrs 의 inversion_applied=True / mask_applied=True 확인
    ...

def build_normalized_dataset(raw):
    """static 2-feat (build_height, laser_module) + 6 시퀀스 입력 정규화."""
    # static: 21-feat 에서 idx [2, 18] 추출 → min-max [-1, 1]
    # sensors / dscnn:        per-channel min-max [-1, 1] (시간축 평탄화 후 패딩 0 제외)
    # cad_patch / scan_patch: per-channel min-max [-1, 1]
    #   - flatten 시점: spatial 차원 (h, w) 까지 평탄화 → 채널 차원에 대해서만 min/max 추정
    #   - cad_patch 의 분말 영역 0 (mask 곱셈 결과) 도 정규화 통계에 포함 — 0 이 가장 흔한 값이라 분포 기형 우려.
    #     해결: 0 픽셀 (=분말) 을 통계 집계에서 제외 → real 값 분포로만 min/max 계산.
    # 카메라 stacks 는 baseline 과 동일하게 raw float16 보존 (CNN 안에서 처리)
    ...

class VPPMLSTMSeptetDataset(Dataset):
    def __getitem__(self, i):
        return (self.features_static[i],
                self.stacks_v0[i], self.stacks_v1[i],
                self.sensors[i], self.dscnn[i],
                self.cad_patch[i], self.scan_patch[i],
                self.lengths[i], self.targets[i])
```

---

## 6. config.py 추가

```python
# ============================================================
# VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-1DCNN-Sensor-4
# (Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md)
# 짧은 별칭 prefix: LSTM_FULL86_  (정식 이름이 너무 길어서 86-feat MLP 입력 차원 기반)
# ============================================================
LSTM_FULL86_D_HIDDEN_CAM = 16
LSTM_FULL86_D_EMBED_V0   = 16
LSTM_FULL86_D_EMBED_V1   = 16

LSTM_FULL86_N_SENSOR_FIELDS    = 7
LSTM_FULL86_D_PER_SENSOR_FIELD = 4
LSTM_FULL86_SENSOR_HIDDEN_CH   = 16
LSTM_FULL86_SENSOR_KERNEL      = 5

LSTM_FULL86_N_DSCNN_CH = 8
LSTM_FULL86_D_HIDDEN_D = 16
LSTM_FULL86_D_EMBED_D  = 8

LSTM_FULL86_N_CAD_CH      = 2                                                  # edge_proximity + overhang_proximity
LSTM_FULL86_CAD_PATCH_H   = 8
LSTM_FULL86_CAD_PATCH_W   = 8
LSTM_FULL86_D_CNN_C       = 32                                                 # cad spatial CNN 출력 차원
LSTM_FULL86_D_HIDDEN_C    = 16
LSTM_FULL86_D_EMBED_C     = 8                                                  # 채널당 4-dim

LSTM_FULL86_N_SCAN_CH     = 2                                                  # return_delay + stripe_boundaries
LSTM_FULL86_SCAN_PATCH_H  = 8
LSTM_FULL86_SCAN_PATCH_W  = 8
LSTM_FULL86_D_CNN_SC      = 32                                                 # scan spatial CNN 출력 차원
LSTM_FULL86_D_HIDDEN_SC   = 16
LSTM_FULL86_D_EMBED_SC    = 8                                                  # 채널당 4-dim

# CAD 캐시 attrs (cache_cad_patch.py 빌드 시 기록)
LSTM_FULL86_CAD_INVERSION_APPLIED = True                                       # 3.0 - dist / 71 - dist
LSTM_FULL86_CAD_MASK_APPLIED      = True                                       # cad_mask 픽셀곱
# Scan 캐시 attrs (cache_scan_patch.py 빌드 시 기록)
LSTM_FULL86_SCAN_INVERSION_APPLIED = False                                     # raw 그대로 (이미 0=nominal 컨벤션)
LSTM_FULL86_SCAN_MASK_APPLIED      = False                                     # baseline 처리가 미용융=0 부여

LSTM_FULL86_STATIC_IDX = [2, 18]                                               # build_height, laser_module

# 결합 MLP — 86 → 256 → 128 → 64 → 1 (4 fc layer, baseline 의 1-layer 보다 깊음)
LSTM_FULL86_MLP_HIDDEN = (256, 128, 64)

LSTM_FULL86_EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / "vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4"
LSTM_FULL86_CACHE_DIR      = LSTM_FULL86_EXPERIMENT_DIR / "cache"
LSTM_FULL86_MODELS_DIR     = LSTM_FULL86_EXPERIMENT_DIR / "models"
LSTM_FULL86_RESULTS_DIR    = LSTM_FULL86_EXPERIMENT_DIR / "results"
LSTM_FULL86_FEATURES_DIR   = LSTM_FULL86_EXPERIMENT_DIR / "features"

# 카메라 / sensor / dscnn 캐시는 기존 디렉터리 재사용
LSTM_FULL86_CACHE_V0_DIR     = LSTM_CACHE_DIR
LSTM_FULL86_CACHE_V1_DIR     = LSTM_DUAL_CACHE_DIR
LSTM_FULL86_CACHE_SENSOR_DIR = LSTM_DUAL_IMG_4_SENSOR_7_CACHE_DIR
LSTM_FULL86_CACHE_DSCNN_DIR  = LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_DIR
# cad_patch / scan_patch 는 본 실험 디렉터리에 신규 빌드
LSTM_FULL86_CACHE_CAD_DIR    = LSTM_FULL86_CACHE_DIR
LSTM_FULL86_CACHE_SCAN_DIR   = LSTM_FULL86_CACHE_DIR
```

---

## 7. 변경 사항 요약 (dscnn_8 대비)

| 항목 | dscnn_8 (기존) | **본 실험 (신규)** |
|:--:|:--:|:--:|
| 카메라 v0/v1 d_embed | 4 / 4 | **16 / 16** |
| 카메라 d_hidden | 16 | 16 (proj 매트릭스 16→16) |
| Sensor 분기 | 7-ch LSTM (d=7) | **7개 1D-CNN, 필드당 4-dim (총 28)** |
| DSCNN 분기 | 8-ch LSTM (d=8) | 동일 |
| **CAD 분기** | **없음** (#1, #2 평균 → baseline 6-feat 에 포함) | **spatial-CNN+LSTM**, in_channels=2, 8×8 패치 보존 (d=8). **inversion + cad_mask 픽셀곱** 적용 |
| **Scan 분기** | **없음** (#20, #21 평균 → baseline 6-feat 에 포함) | **spatial-CNN+LSTM**, in_channels=2, 8×8 패치 보존 (d=8). raw 그대로 (baseline 이 미용융=0 처리) |
| Baseline feat | 6 (G3+G4 = #1, #2, #3, #19, #20, #21) | **2** (#3, #19 정적만) |
| MLP 입력 차원 | 29 | **86** |
| MLP 구조 | 1 hidden (29→128→1) | **3 hidden (86→256→128→64→1)** |
| 신규 캐시 | dscnn (1) | **cad_patch + scan_patch (2, 8×8 패치 보존)** |
| 학습 hp | 1e-3 / 256 / 50 | 동일 (controlled 비교 위해 유지) |

### 파라미터 카운트 추정 (대략, 1-layer LSTM 기준)

| 분기 | 파라미터 | 비고 |
|:--|:--:|:--|
| 카메라 FrameCNN ×2 (in=1) | ~10k ×2 ≈ 20k | 기존과 동일 |
| 카메라 LSTM ×2 (in=32, hid=16) | ~3k ×2 ≈ 6k | proj 16→16 = 256 (기존 16→4=64 대비 +192) |
| Sensor 1D-CNN ×7 (1→16→16, k=5, pool→4, proj 64→4) | ~2.2k ×7 ≈ 15k | sensor LSTM (~2k) 대비 ~7-8x. 단 표현력 28-dim 으로 +21 dim |
| DSCNN LSTM (in=8, hid=16) | ~1.5k | 기존과 동일 |
| **CAD FrameCNN (in=2)** + **LSTM (in=32, hid=16)** + proj | ~13k | spatial-CNN 추가로 LSTM-only (1.2k) 대비 ~11× |
| **Scan FrameCNN (in=2)** + **LSTM (in=32, hid=16)** + proj | ~13k | 동일 |
| MLP (86→256→128→64→1) | ~58k | fc1=22k, fc2=33k, fc3=8.3k, fc4=65 — dscnn_8 (29→128→1, ~4k) 대비 ~14x |
| **합계** | **~125-135k** | dscnn_8 (~30k) 대비 약 4.3배. **과적합 모니터링 최우선** |

> 6,373 SV 데이터 대비 ~130k 파라미터는 ~5 SV/param. **과적합 위험이 dscnn_8 대비 매우 높음**. 1차 학습은 dropout 0.1 그대로 진행하되 train↔val gap 모니터링 필수. gap 이 커지면 (1) dropout 0.1 → 0.2-0.3, (2) MLP 폭 축소 (256→192, 128→96, 64→48), (3) cad/scan d_cnn 32→16 으로 spatial CNN 축소, (4) early stopping patience 50 → 30 순으로 조정.

---

## 8. 디렉터리 / 파일 구조

```
Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/
├── PLAN.md                     (이 파일)
├── __init__.py
├── cache_cad_patch.py          # P1 (#1, #2) 8×8 패치 시퀀스 캐시 빌드 (inversion + cad_mask 픽셀곱)
├── cache_scan_patch.py         # P2 (#20, #21) 8×8 패치 시퀀스 캐시 빌드 (raw + NaN→0)
├── dataset.py                  # 7-입력 로드/정규화 (load_septet_dataset)
├── model.py                    # FrameCNN(in_channels) + _LSTMBranch(in_channels) 일반화 + _PerFieldConv1DBranch + _GroupLSTMBranch + 메인 모델
├── train.py                    # forward 시그니처 (8 인자)
├── evaluate.py
└── run.py                      # cache_cad_patch / cache_scan_patch / train / evaluate 진입점

Sources/vppm/common/config.py   # LSTM_FULL86_* 상수 추가

Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/
├── cache/
│   ├── cad_patch_stacks_B1.{1..5}.h5     # 빌드당 ~5 MB (float16, 2-ch, 8×8 패치)
│   └── scan_patch_stacks_B1.{1..5}.h5    # 빌드당 ~5 MB
├── models/                     # 4 props × 5 folds = 20 .pt
├── results/
│   ├── metrics_raw.json
│   ├── metrics_summary.json
│   ├── predictions_{YS,UTS,UE,TE}.csv
│   ├── correlation_plots.png
│   └── scatter_plot_uts.png
├── features/normalization.json # static(2) + sensor(7) + dscnn(8) + cad_patch(2 ch, 분말 0 제외 통계) + scan_patch(2 ch) min/max
└── experiment_meta.json
```

---

## 9. 실행 단계

| Phase | 작업 | 산출물 | 예상 시간 |
|:--:|:--|:--|:--:|
| **S0** | `config.py` 에 `LSTM_FULL86_*` 추가, 디렉터리 생성 | (코드) | 10분 |
| **S1a** | `cache_cad_patch.py` 구현 (inversion + cad_mask 픽셀곱) + 5빌드 캐시 빌드 | `cad_patch_stacks_*.h5` × 5 | 빌드당 ~20-35분 |
| **S1b** | `cache_scan_patch.py` 구현 (NaN→0 변환) + 5빌드 캐시 빌드 | `scan_patch_stacks_*.h5` × 5 | 빌드당 ~15-25분 |
| **S1c** | v0 캐시와 lengths/sv_indices/sample_ids 비트 일치 + cad attrs (`inversion_applied`/`mask_applied`) + 채널 범위 sanity check | verify 통과 로그 | 5분 |
| **S2** | `dataset.py` — 7-입력 로드 + 정규화 (cad/scan 은 spatial 차원까지 평탄화 후 channel-wise min-max) | dataset.py | 1h |
| **S3** | `model.py` — FrameCNN/`_LSTMBranch` in_channels 일반화 + `_PerFieldConv1DBranch` + `_GroupLSTMBranch` + 메인 모델 | model.py | 1.5h |
| **S4** | `train.py` / `evaluate.py` — forward 호출 8-인자 시그니처 반영 | train.py, evaluate.py | 30분 |
| **S5** | `run.py` 진입점 + `experiment_meta.json` | run.py | 20분 |
| **S6** | smoke test (`--quick` 1 fold × YS, epochs=10) — forward/backward 통과 + lengths 검증 | smoke 로그 | 15분 |
| **S7** | `docker/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/` 환경 — `docker-setup` 서브에이전트 | Dockerfile/compose | 20분 |
| **S8** | **풀런 — 사용자 실행** (4 props × 5 folds, ~4-5h GPU 가정) | metrics_*.json, plots | ~4-5h |
| **S9** | `RESULTS.md` — 가설 검증, dual_4 / sensor_7 / dscnn_8 / 본 실험 누적 비교 + ablation 분석 | RESULTS.md | 1.5h |

> dscnn_8 대비 추가 작업: **신규 캐시 2종 (8×8 패치 보존) + 신규/일반화 분기 3종 (sensor 1D-CNN, CAD spatial-CNN+LSTM, scan spatial-CNN+LSTM) + FrameCNN/_LSTMBranch in_channels 일반화 + MLP 입력 차원 29→86 + MLP 깊이 1 hidden → 3 hidden (256/128/64)**.

---

## 10. 가설별 기대치

| 시나리오 | 예상 결과 | 해석 |
|:--|:--|:--|
| **A. 카메라 d_embed=16 + 시간성 정보 전체 복원이 의미** | dscnn_8 대비 RMSE −2~−5 % (UTS / UE / TE 중심) | 이미지 spatial-temporal 압축 손실 + sensor 채널별 표현력이 모두 풀림 |
| **B. 과적합** | val 성능 ↓, train↔val gap ↑ (특히 epoch ≥ 100) | **파라미터 ~4.3배 증가** (입력 29→86 + MLP 1 hidden → 3 hidden + cad/scan spatial-CNN 추가, ~30k → ~130k). 6,373 SV 대비 ~5 SV/param 으로 dscnn_8 (~17 SV/param) 대비 매우 빡빡. **본 실험의 가장 큰 위험요인** |
| **C. 카메라 d_embed=16 만 효과 (sensor 1D-CNN / CAD/scan LSTM 평탄)** | dscnn_8 ≈ 본 실험, 단 카메라 component ablation 시 d=16 효과만 잡힘 | 시간성 정보 일부는 평균과 LSTM 차이가 측정한계 미만 |
| **D. Sensor 1D-CNN 이 LSTM 대비 우세** | sensor_7 LSTM(d=7) 결과 위로 +α | 채널별 독립 처리 + 표현력 28-dim |
| **E. CAD/Scan spatial-CNN+LSTM 도 효과** | B1.3(오버행 형상) / B1.4(스패터) 빌드별 RMSE 분해에서 큰 개선 | distance_overhang z-축 + 8×8 spatial 변화 / scan stripe 패턴 spatial 형상 결함 신호 살아남 |

### 빌드별 예측

| 빌드 | 가장 큰 효과 후보 | 분기 |
|:--|:--|:--|
| **B1.5** (리코터 손상) | DSCNN recoater_streaking 시간 패턴 | DSCNN LSTM (이미 dscnn_8 에서 잡힘) |
| **B1.4** (스패터/가스) | Sensor 가스 유량 + scan stripe 패턴 | sensor 1D-CNN + scan LSTM |
| **B1.3** (오버행) | overhang_proximity z-축 변화 + 8×8 spatial 분포 | CAD spatial-CNN+LSTM ⭐ 본 실험 신규 |
| **B1.2** (Keyhole) | DSCNN excessive_melting 연속 패턴 | DSCNN LSTM (이미 dscnn_8) |
| **B1.1** (기준) | 결함 적음 | 차이 거의 없을 것 |

→ **B1.3 RMSE 분해** 가 본 실험의 핵심 차별 검증 포인트 (CAD spatial-CNN+LSTM 효과).

---

## 11. 주의 사항

1. **lengths 통일**: scan_patch 의 raw 출처(melt 픽셀)는 v0 의 valid_mask(CAD) 와 다르지만, 본 실험에서는 v0 의 lengths 로 정렬 → melt 0 인 layer 는 패치 전체 0 으로 채움 (= nominal). LSTM 이 packed 로 처리해도 패딩 영역의 0 은 안 봄.

2. **CAD inversion (사용자 결정)**: baseline 21-feat 의 `distance_from_edge` / `distance_from_overhang` 은 raw 값이 클수록 nominal 이라 다른 시퀀스 입력 (DSCNN/scan/sensor) 과 컨벤션 정반대. cache 빌드 시점에 `proximity = saturation - distance` 로 뒤집어 모든 시퀀스 입력의 padding 0 / 정규화 −1 의미를 "nominal" 로 통일. baseline 결과와 raw 값 부호가 다르므로 직접 비교 불가 — 본 실험은 SV 단위 시퀀스만 사용하니 무관.

3. **CAD pixel-wise mask 곱셈 (사용자 결정)**: 8×8 패치 안 분말 영역 픽셀에 `cad_mask=0` 을 곱해 0 (= nominal) 으로 강제. 별도 mask 채널 두는 옵션도 있었지만 (a) inversion 후 의미 채널의 0 (interior nominal) 과 동일 해석으로 일관 통일, (b) FrameCNN in_channels 2 → 3 회피로 파라미터 약간 절약, 두 가지 이유로 곱셈 방식 채택. 분말 비율 정보는 spatial 패턴(0 픽셀 분포)으로 CNN 이 암시적 학습 가능.

4. **Scan inversion / 마스킹 안 함 (사용자 결정)**: scan 의 raw 컨벤션은 "0=no melt(nominal), large=signal" 로 이미 다른 채널과 정렬됨 → inversion 불필요. baseline `scan_features.py` 가 미용융 픽셀에 명시적 0 (#21 stripe_boundaries) / NaN→0 (#20 return_delay 캐시 시 변환) 부여 → 추가 곱셈 없이도 컨벤션 일관.

5. **CAD-temporal 의 layer-major 처리**: `_last_overhang_layer` 상태는 layer-major 로 carry-over. SV-major 루프로 짜면 상태가 깨짐. baseline `extract_features` 의 진입 지점에서 1회 초기화하는 패턴([`features.py:71-79`](../baseline/features.py#L71-L79)) 을 캐시 빌드 시 유지.

6. **정규화 통계**: cad_patch / scan_patch / sensor / dscnn 모두 **패딩 0 제외 per-channel min-max [-1, 1]**.
   - cad_patch / scan_patch 는 spatial 차원 (h, w) 까지 평탄화 후 channel-wise min/max 추정.
   - **cad_patch 는 분말 영역 0 (mask 곱셈 결과) 도 통계 집계에서 제외** 추천 — 0 이 가장 흔한 값이라 분포 기형 우려. 실제 의미 픽셀의 분포로 min/max 계산해야 정규화 [-1, +1] 범위가 의미 있게 펼쳐짐.

7. **`laser_module` 정규화**: binary {0, 1} 이지만 min-max 적용하면 {-1, +1} 로 매핑 → MLP 입력 단위가 다른 정적 피처 (build_height) 와 호환됨. 이는 baseline 과 동일.

8. **AdaptiveAvgPool 패딩 누설**: 1D-CNN sensor 분기는 AdaptiveAvgPool 로 시간축 압축 → 패딩(0) 이 평균에 섞여 신호 ~30% 희석. 결과가 LSTM-sensor_7 대비 평탄(시나리오 C) 하면 lengths-aware mean (lengths 마스크 곱한 sum/count) 으로 업그레이드. 1차는 단순 구조로 진행.

9. **FrameCNN in_channels 일반화**: 기존 [`lstm/model.py`](../lstm/model.py) 의 `FrameCNN` 은 `in_channels=1` 하드코딩. 본 실험에서 파라미터화한 신규 클래스 사용 (또는 본 실험 디렉터리 model.py 안에 신규 정의). 카메라 v0/v1 호출은 `in_channels=1` 디폴트로 동작 변화 없음.

10. **gradient 안정성**: 분기 7개로 늘어나면 backward 시간 ↑. AMP 적용 검토 (현재 baseline 은 fp32). 1차는 fp32 로 진행.

---

## 12. 후속 실험 분기 (본 실험 결과 의존)

| 결과 시나리오 | 다음 실험 |
|:--|:--|
| **A. 전반적 +2~5 %** | sensor 1D-CNN ablation — kernel size, hidden_ch, d_per_field (2/3/5/8) sweep |
| **B. 과적합** | dropout 강화 + sensor hidden_ch 축소 + early stopping patience 조정 |
| **C. 카메라 d=16 만 효과** | `lstm_dual_16` (이미지만 d=16, 그 외 dscnn_8 와 동일) ablation 으로 격리 |
| **D. Sensor 1D-CNN 우세** | sensor 필드 sub-ablation — 어느 필드의 시간 패턴이 핵심인지 분리 |
| **E. CAD/Scan spatial-CNN+LSTM 효과** | overhang_proximity 단독 분기 ablation — B1.3 빌드 정밀 평가; DSCNN 도 spatial-CNN+LSTM 으로 확장 (별도 실험) |

---

## 13. 참조

- **직접 베이스**: [`Sources/vppm/lstm_dual_img_4_sensor_7_dscnn_8/`](../lstm_dual_img_4_sensor_7_dscnn_8/) (PLAN/cache_dscnn/dataset/model)
- **카메라 분기**: [`Sources/vppm/lstm_dual/model.py`](../lstm_dual/model.py) (`_LSTMBranch`)
- **카메라 v0/v1 캐시**: [`Sources/vppm/lstm/crop_stacks.py`](../lstm/crop_stacks.py), [`lstm_dual/crop_stacks_v1.py`](../lstm_dual/crop_stacks_v1.py)
- **Sensor 캐시 (재사용)**: [`Sources/vppm/lstm_dual_img_4_sensor_7/cache_sensor.py`](../lstm_dual_img_4_sensor_7/cache_sensor.py)
- **DSCNN 캐시 (재사용)**: [`Sources/vppm/lstm_dual_img_4_sensor_7_dscnn_8/cache_dscnn.py`](../lstm_dual_img_4_sensor_7_dscnn_8/cache_dscnn.py)
- **baseline P1/P2 처리 로직**: [`Sources/vppm/baseline/features.py:156-308`](../baseline/features.py#L156-L308) (CAD + scan)
- **baseline scan map 계산**: [`Sources/vppm/baseline/scan_features.py`](../baseline/scan_features.py)
- **21-feat 패턴 분류**: [`Sources/vppm/FEATURES.md` § 평균 처리 방식별 분류](../FEATURES.md#평균-처리-방식별-분류-1d-cnn-시퀀스화-관점)
- **결과 해석 표준**: [`Sources/pipeline_outputs/experiments/vppm_lstm/LSTM_RESULTS.md`](../../pipeline_outputs/experiments/vppm_lstm/LSTM_RESULTS.md)
