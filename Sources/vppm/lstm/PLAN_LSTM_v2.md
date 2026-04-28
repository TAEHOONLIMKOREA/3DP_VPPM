# Sample-LSTM (v2) 실험 계획

> **목적**: VPPM 의 21 차원 핸드크래프트 피처에 **샘플 단위 시퀀스 LSTM 출력** 을 추가해 인장 물성
> 예측 RMSE 를 개선할 수 있는지 측정.
>
> **4 모드 (단/양방향 × 임베딩 1-dim/16-dim)**:
>
> | mode | LSTM | d_embed | n_feats |
> |:----|:----|:--:|:--:|
> | `fwd1`    | Forward       | 1  | 22 |
> | `bidir1`  | Bidirectional | 1  | 22 |
> | `fwd16`   | Forward       | 16 | 37 |
> | **`bidir16`** | **Bidirectional** | **16** | **37** (이전 v1 설계) |
>
> 각 모드 결과를 baseline (21-feat) 대비 모두 비교 — 양방향 효과(fwd vs bidir) 와 임베딩 차원 효과
> (1 vs 16) 의 ablation 가능.
>
> **이전 v1 (제거됨)** 과의 차이: 슈퍼복셀 단위 8×8 패치 + DSCNN 9 채널 → **샘플 단위 bbox-crop +
> raw 단일 채널** 로 단순화. DSCNN 정보는 기존 21 피처에 이미 들어 있으므로 LSTM 입력에서는 제외.

---

## 1. 설계 개요

### 1.1 데이터 흐름

```
HDF5 ── slices/camera_data/visible/0  (T, 1842, 1842) float32 — 용융 직후 카메라
   │    slices/sample_ids              (T, 1842, 1842) uint32 — 픽셀별 sample id
   ▼
[Phase L1] sample_stack 캐시 빌드  (mode 무관, 1회만)
   샘플 s 의 모든 등장 레이어 layers(s) 에 대해:
     bbox = bounding box of (sample_ids[L] == s) for L in layers(s)
     crop = raw[L, bbox] resize → 64×64
   → (T_s, 64, 64) float16 시퀀스
   → Sources/pipeline_outputs/sample_stacks/{build}.h5
   ▼
[Phase L2] LSTM 학습  (mode 별 — fwd / bidir 각각 1회)
   model: CNN encoder → LSTM(hidden=8, bidir=mode_bidir) → Linear(hidden_out → 1)  # 추가 피처
                                                                ↘ Linear(1 → 4)    # 학습 헤드
   loss : L1, [-1,1] 정규화  ·  CV : 5-Fold sample-단위
   → Sources/pipeline_outputs/models_lstm/{mode}/lstm_sample_fold{0-4}.pt
   ▼
[Phase L3] 임베딩 추출 + 피처 통합  (mode 별)
   학습된 LSTM 의 1-dim 출력 (proj 결과) 을 각 샘플에 대해 추출.
   기존 all_features.npz (N_voxels, 21) → all_features_with_lstm_{mode}.npz (N_voxels, 22)
   broadcast: voxel.sample_id 의 1-dim 출력을 21 차원 뒤에 concat.
   ▼
[Phase L4] VPPM 재학습  (mode 별)
   기존 VPPM 파이프라인을 22 차원 입력으로 호출.
   → Sources/pipeline_outputs/results/vppm_lstm_{mode}/
```

### 1.2 핵심 결정 사항

| 항목 | 값 | 근거 |
|:----|:--|:----|
| 입력 채널 | raw camera 0 (용융 직후) 만 | DSCNN 정보는 기존 21 피처에 포함, 이미지 자체 시퀀스 패턴이 추가 신호 |
| Crop 단위 | **샘플 bounding box per layer** | 사용자 지시 — 슈퍼복셀 단위 (v1) 가 아닌 샘플 단위 |
| Crop 크기 | 64 × 64 (resize) | 샘플 마다 bbox 다름 → 통일 필요. 64 = 약 8.5 mm 영역 (pixel 0.13mm) |
| 시퀀스 길이 | **가변 T_s** (샘플이 등장하는 모든 layer) | pack_padded_sequence 사용 |
| LSTM hidden (모든 모드) | 8 | 모드 간 비교 일관성 위해 고정 |
| LSTM 출력 차원 | **1 (mode=*1) / 16 (mode=*16)** | mode 별 |
| 학습 타겟 | YS / UTS / UE / TE 모두 (multi-task) | 1-dim 모드는 강한 bottleneck — 4 물성 모두 설명 필요 |
| LSTM 모드 | **fwd1 / bidir1 / fwd16 / bidir16 — 4 모드** | 양방향 × 차원 ablation |
| 임베딩 broadcast | 같은 sample 의 모든 슈퍼복셀에 동일 d_embed-dim 부여 | 사용자 지시 |
| 최종 모델 | VPPM (2-layer MLP, hidden=128) — 입력 21 → **22 또는 37** | VPPM 구조 동일, 입력 차원만 mode 별 |

### 1.3 21 피처 보존

기존 `all_features.npz` 의 21 차원은 **그대로 유지** — DSCNN 8 채널 평균 (idx 3-10) 도 보존.
LSTM 은 raw 이미지 시퀀스로부터 *추가적인* 표현을 학습.

---

## 2. 데이터 명세

### 2.1 입력

| HDF5 키 | shape | dtype | 의미 |
|:----|:----:|:----:|:----|
| `slices/camera_data/visible/0`  | (T_build, 1842, 1842) | float32 | 레이어별 용융 직후 카메라 |
| `slices/sample_ids`             | (T_build, 1842, 1842) | uint32  | 픽셀별 sample id (0=배경) |
| `samples/test_results/yield_strength`           | (n_samples,) | float32 | 4 인장 물성 |
| `samples/test_results/ultimate_tensile_strength`| 〃           | 〃     | |
| `samples/test_results/uniform_elongation`       | 〃           | 〃     | |
| `samples/test_results/total_elongation`         | 〃           | 〃     | |

### 2.2 샘플 처리 단위

- **유효 샘플**: VPPM origin 파이프라인이 사용하는 6,373 개 (test_result NaN 제외).
- 각 샘플의 layer 범위: `samples/layer_ranges` 또는 `sample_ids == s` 로 직접 검색.
- 빌드별 분포 (예상):
  - B1.1 503 / B1.2 2,705 / B1.3 813 / B1.4 694 / B1.5 1,584
- 평균 layer 수 추정: ~100/샘플 (실측 후 PLAN 갱신).

### 2.3 캐시 구조

`Sources/pipeline_outputs/sample_stacks/{build_id}.h5`:

```python
# h5py 그룹 — 빌드 1개당 파일 1개
build.h5
├── /sample_ids       # (n_samples_in_build,) uint32
├── /seq_lengths      # (n_samples,) int32 — 각 샘플의 layer 수 T_s
├── /layer_offsets    # (n_samples + 1,) int64 — 시퀀스 평탄화 인덱스
├── /sequences        # (sum(T_s), 64, 64) float16 — 평탄화된 모든 시퀀스
├── /layer_ids        # (sum(T_s),) int32 — 각 (샘플, t) 가 원본 어느 layer 인지
└── /bboxes           # (sum(T_s), 4) int32 — 각 frame 의 (r0, r1, c0, c1) bbox
```

용량 추정:
- B1.2: 2705 샘플 × 100 layer × 64×64 × 2 byte ≈ **2.2 GB**
- 5 빌드 합산: **~5 GB**

### 2.4 정규화

- 카메라 raw 값은 빌드 / 레이어 별 분포가 다를 수 있음 → 캐시 빌드 시 **per-build min/max** 로
  [0, 1] 정규화 후 float16 저장.
- 통계는 `Sources/pipeline_outputs/sample_stacks/normalization.json` 에 저장.

---

## 3. 모델 명세

### 3.1 아키텍처

```
입력 시퀀스 (B, T_max, 1, 64, 64) + lengths (B,)
         │
         ▼      [PatchEncoder] per layer
Conv2d(1 → 32, 3×3) → BN → ReLU
Conv2d(32 → 32, 3×3, stride=2) → BN → ReLU   # 32×32
Conv2d(32 → 64, 3×3, stride=2) → BN → ReLU   # 16×16
Conv2d(64 → 64, 3×3, stride=2) → BN → ReLU   # 8×8
AdaptiveAvgPool2d(1) → Flatten → Linear(64 → 64)   ∈ ℝ⁶⁴
         │
         ▼      [LSTM]  mode 에 따라:
pack_padded_sequence
LSTM(input=64, hidden=8, bidirectional=mode_bidir, num_layers=1)
pooling="last": h_n
   - fwd1/fwd16:    forward 만 → ∈ ℝ⁸
   - bidir1/bidir16: forward + backward concat → ∈ ℝ¹⁶
         │
         ▼      [Projection — 추가 피처]
LayerNorm(hidden_out) → Linear(hidden_out → d_embed)     ∈ ℝ^{d_embed}
   - *1   : d_embed = 1   → 22 번째 피처 (스칼라)
   - *16  : d_embed = 16  → 22..37 번째 피처 (16-dim)
         │
         ▼      [학습용 헤드 — 추출 시 미사용]
Linear(d_embed → 4)                                      ∈ ℝ⁴ (YS, UTS, UE, TE)
```

### 3.2 하이퍼파라미터 (config.py)

| 변수 | 값 |
|:----|:--|
| LSTM_RAW_CHANNEL    | 0 (camera_data/visible/0) |
| LSTM_CROP_SIZE      | 64 |
| LSTM_D_CNN          | 64 |
| LSTM_D_HIDDEN       | 8 (모든 모드 공통) |
| LSTM_MODES          | `{fwd1, bidir1, fwd16, bidir16}` — mode 별 (bidirectional, d_embed) 사양 |
| LSTM_NUM_LAYERS     | 1 |
| LSTM_POOLING        | "last" |
| **mode**            | **fwd1 / bidir1 / fwd16 / bidir16** — CLI `--mode` |
| LSTM_LR             | 1e-3 |
| LSTM_BATCH_SIZE     | 8 (가변 시퀀스 + 큰 이미지) |
| LSTM_MAX_EPOCHS     | 200 |
| LSTM_EARLY_STOP_PATIENCE | 20 |
| LSTM_GRAD_CLIP      | 1.0 |
| LSTM_WEIGHT_DECAY   | 1e-4 |

### 3.3 학습

- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **손실**: L1 (각 4 물성, [-1, 1] 정규화 공간) — 4 손실의 단순 평균
- **CV**: VPPM 과 동일 5-Fold (sample 단위, seed=42 분할 — `Sources/vppm/origin/run_pipeline.py:make_folds()`)
- **early stop**: 검증 평균 RMSE 기준 patience=20
- **데이터 로딩**: `torch.utils.data.DataLoader`, `num_workers=2`, `pin_memory=True`
- **시퀀스 패킹**: `nn.utils.rnn.pack_padded_sequence` (mask = mask 길이 기반)

---

## 4. 임베딩 추출 + 피처 통합

### 4.1 추출 단계 (mode 별)

```python
for fold in 0..4:
    model = SampleLSTMRegressor(bidirectional=mode_bidir, d_embed=mode_d_embed).load(ckpt[mode][fold])
    model.eval()
    for sample_id, seq in all_samples:
        emb = model.encode(seq, lengths)       # (d_embed,)
        embeddings[sample_id, fold] = emb
```

`fold` 차원을 보존하는 이유: 각 fold 의 LSTM 이 약간 다르게 학습되므로, **VPPM 학습 시 같은 fold 의
임베딩을 사용** 해야 데이터 누출 방지 (validation 샘플의 임베딩이 test 시점에 다른 fold 의 LSTM
으로 추출되면 안 됨).

산출물: `Sources/pipeline_outputs/lstm_embeddings/{mode}/embeddings.npz`
```
sample_ids   : (n_samples,) uint32
build_ids    : (n_samples,) string
embeddings   : (n_samples, n_folds=5, d_embed)  float32   # d_embed=1 또는 16
```

### 4.2 통합 (`all_features_with_lstm_{mode}.npz` 생성)

```python
# 입력: features (N_voxels, 21), sample_ids (N_voxels,), fold_assignment (N_voxels,)
# 출력: features_v2 (N_voxels, 21 + d_embed)   # 22 (d=1) 또는 37 (d=16)

for v in range(N_voxels):
    s = sample_ids[v]
    f = fold_assignment[v]   # voxel 이 어느 fold 의 valid set 에 속하는지
    features_v2[v, :21] = features[v]
    features_v2[v, 21:] = embeddings[s, f]    # broadcast (1 또는 16 차원)
```

산출물: `Sources/pipeline_outputs/features/all_features_with_lstm_{mode}.npz` — features dim = 22 or 37.

### 4.3 VPPM 재학습 (mode 별)

- 기존 [Sources/vppm/common/dataset.py](../common/dataset.py) + [origin/train.py](../origin/train.py)
  로직을 (21 + d_embed) 차원 입력으로 호출 (`run_vppm.py` 가 래핑)
- 출력: `Sources/pipeline_outputs/results/vppm_lstm_{mode}/{metrics_raw.json, predictions_*.csv, ...}`
- 모델: `Sources/pipeline_outputs/models_lstm/{mode}/vppm_{prop}_fold{0-4}.pt`

---

## 5. 디렉터리 구조

```
Sources/vppm/lstm/
├── PLAN_LSTM_v2.md
├── __init__.py
├── sample_dataset.py       # 샘플별 시퀀스 Dataset / collate (가변 길이)
├── sample_stack.py         # Phase L1 — HDF5 → 캐시
├── encoder.py              # CNN 패치 인코더
├── lstm_model.py           # LSTM + 1-dim projection + 4-output head
├── train.py                # Phase L2 — 5-Fold CV 학습 (--mode {fwd,bidir})
├── extract.py              # Phase L3 — 임베딩 추출 + 22차원 npz 통합 (--mode)
└── run_vppm.py             # Phase L4 — 22차원 VPPM 재학습 (--mode)

Sources/pipeline_outputs/
├── sample_stacks/          # Phase L1 캐시 (mode 무관, ~5 GB)
│   ├── B1.1.h5 .. B1.5.h5
│   └── normalization.json
├── models_lstm/
│   ├── fwd1/    bidir1/    fwd16/    bidir16/   # 모드별
│   │   ├── lstm_sample_fold{0-4}.pt
│   │   └── vppm_{prop}_fold{0-4}.pt
├── lstm_embeddings/
│   ├── fwd1/embeddings.npz   bidir1/embeddings.npz
│   ├── fwd16/embeddings.npz  bidir16/embeddings.npz
├── features/
│   ├── all_features.npz                       # 21 차원 (baseline)
│   ├── all_features_with_lstm_fwd1.npz        # 22 차원
│   ├── all_features_with_lstm_bidir1.npz      # 22 차원
│   ├── all_features_with_lstm_fwd16.npz       # 37 차원
│   └── all_features_with_lstm_bidir16.npz     # 37 차원 (이전 v1 설계)
└── results/
    ├── vppm_lstm_fwd1/     vppm_lstm_bidir1/
    ├── vppm_lstm_fwd16/    vppm_lstm_bidir16/

docker/lstm/                # Phase L1~L4 도커
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── run_cache.sh            # Phase L1 (mode 무관)
├── run_train.sh            # Phase L2 — --mode {fwd1|bidir1|fwd16|bidir16}
├── run_extract.sh          # Phase L3 — --mode
├── run_vppm.sh             # Phase L4 — --mode
└── run_all.sh              # --mode {fwd1|bidir1|fwd16|bidir16|all}
```

---

## 6. 실행 절차

### 6.1 호스트에서 (개발)

```bash
# Phase L1 — 캐시 빌드 (mode 무관, ~30분)
./venv/bin/python -m Sources.vppm.lstm.sample_stack

# Phase L2~L4 — mode 별 (예: bidir16 = 이전 37 차원 설계)
for M in fwd1 bidir1 fwd16 bidir16; do
  ./venv/bin/python -m Sources.vppm.lstm.train     --mode $M
  ./venv/bin/python -m Sources.vppm.lstm.extract   --mode $M
  ./venv/bin/python -m Sources.vppm.lstm.run_vppm  --mode $M
done
```

### 6.2 도커 (권장)

```bash
cd docker/lstm
./run_all.sh --mode all              # L1 1회 + (L2~L4) × 4 모드 (~12시간)
./run_all.sh --mode bidir16          # 이전 37 차원 설계만
./run_all.sh --mode fwd1             # 새로운 22 차원 forward 만
./run_all.sh --mode all --quick      # smoke test (4 모드 모두)
```

단계별:
```bash
./run_cache.sh                                # L1 (1회만)
for M in fwd1 bidir1 fwd16 bidir16; do
  ./run_train.sh   --mode $M
  ./run_extract.sh --mode $M
  ./run_vppm.sh    --mode $M
done
```

---

## 7. 성공 기준

- [ ] **Phase L1**: 모든 빌드의 cache 파일 생성. samples 합산 = 6,373.
- [ ] **Phase L2**: 5 fold 모두 수렴 또는 early-stop. validation L1 < initial 의 50%.
- [ ] **Phase L3**: `all_features_with_lstm.npz` 의 features.shape == (36047, 37).
- [ ] **Phase L4**: `vppm_lstm/metrics_raw.json` 4 속성 평균 RMSE 모두 baseline (24.28/42.88/9.34/11.27)
      대비 **개선** 또는 동등.
- [ ] **개선 여부 보고**: 21-feat baseline vs 37-feat (LSTM 추가) RMSE 비교 표.

---

## 8. 리스크 & 한계

| 리스크 | 대응 |
|:----|:---|
| **샘플의 layer 수 매우 다양** (10~수천?) — 매우 긴 시퀀스가 일부 존재 가능 | 시퀀스 max 길이 cap (예: 200) — PLAN §2.2 실측 후 결정 |
| **이미지 정규화** — 빌드 간 카메라 노출 차이 | per-build min/max 정규화 (§2.4). 다른 옵션: zscore-per-layer |
| **임베딩 데이터 누출** | fold 별로 LSTM 따로 학습 + voxel 의 fold assignment 일치 (§4.1) |
| **샘플 크기 매우 다름** | 64×64 resize — 작은 샘플은 upscale, 큰 샘플은 downscale. anti-aliasing 적용 권장 |
| **GPU 메모리** — batch 8, 200 layer, 64×64 → 8 × 200 × 64×64×4 ≈ 13 MB / batch (충분) | OK |
| **캐시 디스크** — 5 GB | 사전 확보 |

---

## 9. 후속 가능성

본 실험 결과 (RMSE 개선 여부) 에 따라:

- **개선 있음**: LSTM_D_EMBED sweep (8/16/32/64), Crop size sweep (32/64/128), Multi-channel (raw 0+1) 비교
- **개선 미미**: 임베딩이 21 피처와 redundant — 다른 입력 (예: post-recoat camera, scan path image) 시도
- **악화**: regularization 부족 또는 이미지 노이즈 우세 — augmentation, 더 작은 d_embed 시도
