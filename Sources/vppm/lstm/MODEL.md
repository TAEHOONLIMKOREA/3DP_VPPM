# VPPM-LSTM 모델 설명

논문(Scime et al., *Materials* 2023, 16, 7293) 의 VPPM 을 확장해, 슈퍼복셀의 **레이어별 in-situ 이미지 시퀀스** 를 CNN+LSTM 으로 임베딩하고 기존 21개 핸드크래프트 피처와 결합해 인장 특성(YS/UTS/UE/TE) 을 회귀하는 모델이다.

---

## 1. 전체 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│                       한 슈퍼복셀 (1×1×3.5 mm)                        │
└──────────────────────────────────────────────────────────────────────┘
              │
              ├─── ❶ 21-feat (핸드크래프트)                   x21 ∈ ℝ²¹
              │       ├─ CAD        (3)  : distance_edge / overhang / height
              │       ├─ DSCNN 평균 (8)  : 8 결함 클래스 픽셀 확률 평균
              │       ├─ Temporal   (7)  : 온도/산소/유량 등
              │       └─ Scan       (3)  : laser/return/stripe
              │                                           LayerNorm(21)
              │
              └─── ❷ 이미지 시퀀스 (T=70, C=9, 8×8)
                     │   C0       : raw 카메라 (용융 직후 흑백)
                     │   C1..C8   : DSCNN 8-class 확률맵
                     │
                     ▼                       [PatchEncoder]    per layer
                  Conv(9→32, 3×3) → BN → ReLU
                  Conv(32→32, 3×3) → BN → ReLU
                  Conv(32→64, 3×3) → BN → ReLU
                  AdaptiveAvgPool(1) → Flatten → Linear(64→64)     ∈ ℝ⁶⁴
                     │
                     ▼                       [Bi-LSTM]      T=70 시퀀스
                  pack_padded_sequence(mask 길이 기반)
                  LSTM(input=64, hidden=8, bidir=True, num_layers=1)
                  pooling="last" : h_n 의 정/역방향 concat           ∈ ℝ¹⁶
                                                        LayerNorm(16)
                     ▼
              ┌──────────────────────────────────┐
              │  concat: 21 + 16 = 37-dim        │
              ├──────────────────────────────────┤
              │  Linear(37→128) → ReLU → Drop(0.1)│
              │  Linear(128→1)                    │
              └──────────────────────────────────┘
                     ▼
               예측: YS / UTS / UE / TE (정규화된 값 → 역정규화)
```

출력 단위는 학습 시 `[-1,1]` 로 정규화된 타겟이며, 추론 시 `denormalize()` 로 MPa(%) 로 되돌린다.

---

## 2. 입력 데이터 상세

### 2-1. 21 핸드크래프트 피처 (기존 VPPM 과 동일)

`config.FEATURE_GROUPS` 기준 0-base 인덱스. origin/features.py 의 `FEATURE_NAMES` 와 일치.

| 그룹 | 인덱스 | 설명 |
|---|---|---|
| CAD     | 0, 1, 2       | distance_edge, distance_overhang, build_height |
| DSCNN   | 3 ~ 10        | Powder / Printed / Recoater Streaking / Edge Swelling / Debris / Super-Elevation / Soot / Excessive Melting 8 클래스 **픽셀 평균 확률** |
| Sensor  | 11 ~ 17       | layer_times, top/bottom_flow_rate, module_oxygen, build_plate_temperature, bottom_flow_temperature, actual_ventilator_flow_rate |
| Scan    | 18, 19, 20    | laser_module, return_delay, stripe_boundaries |

정규화: build_dataset() 에서 build-level min/max 기반 `[-1, 1]` 로 선형 변환.

### 2-2. 이미지 시퀀스 `stacks` — shape `(N, 70, 9, 8, 8)`

- `N = 36,047` (전체 유효 슈퍼복셀 수)
- `T = 70` — 슈퍼복셀의 z 방향 70 레이어 (= 3.5 mm / 0.05 mm)
- `C = 9` — **raw(1) + DSCNN(8)** (`config.LSTM_INPUT_CHANNELS = "raw+dscnn"`)
- `H = W = 8` — `config.LSTM_PATCH_PX`. 실제 공간 1×1 mm ≈ 7.52 픽셀 → 8 로 rounding 후 크롭
- dtype: `float16` (원본 `stacks_all.h5`), 학습/평가 시 `float32` 로 변환

**채널 매핑** (`image_stack.resolve_channels("raw+dscnn")`):

| C | HDF5 키 | 내용 |
|---|---|---|
| 0 | `slices/camera_data/visible/0` | 용융 직후 가시광 이미지 (0~255 강도) |
| 1 | `slices/segmentation_results/0` | Powder |
| 2 | `slices/segmentation_results/1` | Printed |
| 3 | `slices/segmentation_results/3` | Recoater Streaking |
| 4 | `slices/segmentation_results/5` | Edge Swelling |
| 5 | `slices/segmentation_results/6` | Debris |
| 6 | `slices/segmentation_results/7` | Super-Elevation |
| 7 | `slices/segmentation_results/8` | Soot |
| 8 | `slices/segmentation_results/10` | Excessive Melting |

### 2-3. 마스크 `masks` — shape `(N, 70)` bool

슈퍼복셀의 실제 레이어가 70 보다 적은 경우(빌드 경계 등) 유효 레이어만 `True`. LSTM `pack_padded_sequence(lengths=mask.sum(dim=1))` 로 전달되어 패딩 영향 제거.

> **주의**: DSCNN 결과는 **핸드크래프트 21 피처의 3~10 번 (스칼라 평균)** 과 **이미지 채널 1~8 (공간 맵 시퀀스)** 양쪽에 동시에 들어간다. 전자는 슈퍼복셀 전체 집계 통계, 후자는 시공간 패턴 — 표현 수준이 다르다.

---

## 3. 컴포넌트 상세

### 3-1. `PatchEncoder` — `lstm/encoder.py`

입력 `(B·T, 9, 8, 8)` → 출력 `(B·T, 64)`. 패치가 작아서 깊이 얕게.

| 레이어 | 파라미터 |
|---|---|
| Conv2d(9→32, k=3, pad=1) + BN + ReLU | ≈ 2.6k + 64 |
| Conv2d(32→32, k=3, pad=1) + BN + ReLU | ≈ 9.2k + 64 |
| Conv2d(32→64, k=3, pad=1) + BN + ReLU | ≈ 18.5k + 128 |
| AdaptiveAvgPool2d(1) → Flatten | — |
| Linear(64→64) | ≈ 4.2k |

Padding=1 이라 8×8 공간 해상도 유지 → 마지막 pool 단계에서 1×1 로 축소.

### 3-2. `SupervoxelLSTM` — `lstm/sequence_model.py`

한 슈퍼복셀의 T=70 레이어를 시간축으로 읽어 임베딩.

1. `img.reshape(B·T, 9, 8, 8)` → `PatchEncoder` → `(B, T, 64)`
2. `mask.sum(dim=1)` → 각 샘플의 유효 길이
3. `pack_padded_sequence(..., batch_first=True, enforce_sorted=False)`
4. `nn.LSTM(64, 8, num_layers=1, bidirectional=True, batch_first=True)`
5. **pooling="last"**: `h_n` 의 정방향(`h_n[-2]`) + 역방향(`h_n[-1]`) concat → `(B, 16)`
6. 대안: `pooling="mean"` — `pad_packed_sequence` 후 마스크 기반 평균

출력 차원 `d_lstm = 16 = 8 × 2`. Bi-LSTM 이므로 `hidden_per_dir = d_lstm // 2`.

### 3-3. `VPPM_LSTM` — `model.py`

결합 헤드:

```python
emb = self.seq(img, mask)                              # (B, 16)
x = torch.cat([LN_hand(x21), LN_lstm(emb)], dim=1)     # (B, 37)
x = F.relu(self.fc1(x))                                # (B, 128)
x = self.dropout(x)                                    # p=0.1
return self.fc2(x)                                     # (B, 1)
```

- **LayerNorm 두 개**: 21 피처(이미 정규화됨) 와 LSTM 임베딩(학습에서 자유롭게 스케일 변동) 의 신호 크기를 맞춤 → 한쪽이 헤드를 지배하는 것을 방지.
- **초기화**: `_init_fc` 에서 fc1/fc2 만 `N(0, σ=0.1)` 로 재설정 (CNN/LSTM 기본 초기화 유지). bias=0.

---

## 4. 학습 설정 — `lstm/train_lstm.py`

| 항목 | 값 | 근거 |
|---|---|---|
| Loss | `L1Loss` (MAE) | 정규화 타겟 [-1,1] 에서 안정 |
| Optimizer | Adam | lr=`1e-3`, betas=(0.9, 0.999), eps=`1e-4` |
| Weight decay | `1e-4` | 이미지 시퀀스 over-fit 방지 |
| Grad clip | 1.0 | LSTM exploding gradient 방어 |
| Batch size | 64 | GPU 메모리 고려 (3-4GB 수준) |
| Max epochs | 200 | 실제 수렴 40~140 |
| Early stop patience | 20 | val loss 개선 없을 때 |
| CV | **sample-wise** 5-fold | `dataset.create_cv_splits(sample_ids)` — 같은 sample 의 여러 슈퍼복셀이 train/val 로 분리되지 않도록 |

학습 대상: 4 properties × 5 folds = **20 모델**. 각각 `models_lstm/vppm_lstm_{YS|UTS|UE|TE}_fold{0..4}.pt` 로 저장. best fold 포인터는 `*_best.json`.

### 학습 루프 핵심

```python
# train_lstm.py:_train_one_fold
pred = model(x21, img, mask)
loss = criterion(pred, y)         # L1Loss
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
optim.step()
# validation 후 best_val 갱신 + early stop
```

---

## 5. 추론 및 평가 — `lstm/eval_lstm.py`

1. 각 fold 의 val split 에 대해 저장된 `.pt` 로드 → forward → 정규화 예측값.
2. `denormalize(pred, target_min, target_max)` 로 MPa/%  복원.
3. **per-sample min 집계** (기존 origin/evaluate.py 와 동일, 논문 Section 3.1 "보수적 추정"): 한 sample 에 속한 여러 슈퍼복셀 예측 중 최솟값을 그 sample 의 예측으로 사용 → 실제 인장 파단은 가장 약한 부분에서 시작한다는 물리적 가정.
4. RMSE(fold) = √mean((p - t)²) → 5-fold 평균/편차.
5. `correlation_plots.png` (2D hist2d), `predictions_*.csv` (per-sample), `cv_metrics.json` 생성.

### 최신 5-fold CV 결과

| Property | VPPM-LSTM RMSE | Naive | Reduction |
|---|---|---|---|
| YS  | **21.1 ± 1.0** MPa | 33.9 | 38% |
| UTS | **30.1 ± 1.2** MPa | 68.4 | 56% |
| UE  | **6.7 ± 0.3** %    | 15.0 | 56% |
| TE  | **8.4 ± 0.4** %    | 18.5 | 55% |

(Naive = 전체 평균 예측 RMSE)

---

## 6. 파일 맵

```
Sources/vppm/
├── run_pipeline.py                # 파이프라인 entry point
├── regen_plots.py                 # origin/lstm PNG 재생성 유틸 (공용)
├── common/                        # 공용 모듈
│   ├── config.py                  # 하이퍼파라미터, 채널 조합, 경로
│   ├── dataset.py                 # build_dataset, create_cv_splits, normalize/denormalize, VPPMDataset
│   ├── model.py                   # VPPM, VPPM_LSTM
│   └── supervoxel.py              # SuperVoxelGrid, find_valid_supervoxels
├── origin/                        # 기존 VPPM 파이프라인
│   ├── features.py                # 21 피처 추출
│   ├── train.py                   # VPPM 학습
│   └── evaluate.py                # VPPM 평가 + 플롯
├── lstm/                          # VPPM-LSTM 확장
│   ├── MODEL.md                   # ← 본 문서
│   ├── image_stack.py             # HDF5 → stacks_all.h5 (Phase L1)
│   ├── encoder.py                 # PatchEncoder CNN (Phase L2)
│   ├── sequence_model.py          # SupervoxelLSTM (Phase L3)
│   ├── dataset.py                 # VppmLstmDataset, collate, load_aligned_arrays (L4)
│   ├── train_lstm.py              # train_vppm_lstm (L5)
│   └── eval_lstm.py               # evaluate_vppm_lstm, export_lstm_embeddings (L6)
└── tools/                         # 데이터 시각화/포맷 변환 유틸 (LSTM 무관)
    ├── split_stacks_by_build.py   # stacks_all.h5 → 빌드별 분리 (H5Web 뷰용)
    ├── view_stacks_example.py     # stacks_all.h5 시각화 예제
    ├── view_per_build.py          # 빌드별 파일 시각화
    └── export_crop_png.py         # 슈퍼복셀 crop PNG 추출
```

### 산출물

```
Sources/pipeline_outputs/
├── features/all_features.npz              # 21 피처 + 타겟 + sample_ids + build_ids
├── image_stacks/
│   ├── stacks_all.h5                      # (36047, 70, 9, 8, 8) float16, 2.8 GB
│   └── per_build/stacks_B1.{1..5}.h5      # 빌드별 분리본 (H5Web 열람용, float32+gzip)
├── models_lstm/vppm_lstm_{YS|UTS|UE|TE}_fold{0..4}.pt  # 20 모델
├── lstm_embeddings/lstm_emb_raw_dscnn_d16.npz          # UTS best fold 로 전체 N×16
└── results/vppm_lstm/
    ├── cv_metrics.json
    ├── training_log.json
    ├── predictions_{YS|UTS|UE|TE}.csv     # per-sample ground truth + prediction
    └── correlation_plots.png
```

---

## 7. 실행 방법

```bash
# 1) 21 피처 추출 (기존 VPPM 파이프라인 — 이미 수행되어 있음)
python -m Sources.vppm.run_pipeline --phase features --builds B1.1 B1.2 B1.3 B1.4 B1.5

# 2) 이미지 스택 캐시 구축 (HDF5 5 개 읽음, ~30분)
python -m Sources.vppm.run_pipeline --use-lstm --phase image-stack

# 3) LSTM 학습 (4 property × 5 fold = 20 모델, GPU 권장)
python -m Sources.vppm.run_pipeline --use-lstm --phase train-lstm

# 4) 평가 + 임베딩 export
python -m Sources.vppm.run_pipeline --use-lstm --phase eval-lstm

# 또는 전체 일괄:
python -m Sources.vppm.run_pipeline --use-lstm
```

환경 변수 `LSTM_CACHE_DIR` 로 스택 캐시 경로 재지정 가능 (기본: `Sources/pipeline_outputs/image_stacks/`).

---

## 8. 주요 설계 선택 이유

1. **왜 raw + DSCNN 9 채널인가?**
   - raw 만 쓰면 "밝기 패턴"만 봄 → 결함 종류 식별이 간접적.
   - DSCNN 만 쓰면 "어떤 결함인지"는 알지만 "강도/지속 시간" 정보 소실.
   - 둘 다 채널로 쌓아 CNN 이 상호작용을 직접 학습하도록 함.

2. **왜 8×8 패치인가?**
   - 슈퍼복셀 실제 크기(1×1 mm) 에 대응하는 최소 픽셀 수. 더 크면 이웃 supervoxel 과 중복.
   - 패치가 작아 CNN 이 깊을 필요가 없음 → 학습 시간·메모리 절감.

3. **왜 Bi-LSTM + pooling="last" 인가?**
   - 70 레이어는 위에서부터 순차 적층 → 과거(아래쪽 레이어) 가 현재의 미세구조를 결정.
   - 양방향은 "다음 레이어의 열이 현재에 주는 영향" 도 포착.
   - "last" pooling 은 마지막 레이어(슈퍼복셀 상단) 기준 요약 → 최종 미세구조와 가장 직결.

4. **왜 LayerNorm 을 두 경로에 각각 적용?**
   - 21 피처는 [-1,1] 로 잘 정규화돼 있으나, LSTM 임베딩은 스케일이 학습 중 임의로 커질 수 있음.
   - 양쪽을 같은 레벨로 맞춰야 FC1 이 한쪽을 무시하지 않음.

5. **왜 L1Loss?**
   - 인장 특성에 소수의 이상치(measurement artifact)가 존재 → MSE 보다 robust.
   - 논문 VPPM 원본도 유사한 경험적 선택.

---

## 9. 알려진 한계

- 패치 8×8 는 물리적으로 작음 → 넓은 영역 결함(대형 keyhole 군집 등)은 이웃 supervoxel 집합으로만 감지 가능.
- Bi-LSTM 이 슈퍼복셀 경계를 넘지 못함 — 수평 방향 공간 상관은 21 피처에만 의존.
- `pooling="last"` 는 초기 레이어(기판 근처) 신호를 덜 반영. 필요 시 `pooling="mean"` 으로 변경.
- `LSTM_D_EMBED=16` 은 21 피처에 비해 작게 잡아 21 피처가 dominate 하도록 했음. Ablation 실험이 필요할 경우 32/64 로 키워 비교 권장.
