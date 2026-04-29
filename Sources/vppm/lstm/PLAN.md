# VPPM-LSTM 구현 계획

> **목표**: Baseline VPPM(21 핸드크래프트 피처 → MLP)에 **슈퍼복셀별 CNN+LSTM 임베딩 1개**를 추가해 22(=21+1) 피처 입력으로 학습/평가한다.
> Baseline 의 데이터 파이프라인·학습 루프·평가 메트릭을 그대로 재사용하고, **신규 코드는 LSTM 임베딩 추출 단계에만 한정**한다.

---

## 1. 핵심 설계 결정 (먼저 확정 필요)

| 항목 | 기본값(제안) | 메모 |
|---|---|---|
| **임베딩 단위** | **per-supervoxel** | 사용자 요구. baseline 21 피처와 1:1 대응 |
| **카메라 채널** | `slices/camera_data/visible/0` (용융 직후) | DSCNN 결과맵은 21 피처에 이미 포함 → 원시 시각 채널만 사용 |
| **시퀀스 길이 T** | **가변, 1 ≤ T ≤ 70** | 해당 SV 의 xy 영역에 **CAD 가 실제로 존재하는 레이어만** 시퀀스에 포함. PyTorch `pack_padded_sequence` 로 처리 |
| **이미지 크롭 H×W** | **8×8 (raw)** → CNN 내부에서 16×16 으로 zero-pad 또는 그대로 처리 | `SV_XY_PIXELS≈8`. resize 보다 raw + 작은 CNN 권장 |
| **CNN 출력 차원 d_cnn** | 32 | conv 2~3 stage. 작게 시작 |
| **LSTM hidden** | 16 | 작은 모델로 시작 — 과적합 방지 |
| **LSTM 방향** | **forward 1-layer** | 빌드 진행 방향 따라 시간 축. 필요 시 bidirectional 옵션 |
| **임베딩 차원 d_embed** | **1** | 사용자 요구 "22개 features" → 마지막 hidden 을 `Linear(16→1)` 로 사영 |
| **Pooling** | last hidden | (옵션: mean) |
| **결합 방식** | **concat** (`feat21 ⊕ embed1 → feat22`) | "더한다" 는 결합 의미로 해석 |
| **학습 방식** | **End-to-end joint** (LSTM + MLP 한 번에) | 또는 2-stage(임베딩 frozen 후 MLP) — §6 참조 |

> 위 값들은 모두 `common/config.py` 에 새 상수 그룹으로 추가하고 CLI flag 로 override 가능하게 설계.
> 사용자가 다른 값을 원하면 §11 Open Questions 에서 합의 후 수정.

---

## 2. 전체 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│ HDF5 (5 빌드)                                                         │
│  ├─ slices/part_ids                  (CAD 마스크)                      │
│  ├─ slices/sample_ids                (시편 위치)                       │
│  └─ slices/camera_data/visible/0     (용융 직후 카메라, T_layers×H×W)  │
└──────────────────────────────────────────────────────────────────────┘
        │
        ├─[Baseline 그대로]──> 21 피처 추출 (origin/features.py)
        │                       └─ Sources/pipeline_outputs/features/all_features.npz
        │
        └─[신규 lstm/]
             ├─ Phase L1  crop_stacks.py
             │   └─ 유효 SV 별 (T_sv ≤ 70, H=8, W=8) 가변 길이 크롭 시퀀스를 HDF5 캐시로 저장
             │      (저장은 (N, 70, 8, 8) zero-padded + lengths(N,) 형태 — 단순/효율적)
             │      → Sources/pipeline_outputs/experiments/vppm_lstm/cache/
             │           crop_stacks_B1.{1..5}.h5  +  index.npz
             │
             └─ Phase L2  train.py
                 (a) DataLoader: (21 feat, crop_stack T_max×H×W, length, target) batch
                 (b) Model: VPPM_LSTM
                       crop_stack ──[CNN encoder per-frame]──> (T_max, d_cnn)
                                                              │
                                       pack_padded_sequence(lengths)
                                                              ▼
                                                      [LSTM(forward,1L)]──> h_last (d_lstm)
                                                              │  (각 시퀀스의 실제 마지막 step)
                                                              ▼
                                                      [Linear(→1)]──> embed (1)
                       feat21 ⊕ embed ──> (22)
                       └──[MLP: FC(22→128)→ReLU→Drop(0.1)→FC(128→1)]──> ŷ
                 (c) Loss: L1, Optim: Adam, EarlyStop: patience 50  (baseline 동일)
                 → Phase L3 evaluate.py: per-sample min 집계 / RMSE / plots
```

핵심: **21 피처 추출은 baseline 산출물 (`all_features.npz`) 재사용** — 다시 돌리지 않는다. LSTM 분기는 **추가 입력(이미지 시퀀스 캐시)** 만 별도 생성.

---

## 3. 디렉터리 / 파일 구조

```
Sources/vppm/lstm/
├── PLAN.md                  ← 본 문서
├── MODEL.md                 (구현 후 작성 — 모델/하이퍼파라미터 문서)
├── __init__.py
├── crop_stacks.py           Phase L1: HDF5 → SV별 크롭 시퀀스 캐시 빌드
├── dataset.py               VPPMLSTMDataset — 21feat + crop_stack 동시 제공
├── model.py                 CNN encoder + LSTM + 결합 MLP (VPPM_LSTM)
├── train.py                 학습 루프 (baseline train 과 동일 골격)
├── evaluate.py              평가 (baseline evaluate 와 동일 골격)
└── run.py                   CLI 진입점 — phase=cache|train|evaluate|all

docker/lstm/
├── Dockerfile               LSTM 전용 (entrypoint 가 ORNL/features/output 검증)
├── docker-compose.yml       cache → train → evaluate 순차 실행 (`docker compose up -d --build`)
├── entrypoint.sh            컨테이너 시작 시 검증 + 산출물 서브디렉터리 mkdir
└── .env                     UID_GID, NVIDIA_VISIBLE_DEVICES, LSTM_PHASE, LSTM_EXTRA

Sources/pipeline_outputs/experiments/vppm_lstm/
├── cache/                   Phase L1 산출물
│   ├── crop_stacks_B1.{1..5}.h5         (gzip float16, T×H×W per SV)
│   └── index.npz                         (sv_id → (build, voxel_idx) 매핑)
├── models/                  vppm_lstm_{YS,UTS,UE,TE}_fold{0..4}.pt + training_log.json
├── results/                 metrics_summary.json, predictions_*.csv, plots
├── features/                normalization.json (22-dim 기준)
└── experiment_meta.json     (config 스냅샷)
```

> **주의**: 사용자 지정 출력 경로가 `Sources/pipeline_outputs/experiments./vppm_lstm` 이었음 — `./` 는 오타로 간주하고 `experiments/vppm_lstm/` 으로 표기. 본 PLAN 확정 시 함께 검토.

---

## 4. Phase L1 — 크롭 시퀀스 캐시 (`crop_stacks.py`)

### 입력
- HDF5 `slices/camera_data/visible/0` — `(num_layers, 1842, 1842) uint8`
- baseline 의 `find_valid_supervoxels()` 결과 (voxel_indices, sample_ids, part_ids)

### 출력 — 빌드별 HDF5 (총 5 파일)
```
crop_stacks_B1.x.h5
├── /stacks       (N_sv, 70, 8, 8) float16   — zero-padded 크롭 (uint8 → /255). T_sv 이후 frame 은 0.
├── /lengths      (N_sv,) int16               — 각 SV 의 실제 시퀀스 길이 T_sv (1 ≤ T_sv ≤ 70)
├── /sv_indices   (N_sv, 3) int32             — (ix, iy, iz) — features.npz 와 1:1 매칭용
├── /sample_ids   (N_sv,) int32
└── attrs:
       T_max=70, H=8, W=8, channel="visible/0", build_id, n_sv,
       valid_layer_rule="part_ids>0 in SV xy region"
```

### "유효 레이어" 정의
하나의 SV(`(ix, iy, iz)`) 의 z-범위 `[l0, l1)` 내에서, 다음 조건을 만족하는 레이어 L 만 시퀀스에 포함:

  **`(part_ids[L][r0:r1, c0:c1] > 0).any() == True`**
  (= 그 레이어에서 SV xy 영역에 CAD/파트가 실제로 존재함)

→ 빌드 시작 전 / 파트 상단 위 / 오버행 등 "파트가 아직 없는" 레이어는 자연스럽게 제외됨.
→ T_sv 분포는 빌드별로 다르겠지만 대부분 ~30~70 범위, 일부 엣지/오버행은 더 짧음.

### 알고리즘
```python
for build in BUILDS:
    grid = SuperVoxelGrid.from_hdf5(path)
    valid = find_valid_supervoxels(grid, path)
    N = len(valid["voxel_indices"])
    out = h5py.File(out_path, 'w')
    stacks  = out.create_dataset('stacks', (N, 70, 8, 8), dtype='float16', compression='gzip', compression_opts=4)
    lengths = out.create_dataset('lengths', (N,), dtype='int16')

    with h5py.File(path) as f:
        cam = f['slices/camera_data/visible/0']
        part_ds = f['slices/part_ids']

        # z-block 단위로 묶어서 한 번에 읽기 (I/O 최적화)
        for iz, sv_in_block in groupby_z(valid):
            l0, l1 = grid.get_layer_range(iz)          # 보통 70 레이어
            block_cam  = cam[l0:l1].astype(np.float16) / 255.0    # (T_block, H, W)
            block_part = part_ds[l0:l1]                            # (T_block, H, W)

            for sv in sv_in_block:
                r0, r1, c0, c1 = grid.get_pixel_range(sv.ix, sv.iy)
                # 유효 레이어만 골라서 순서대로 쌓기
                seq = []
                for t in range(l1 - l0):
                    if (block_part[t, r0:r1, c0:c1] > 0).any():
                        seq.append(block_cam[t, r0:r1, c0:c1])
                if not seq:                              # 이론상 valid SV 면 발생 X — 안전장치
                    seq = [block_cam[(l1-l0)//2, r0:r1, c0:c1]]
                arr = np.stack(seq, axis=0)              # (T_sv, H, W)
                arr = pad_to_8x8(arr)                    # 엣지 SV 의 H/W 보정
                lengths[sv.idx] = arr.shape[0]
                stacks[sv.idx, :arr.shape[0]] = arr      # 나머지는 0
```

- **메모리**: z-block 1개 = 70 × 1842 × 1842 × 2B ≈ 470 MB → 한 번에 로딩 가능 (cam + part = ~1 GB)
- **저장 용량 추정**: 36,047 SV × 70 × 8 × 8 × 2B ≈ **323 MB** (gzip 후 ~150 MB). lengths 추가는 무시할 수준.
  - 가변 길이로 ragged 저장하면 ~50% 절약 가능하지만 **padded + lengths 가 dataloader 단순성에서 압승**.
- **시간 예상**: I/O bound ~ 빌드당 5–10 분, 5 빌드 1 시간 이내

### 인덱싱 일관성 보장
- `features/all_features.npz` 의 `voxel_indices` 와 **동일 순서/길이** 로 저장
- merge 시 baseline 과 동일하게 sample_id 오프셋 적용 → 학습 단계에서 N×22 배열로 묶기 쉬움

---

## 5. Phase L2 — 모델 (`model.py`)

### CNN Encoder (per-frame, 8×8 입력)
```
Input  (B*T, 1, 8, 8)
  ↓ Conv2d(1→16, 3, pad=1) + BN + ReLU
  ↓ Conv2d(16→32, 3, pad=1) + BN + ReLU
  ↓ AdaptiveAvgPool2d(1)               # 8×8 → 1×1
  ↓ Flatten + Linear(32 → d_cnn=32)
Output (B*T, 32)
```
파라미터 ≈ 5k. 8×8 입력에 stride/pool 을 과하게 쓰면 정보 소실 → pad 만 사용.

### LSTM (가변 길이)
```
Input  features  (B, T_max=70, 32)   ← CNN 출력
       lengths   (B,)                — 각 SV 의 실제 T_sv
  ↓ pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
  ↓ LSTM(input=32, hidden=16, num_layers=1, batch_first=True, bidirectional=False)
  ↓ → packed output, (h_n, c_n)        ← h_n[-1] 이 곧 "각 시퀀스의 실제 마지막 step hidden"
  ↓ h_last = h_n[-1]                   # (B, 16)
  ↓ Linear(16 → d_embed=1)
Output (B, 1)
```

> **포인트**: `pack_padded_sequence` 를 쓰면 LSTM 이 padding 0 frame 을 무시하고, `h_n[-1]` 이 자동으로 각 샘플의 **유효 마지막 step** 을 가리킨다. 직접 `out[range(B), lengths-1]` 인덱싱할 필요 X.
> bidirectional 옵션 시 `h_n` 마지막 두 layer (forward+backward) 를 concat → `(B, 32)`. proj `Linear(32→1)` 로 일관된 1-dim 임베딩 출력.

### 결합 MLP (= 기존 VPPM 와 동일 골격, 입력만 22)
```
Input  (B, 22)
  ↓ Linear(22 → 128) + ReLU + Dropout(0.1)
  ↓ Linear(128 → 1)
Output (B, 1)
```

### 가중치 초기화
- MLP: `N(0, 0.1)` (baseline 과 동일)
- CNN/LSTM: PyTorch 기본 — 너무 작은 std 주면 LSTM 입력이 죽어 학습 안됨

총 파라미터 ≈ 2.9 k (MLP) + 5 k (CNN) + 3.2 k (LSTM) + 17 (proj) ≈ **11 k** — baseline 의 4× 수준.

---

## 6. 학습 전략

### (A) End-to-end joint (기본)
- 모든 파라미터를 한 옵티마이저로 학습. 4 properties × 5 folds = 20 모델.
- LR=`1e-3`, Adam, L1Loss, batch=256(메모리 고려), max_epochs=500, early-stop 50.
- **GPU 필수** (배치당 (256×70×8×8) 텐서 forward).

### (B) 2-stage (대안 — joint 가 잘 안되면)
1. CNN+LSTM+(임시)Linear → property 직접 회귀로 임베딩 사전학습 (1~2 epoch)
2. 학습된 LSTM 으로 N×1 임베딩 추출 → `all_features_lstm.npz` 저장
3. baseline `train_all` 을 22-feat 으로 그대로 호출

(B) 가 디버깅·재실행에 유리하지만 사용자 요구사항(통합 모델)에는 (A) 가 더 적합. **기본은 (A)** — (B) 는 폴백.

### Sample-wise K-Fold (baseline 동일)
- `create_cv_splits(sample_ids)` 그대로 사용. 같은 시편의 SV 가 train/val 에 동시에 존재하지 않게.

### 평가
- baseline evaluate 와 동일: per-sample **min** 집계 → fold 별 RMSE → 평균 ± 표준편차
- Naive baseline 대비 reduction% 계산
- 산출물 위치만 `experiments/vppm_lstm/results/` 로 변경

---

## 7. 데이터로더 (`dataset.py`)

```python
class VPPMLSTMDataset(Dataset):
    def __init__(self, features21, stacks, lengths, targets):
        # features21: (N, 21) float32  — 정규화된 baseline 피처
        # stacks:     (N, 70, 8, 8) float16  — padded crop sequences (in-memory 권장)
        # lengths:    (N,) int — 각 SV 의 실제 시퀀스 길이
        # targets:    (N,) float32  — 정규화된 타겟
        ...

    def __getitem__(self, i):
        return self.features21[i], self.stacks[i], int(self.lengths[i]), self.targets[i]


def collate_fn(batch):
    """가변 길이 처리: padded 그대로 반환, lengths 만 텐서로."""
    feats   = torch.stack([b[0] for b in batch])           # (B, 21)
    stacks  = torch.stack([b[1] for b in batch])           # (B, 70, 8, 8) — 이미 0-padded
    lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)
    targets = torch.stack([b[3] for b in batch])
    return feats, stacks, lengths, targets
```

- **In-memory mode** (기본): 학습 시작 시 5 빌드 H5 → numpy 통째 로딩 (~1 GB). DataLoader worker 무관 빠름.
- **Lazy mode**: `num_workers>0` + 각 worker 별로 H5 핸들 새로 열기 (multi-process safe).
- 시퀀스를 길이순 정렬할 필요 없음 — `pack_padded_sequence(..., enforce_sorted=False)` 가 처리.

---

## 8. CLI / 실행 진입점 (`run.py`)

```bash
# 0) 사전조건: features/all_features.npz 가 있어야 함 (baseline 의 phase=features 산출물)

# 1) 크롭 시퀀스 캐시 빌드 (5 빌드, ~30 분)
python -m Sources.vppm.lstm.run --phase cache

# 2) 학습 (4 properties × 5 fold = 20 모델, GPU 권장)
python -m Sources.vppm.lstm.run --phase train

# 3) 평가
python -m Sources.vppm.lstm.run --phase evaluate

# 한 번에
python -m Sources.vppm.lstm.run --all

# 옵션 — 모델 변형
python -m Sources.vppm.lstm.run --all --d-embed 16 --bidirectional
```

---

## 9. Docker (`docker/lstm/`)

기존 `docker/ablation/Dockerfile` 의 패턴을 그대로 따른다 — 호스트 venv 를 bind mount.

### 실행 (한 줄)

`.env` 파일이 GPU/UID/단계 기본값을 들고 있어 wrapper 스크립트 없이 docker compose 한 줄로 끝남:

```bash
# 백그라운드 실행 (SSH 끊겨도 계속됨)
docker compose -f docker/lstm/docker-compose.yml up -d --build

# 로그 확인
docker compose -f docker/lstm/docker-compose.yml logs -f

# 정리
docker compose -f docker/lstm/docker-compose.yml down
```

GPU 변경은 `docker/lstm/.env` 의 `NVIDIA_VISIBLE_DEVICES` 또는 인라인:
```bash
NVIDIA_VISIBLE_DEVICES=2 LSTM_EXTRA=--quick docker compose -f docker/lstm/docker-compose.yml up -d --build
```

### Compose 핵심
- `restart: "no"` — 학습 끝나면 자동 재시작 안 함
- entrypoint 가 ORNL/features/output 마운트 + 쓰기 권한 검증 후 산출물 서브디렉터리 mkdir
- `LSTM_PHASE=all|cache|train|evaluate` 로 단계 선택, `LSTM_EXTRA` 로 `--quick` 등 추가 인자

---

## 10. 단계별 구현 순서 (TODO)

1. **PLAN.md 검토 / 합의** ← **현재**
2. `common/config.py` 에 LSTM 신규 상수 추가 (CROP_H=8, T=70, D_CNN=32, D_LSTM=16, D_EMBED=1 등)
3. `lstm/crop_stacks.py` — 캐시 빌더 + 1 빌드 smoke test
4. `lstm/dataset.py` — VPPMLSTMDataset (in-memory 우선)
5. `lstm/model.py` — CNN encoder, LSTM, VPPM_LSTM 결합 모델
6. `lstm/train.py` — single fold + train_all (baseline 재사용 가능한 부분 import)
7. `lstm/evaluate.py` — baseline evaluate 구조 그대로 + 22-feat 적응
8. `lstm/run.py` — CLI 진입점
9. `docker/lstm/` — Dockerfile + entrypoint + compose + .env
10. **Smoke test**: `--quick` (epochs=10) 로 1 빌드 학습 검증
11. 전체 학습 + baseline 대비 RMSE 표 비교 → `MODEL.md` 작성

---

## 11. Open Questions (사용자 확인 필요)

1. **임베딩 차원**: 사용자 발언 "22개 features" 기준으로 `d_embed=1` 을 기본값으로 했음. 만약 16-dim 이상을 원하면 `--d-embed 16` 옵션으로 37-feat MLP 가 됨.
2. **카메라 채널**: `visible/0` (용융 직후) 단일 채널 가정. 만약 `visible/1` (분말 도포 후) 도 함께 쓰면 입력 채널이 (T, 2, 8, 8) 이 됨 — 더 풍부하지만 캐시 크기 2배.
3. **결합 방식**: "더한다" 를 **concat** 으로 해석. 만약 정말 elementwise add 의도였다면 21 차원으로 broadcast 하거나 21 차원 embed 가 필요 — 일반적으론 concat 이 맞음.
4. **End-to-end vs 2-stage**: 기본은 end-to-end. GPU 메모리/안정성 이슈 발생 시 2-stage 폴백 가능.
5. **학습 epochs/early-stop**: baseline (max=5000, patience=50) 과 동일. LSTM 은 보통 더 빨리 수렴 — 줄여도 됨.
6. **Bidirectional?**: 빌드 진행은 시간축이라 forward 가 자연스럽지만, 양방향이 임베딩 표현력은 좋음. 옵션 flag 로.
7. **유효 레이어 정의**: 위 §4 에서 **`part_ids > 0` 가 SV xy 안에 한 픽셀이라도 있는 레이어만** 시퀀스에 포함하기로 했음 (가변 길이 1 ≤ T_sv ≤ 70). 더 엄격하게(예: CAD 픽셀 비율 ≥ 10%) 가고 싶다면 threshold 를 옵션으로. 너무 엄격하면 T_sv=0 인 SV 가 생겨 안전장치 필요.

위 항목들에 대한 답변을 받으면 본 PLAN 을 확정하고 구현 단계로 들어간다.

---

## 12. 참고 — 재사용 모듈

| 모듈 | LSTM 단계에서의 역할 |
|---|---|
| `common/config.py` | 경로 / 하이퍼파라미터. LSTM_* 상수 추가 |
| `common/supervoxel.py` | `SuperVoxelGrid`, `find_valid_supervoxels` — 캐시 빌드에 필수 |
| `common/dataset.py` | `build_dataset`, `create_cv_splits`, `normalize/denormalize` — 22-feat 입력에도 그대로 동작 |
| `baseline/train.py::EarlyStopper` | 그대로 import |
| `baseline/evaluate.py::evaluate_fold/save_metrics/plot_*` | 호출 시그니처가 model 과 features 만 받으므로 재사용 가능 (모델만 VPPM_LSTM 으로 교체) |
| `baseline/features.py` | 변경 없음. `all_features.npz` 그대로 입력 |
