# VPPM-LSTM-Dual 구현 계획

> **목표**: 기존 `vppm_lstm` (21 baseline + 1 LSTM 임베딩 = **22**) 을 확장해
> **`visible/0` (용융 직후) + `visible/1` (분말 도포 후)** 두 카메라 채널 각각에서
> CNN+LSTM 임베딩을 1개씩 뽑아 baseline 21 피처에 concat → **23-dim** 입력으로 학습/평가한다.
>
> "똑같이 CNN+LSTM 을 활용" — 기존 `lstm/` 의 아키텍처를 채널마다 한 번씩 **2 회 미러링**.
> 21 + 1 (visible/0) + 1 (visible/1) = **23 features**.

---

## 1. 핵심 설계 결정

### 1.1 채널 처리 — 독립 브랜치 (mirror)

| 옵션 | 설명 | 채택 |
|---|---|---|
| **(A) 독립 브랜치** | visible/0 / visible/1 각각 별도 CNN+LSTM → 각 1-dim 임베딩 | ✅ 사용자 요구 "똑같이 CNN+LSTM" 에 정확히 부합. 23 = 21 + 1 + 1 |
| (B) 채널 stacking | 입력 (T, 2, 8, 8) 단일 CNN+LSTM | ❌ 출력이 1-dim → 22 features 밖에 안 됨 |
| (C) Shared CNN, separate LSTMs | CNN 가중치 공유, LSTM 만 분리 | △ 두 채널의 통계가 달라 (용융 후 vs 분말 도포 후) shared 가 표현력 손실 가능 — fallback |

→ **기본 (A)**. 두 채널의 의미가 다르므로 (melt pool 직후 vs 분말 도포 직후) **CNN 부터 분리**가 자연스럽다.

### 1.2 핵심 하이퍼파라미터 (기존 lstm 동일, 채널만 2 배)

| 항목 | 값 | 비고 |
|---|---|---|
| 카메라 채널 | `visible/0` + `visible/1` | 두 채널 모두 사용 |
| 시퀀스 길이 T | 가변, 1 ≤ T ≤ 70 | "유효 레이어" 정의 동일 (part_ids > 0) — 두 채널 같은 lengths 공유 |
| 크롭 H×W | 8×8 | `SV_XY_PIXELS ≈ 8`, 기존 동일 |
| CNN 출력 d_cnn | 32 | 채널별 동일 — `FrameCNN` 두 번 instantiate |
| LSTM hidden | 16 | 채널별 동일 |
| LSTM 방향 | forward 1-layer | 빌드 진행 = 시간축 |
| 임베딩 차원 d_embed | **1 per channel** | 23 = 21 + 1 + 1 (사용자 요구) |
| 결합 방식 | concat (`feat21 ⊕ embed_v0 ⊕ embed_v1`) | (B, 23) |
| 학습 방식 | End-to-end joint | 모든 파라미터를 한 옵티마이저로 |

> 위 값들은 `common/config.py` 에 `LSTM_DUAL_*` 새 상수 그룹으로 추가하고 CLI flag override 가능.

### 1.3 기존 visible/0 캐시 재사용

기존 `experiments/vppm_lstm/cache/crop_stacks_B1.x.h5` 의 visible/0 stacks 를 **그대로 재사용** —
visible/1 만 **새로 캐시 빌드**. 빌드 비용 절반.

| 캐시 | 위치 | 채널 |
|---|---|---|
| (재사용) `crop_stacks_B1.x.h5` | `experiments/vppm_lstm/cache/` | visible/0 |
| (신규) `crop_stacks_v1_B1.x.h5` | `experiments/vppm_lstm_dual/cache/` | visible/1 |

> "유효 레이어 = part_ids > 0 in SV xy" 규칙은 카메라 채널과 무관 → 두 캐시의 `lengths` / `sv_indices` / `sample_ids` 가 **bit-identical** 해야 함. 캐시 빌드 시 이 무결성을 검증한다.

---

## 2. 전체 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│ HDF5 (5 빌드)                                                         │
│  ├─ slices/part_ids                  (CAD 마스크 — 유효 레이어 판정)    │
│  ├─ slices/sample_ids                (시편 위치)                       │
│  ├─ slices/camera_data/visible/0     (용융 직후, T_layers×H×W)         │
│  └─ slices/camera_data/visible/1     (분말 도포 후, T_layers×H×W)      │
└──────────────────────────────────────────────────────────────────────┘
        │
        ├─[Baseline 그대로]──> 21 피처 (재사용)
        │     └─ Sources/pipeline_outputs/features/all_features.npz
        │
        ├─[기존 lstm/]────────> visible/0 캐시 (재사용)
        │     └─ experiments/vppm_lstm/cache/crop_stacks_B1.{1..5}.h5
        │
        └─[신규 lstm_dual/]
             ├─ Phase D1  crop_stacks_v1.py
             │   └─ visible/1 채널만 별도 캐시 빌드 (lengths 검증)
             │      → experiments/vppm_lstm_dual/cache/
             │           crop_stacks_v1_B1.{1..5}.h5
             │
             └─ Phase D2  train.py (VPPM_LSTM_Dual)
                 (a) DataLoader: (feat21, stack_v0, stack_v1, length, target) batch
                 (b) Model:
                       stack_v0 ──[CNN_v0]──> (T, d_cnn) ──[LSTM_v0]──> h_v0 ──[Linear→1]──> embed_v0
                       stack_v1 ──[CNN_v1]──> (T, d_cnn) ──[LSTM_v1]──> h_v1 ──[Linear→1]──> embed_v1
                       feat21 ⊕ embed_v0 ⊕ embed_v1 ──> (23)
                       └──[MLP: FC(23→128)→ReLU→Drop(0.1)→FC(128→1)]──> ŷ
                 (c) Loss: L1, Optim: Adam, EarlyStop: patience 50 (baseline 동일)
                 → Phase D3 evaluate.py: per-sample min 집계 / RMSE / plots
```

---

## 3. 디렉터리 / 파일 구조

```
Sources/vppm/lstm_dual/
├── PLAN.md                  ← 본 문서
├── MODEL.md                 (구현 후 작성)
├── FLOW.md                  (구현 후 작성)
├── __init__.py
├── crop_stacks_v1.py        Phase D1: visible/1 캐시 빌드 (visible/0 캐시는 재사용)
├── dataset.py               VPPMLSTMDualDataset — 21feat + stack_v0 + stack_v1 + length
├── model.py                 VPPM_LSTM_Dual — CNN×2 + LSTM×2 + 결합 MLP
├── train.py                 학습 루프 (lstm/train.py 와 동일 골격, dual 입력)
├── evaluate.py              평가 (lstm/evaluate.py 와 동일 골격)
└── run.py                   CLI 진입점 — phase=cache_v1|train|evaluate|all

docker/lstm_dual/
├── Dockerfile               (docker/lstm/Dockerfile 패턴 그대로)
├── docker-compose.yml       cache_v1 → train → evaluate 순차 실행
├── entrypoint.sh
└── .env                     UID_GID, NVIDIA_VISIBLE_DEVICES, PHASE, EXTRA

Sources/pipeline_outputs/experiments/vppm_lstm_dual/
├── cache/                   Phase D1 산출물 (visible/1 만)
│   └── crop_stacks_v1_B1.{1..5}.h5
├── models/                  vppm_lstm_dual_{YS,UTS,UE,TE}_fold{0..4}.pt + training_log.json
├── results/                 metrics_summary.json, predictions_*.csv, correlation_plots.png
├── features/                normalization.json (23-dim 기준이지만 embed 는 학습 산출)
└── experiment_meta.json     (config 스냅샷)
```

---

## 4. Phase D1 — visible/1 캐시 빌드 (`crop_stacks_v1.py`)

기존 `lstm/crop_stacks.py` 와 99% 동일 — 카메라 키만 `visible/1` 로 변경.

### 입력
- HDF5 `slices/camera_data/visible/1` — `(num_layers, 1842, 1842) uint8`
- 기존 `lstm/crop_stacks.py` 의 `find_valid_supervoxels()` 결과와 동일 정의 (channel-agnostic)

### 출력 — 빌드별 H5 (총 5 파일)
```
crop_stacks_v1_B1.x.h5
├── /stacks       (N_sv, 70, 8, 8) float16   — visible/1 zero-padded 크롭
├── /lengths      (N_sv,) int16              — visible/0 캐시와 동일해야 함 (검증)
├── /sv_indices   (N_sv, 3) int32
├── /sample_ids   (N_sv,) int32
└── attrs:
       T_max=70, H=8, W=8, channel="visible/1", build_id, n_sv,
       valid_layer_rule="part_ids>0 in SV xy region"
```

### 무결성 검증 (캐시 빌드 후)
```python
v0_h5 = h5py.File("experiments/vppm_lstm/cache/crop_stacks_B1.x.h5")
v1_h5 = h5py.File("experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.x.h5")
assert (v0_h5["lengths"][...] == v1_h5["lengths"][...]).all()
assert (v0_h5["sv_indices"][...] == v1_h5["sv_indices"][...]).all()
assert (v0_h5["sample_ids"][...] == v1_h5["sample_ids"][...]).all()
```

> 두 캐시의 SV 순서·길이가 비트 단위로 일치해야 학습 시 인덱싱 충돌 없음.
> 만약 visible/0 캐시가 stale 이라면 `lstm/crop_stacks.py` 부터 다시 돌릴 것.

### 알고리즘 (기존과 동일)
`lstm/crop_stacks.py::_build_one_build` 의 `cam_key = "slices/camera_data/visible/0"` →
`"slices/camera_data/visible/1"` 한 줄만 바꾼 미러. 코드 중복을 줄이기 위해 **공통 함수로 리팩터링** 옵션:

```python
# lstm/crop_stacks.py 의 _build_one_build 를 channel 인자 받게 수정 →
# lstm_dual/crop_stacks_v1.py 는 그 함수를 channel=1 로 호출
def build_one_build(build_id, out_dir, channel: int, file_prefix: str): ...
```

→ 구현 시 기존 `lstm/crop_stacks.py` 를 살짝 일반화하고 둘 다 그것을 import 하는 패턴이 깔끔.

### 비용 추정
- 파일 크기: 36,047 SV × 70 × 8 × 8 × 2B ≈ 323 MB → gzip 후 ~150 MB (×5 빌드 ≈ 750 MB unzipped → ~150 MB zipped per build, 750 MB total uncompressed)
- 시간: visible/0 캐시 빌드와 동일 (~빌드당 5–10 분, 5 빌드 1 시간 이내)

---

## 5. Phase D2 — 모델 (`model.py::VPPM_LSTM_Dual`)

### 5.1 두 개의 독립 CNN+LSTM 브랜치

```python
class VPPM_LSTM_Dual(nn.Module):
    def __init__(self, ...):
        # 채널 0 — 용융 직후
        self.cnn_v0  = FrameCNN(d_cnn=32)         # 기존 lstm/model.py 에서 import 가능
        self.lstm_v0 = nn.LSTM(32, 16, num_layers=1, batch_first=True, bidirectional=False)
        self.proj_v0 = nn.Linear(16, 1)

        # 채널 1 — 분말 도포 후
        self.cnn_v1  = FrameCNN(d_cnn=32)
        self.lstm_v1 = nn.LSTM(32, 16, num_layers=1, batch_first=True, bidirectional=False)
        self.proj_v1 = nn.Linear(16, 1)

        # 결합 MLP — baseline VPPM 동일 골격, 입력만 23
        self.fc1     = nn.Linear(21 + 1 + 1, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc2     = nn.Linear(128, 1)

    def encode_one(self, cnn, lstm, proj, stacks, lengths):
        B, T, H, W = stacks.shape
        x = stacks.view(B*T, 1, H, W)
        x = cnn(x).view(B, T, -1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = lstm(packed)
        return proj(h_n[-1])      # (B, 1)

    def forward(self, feats21, stacks_v0, stacks_v1, lengths):
        embed_v0 = self.encode_one(self.cnn_v0, self.lstm_v0, self.proj_v0, stacks_v0, lengths)
        embed_v1 = self.encode_one(self.cnn_v1, self.lstm_v1, self.proj_v1, stacks_v1, lengths)
        x = torch.cat([feats21, embed_v0, embed_v1], dim=1)   # (B, 23)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```

> **주의**: `lengths` 는 두 채널이 동일해야 함 (D1 의 무결성 검증). 하나의 `lengths` 텐서를 두 LSTM 에 공유.

### 5.2 파라미터 수 추정

| 모듈 | 파라미터 |
|---|---:|
| `FrameCNN` × 2 | ~5,900 × 2 ≈ **11,800** |
| `LSTM(32→16)` × 2 | ~3,200 × 2 ≈ **6,400** |
| `Linear(16→1)` × 2 | 17 × 2 ≈ **34** |
| MLP `FC(23→128) + FC(128→1)` | 23·128+128 + 128+1 ≈ **3,201** |
| **합계** | **~21 k** |

기존 `lstm` (~12 k) 의 약 1.8 배. baseline (~3 k) 의 약 7 배. 여전히 작은 모델.

### 5.3 가중치 초기화
- MLP `fc1`, `fc2` 와 두 `proj_v{0,1}`: `N(0, σ=0.1)` (baseline / lstm 과 동일)
- CNN/LSTM: PyTorch 기본 (Xavier-uniform 류)

---

## 6. Phase D2 — 학습 (`train.py`)

기존 `lstm/train.py` 와 99% 동일. 차이점:
1. 데이터셋이 `(feat21, stack_v0, stack_v1, length, target)` 5-tuple 반환
2. 모델 forward 가 4 입력 받음
3. 모델 저장 prefix `vppm_lstm_dual_*`

```python
for feats, sv0, sv1, lengths, ys in train_loader:
    feats, sv0, sv1, ys = feats.to(dev), sv0.to(dev), sv1.to(dev), ys.to(dev)
    pred = model(feats, sv0, sv1, lengths)   # lengths 는 cpu
    loss = L1Loss()(pred, ys)
    loss.backward()
    clip_grad_norm_(parameters, 1.0)
    optimizer.step()
```

### 학습 설정 (기존 lstm 동일)
- `LR=1e-3`, `Adam(β=0.9/0.999, eps=1e-4)`, `L1Loss`, `BatchSize=256`
- `MaxEpochs=5000`, `EarlyStopPatience=50`, `GradClip=1.0`, `WeightDecay=0`
- sample-wise 5-fold (baseline / lstm 과 동일 splits)

### 메모리
- 배치당 텐서: `(256, 70, 8, 8)` × **2 채널** = 약 1.7 MB float16 → 3.4 MB float32 = 충분히 작음
- GPU 메모리는 LSTM 내부 activations 가 dominate — 기존 lstm 의 약 2 배 (브랜치 2 개)

---

## 7. 데이터로더 (`dataset.py`)

```python
class VPPMLSTMDualDataset(Dataset):
    def __init__(self, features21, stacks_v0, stacks_v1, lengths, targets):
        # features21: (N, 21) float32 normalized
        # stacks_v{0,1}: (N, 70, 8, 8) float16 padded
        # lengths: (N,) int64
        # targets: (N,) float32 normalized
        ...

    def __getitem__(self, i):
        return (self.features21[i], self.stacks_v0[i], self.stacks_v1[i],
                int(self.lengths[i]), self.targets[i])


def collate_fn(batch):
    feats   = torch.stack([b[0] for b in batch])              # (B, 21)
    sv0     = torch.stack([b[1] for b in batch]).float()      # (B, 70, 8, 8) — float16→float32
    sv1     = torch.stack([b[2] for b in batch]).float()
    lengths = torch.tensor([b[3] for b in batch], dtype=torch.long)   # cpu
    targets = torch.stack([b[4] for b in batch])
    return feats, sv0, sv1, lengths, targets
```

### `load_dual_dataset()` (lstm/dataset.py::load_lstm_dataset 의 dual 버전)

```python
def load_dual_dataset(features_npz, cache_v0_dir, cache_v1_dir, build_ids):
    # 1) features.npz 로드 (baseline)
    # 2) 빌드별 v0 캐시 concat → cache_stacks_v0, cache_lengths_v0
    # 3) 빌드별 v1 캐시 concat → cache_stacks_v1, cache_lengths_v1
    # 4) 무결성 검증:
    #    - feature N == cache N
    #    - lengths_v0 == lengths_v1  (assert all)
    #    - sv_indices_v0 == sv_indices_v1
    # 5) baseline build_ids 정렬 순서로 매칭
    # 6) 반환: features, sample_ids, build_ids, targets, stacks_v0, stacks_v1, lengths
```

`build_normalized_dataset()` 도 같은 valid_mask 로 v0/v1 동시 슬라이싱.

---

## 8. CLI / 실행 진입점 (`run.py`)

```bash
# 0) 사전조건:
#    - features/all_features.npz  (baseline)
#    - experiments/vppm_lstm/cache/crop_stacks_B1.x.h5  (visible/0 — 기존 lstm 캐시)

# 1) visible/1 캐시 빌드 (~30 분)
python -m Sources.vppm.lstm_dual.run --phase cache_v1

# 2) 학습 (4 properties × 5 folds = 20 모델, GPU 권장)
python -m Sources.vppm.lstm_dual.run --phase train

# 3) 평가
python -m Sources.vppm.lstm_dual.run --phase evaluate

# 한 번에
python -m Sources.vppm.lstm_dual.run --all

# Smoke test
python -m Sources.vppm.lstm_dual.run --all --quick
```

> `cache_v0` (visible/0) 는 별도 phase 로 두지 않음 — 기존 `lstm/run.py --phase cache` 의 산출물을 그대로 의존. visible/0 캐시가 없으면 친절한 에러 메시지로 안내.

---

## 9. Phase D3 — 평가 (`evaluate.py`)

기존 `lstm/evaluate.py` 와 동일. 모델 forward 시그니처만 4 입력으로.

```python
def _evaluate_fold(model, feats, stacks_v0, stacks_v1, lengths,
                   targets_raw, sample_ids, norm_params, prop, device, batch_size=1024):
    model.eval()
    preds_norm = np.empty(N, dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            i1 = min(i0 + batch_size, N)
            f  = torch.from_numpy(feats[i0:i1]).float().to(device)
            s0 = torch.from_numpy(stacks_v0[i0:i1]).float().to(device)
            s1 = torch.from_numpy(stacks_v1[i0:i1]).float().to(device)
            l  = torch.from_numpy(lengths[i0:i1].astype(np.int64))   # cpu
            preds_norm[i0:i1] = model(f, s0, s1, l).cpu().numpy().flatten()
    # ... per-sample min, RMSE, denormalize 등은 baseline / lstm 과 동일
```

baseline 의 `save_metrics`, `plot_correlation`, `plot_scatter_uts` 그대로 import.

---

## 10. Docker (`docker/lstm_dual/`)

기존 `docker/lstm/` 패턴 복제.
- `entrypoint.sh` 가 ORNL / features.npz / **visible/0 캐시** / output 검증
- `LSTM_DUAL_PHASE=all|cache_v1|train|evaluate`, `LSTM_DUAL_EXTRA=--quick`

```bash
docker compose -f docker/lstm_dual/docker-compose.yml up -d --build
docker compose -f docker/lstm_dual/docker-compose.yml logs -f
```

---

## 11. 단계별 구현 순서 (TODO)

1. **본 PLAN.md 검토 / 합의** ← **현재**
2. `common/config.py` 에 `LSTM_DUAL_*` 신규 상수 그룹 추가
   (`LSTM_DUAL_CACHE_DIR`, `LSTM_DUAL_MODELS_DIR`, `LSTM_DUAL_RESULTS_DIR`, `LSTM_DUAL_FEATURES_DIR`, ...)
3. `lstm/crop_stacks.py` 의 `_build_one_build` 를 `channel` 인자 받도록 일반화 (선택)
4. `lstm_dual/crop_stacks_v1.py` — visible/1 캐시 빌더 + 무결성 검증
5. `lstm_dual/dataset.py` — `VPPMLSTMDualDataset`, `load_dual_dataset`, `build_normalized_dataset`
6. `lstm_dual/model.py` — `FrameCNN` 재사용 import, `VPPM_LSTM_Dual` 신규
7. `lstm_dual/train.py` — `train_single_fold`, `train_all` (lstm/train.py 의 dual 버전)
8. `lstm_dual/evaluate.py` — `_evaluate_fold`, `evaluate_all` (lstm/evaluate.py 의 dual 버전)
9. `lstm_dual/run.py` — CLI 진입점
10. `docker/lstm_dual/` — Dockerfile + entrypoint + compose + .env
11. **Smoke test**: `--quick` 으로 1 빌드 학습 검증
12. 전체 학습 후 결과 비교: baseline (21) vs lstm (22) vs lstm_dual (23) RMSE 표 → `MODEL.md` 작성

---

## 12. Open Questions (사용자 확인 필요)

1. **폴더명**: `lstm_dual` 로 명명 (visible/0+1 dual-channel). `lstm_2ch` / `lstm23` / `lstm_v01` 등 다른 이름 선호하면 변경.
2. **임베딩 차원**: 채널당 1-dim 으로 23-feat 가정. 만약 채널당 더 큰 임베딩 (예: 8-dim 씩) 을 원하면 `--d-embed-v0 8 --d-embed-v1 8` 옵션으로 37-feat 가능.
3. **CNN 가중치 공유 여부**: 기본은 **분리** (옵션 A). 만약 두 채널의 시각 통계가 충분히 비슷하다고 보면 `--share-cnn` 으로 가중치 공유 모드 추가 가능.
4. **두 LSTM 의 hidden 차원**: 동일 (16) 가정. 채널별로 다르게 설정 필요 시 별도 flag.
5. **visible/0 캐시 의존성**: 기존 `experiments/vppm_lstm/cache/` 를 **읽기 전용** 으로 의존. 만약 lstm_dual 을 완전 독립으로 만들고 싶다면 v0 캐시도 lstm_dual/cache/ 로 복제/재빌드. 디스크 비용 (~150 MB × 5) 추가.
6. **공통 코드 리팩터링**: `crop_stacks.py::_build_one_build` 와 `model.py::FrameCNN` 을 lstm/ 와 lstm_dual/ 에서 동시에 import 할지 (DRY) 또는 복제 (독립성 우선) 할지.
7. **End-to-end vs 2-stage**: 기본은 end-to-end. GPU OOM 시 채널별로 임베딩 사전학습 후 23-feat MLP 만 재학습하는 2-stage 폴백 가능.

위 항목 답변 후 본 PLAN 확정 → 구현 단계 진입.

---

## 13. 참고 — 재사용 / 신규 모듈

| 모듈 | 역할 |
|---|---|
| `common/config.py` | 경로 / 하이퍼파라미터. `LSTM_DUAL_*` 추가 |
| `common/supervoxel.py` | `SuperVoxelGrid`, `find_valid_supervoxels` — 캐시 빌드 (재사용) |
| `common/dataset.py` | `create_cv_splits`, `normalize`, `denormalize`, `save_norm_params` (재사용) |
| `lstm/crop_stacks.py` | `_build_one_build` 를 channel 인자로 일반화 후 둘 다 import |
| `lstm/model.py::FrameCNN` | dual 에서도 **그대로 import** (per-frame 8×8 인코더는 동일) |
| `baseline/train.py::EarlyStopper` | (재사용) |
| `baseline/evaluate.py::save_metrics, plot_correlation, plot_scatter_uts` | (재사용) |
| `lstm_dual/model.py::VPPM_LSTM_Dual` | **신규** — CNN×2 + LSTM×2 + MLP 결합 |
| `lstm_dual/dataset.py` | **신규** — dual 채널 동시 로드 + lengths 무결성 검증 |
| `lstm_dual/{train,evaluate,run}.py` | **신규** — lstm/ 의 dual 변형 |

---

## 14. 예상 결과 / 가설

| 모델 | features | RMSE 예상 (UTS) | 비고 |
|---|---:|:---:|---|
| baseline | 21 | 60.7 ± 2.6 MPa | 기준 |
| lstm | 22 | 60 전후 (visible/0 임베딩 1) | 측정 중 |
| **lstm_dual** | **23** | **lstm 보다 1–3 MPa 추가 개선 기대** | 분말 도포 후 결함 (visible/1) 정보가 보완 |

가설: visible/1 (분말 도포 후) 은 리코터 상태·분말 분포 결함을 직접 보여주므로,
B1.5 (리코터 손상 빌드) 와 B1.4 (스패터 빌드) 에서 가장 큰 개선이 있을 것. baseline DSCNN 채널이 이미 일부 정보를 담지만 **공간/시간 패턴** 은 LSTM 임베딩이 보강.
