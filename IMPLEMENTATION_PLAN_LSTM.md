# VPPM-LSTM 업그레이드 설계 계획서

> **기반 문서**: `IMPLEMENTATION_PLAN.md` (논문 기반 21-feature VPPM 재구현)
> **확장 목표**: 한 슈퍼복셀을 구성하는 70개 레이어 이미지의 **시간적 순서**를 보존한 채 LSTM으로 임베딩하여 22번째 피처 소스로 추가

---

## 1. 업그레이드 동기

기존 VPPM은 DSCNN 세그멘테이션 결과를 슈퍼복셀 단위로 **단순 평균(비율)** 만 취했다. 이는 다음 한계를 갖는다.

| 문제 | 설명 |
|-----|-----|
| 레이어 순서 손실 | 70개 레이어의 결함 패턴이 시간적으로 어떻게 진화했는지(예: 초기 레이어의 스패터 → 후기 레이어의 Excessive Melting)는 평균으로 사라짐 |
| 공간 정보 손실 | 슈퍼복셀 내부의 픽셀 분포(결함 위치·크기·모양)가 스칼라 비율로 붕괴됨 |
| CNN-수준 피처 미사용 | 원본 이미지(`slices/camera_data/visible/*`)의 질감·조도·경계 정보가 입력에 포함되지 않음 |

**가설**: 슈퍼복셀이 걸친 레이어 이미지 시퀀스를 CNN+LSTM으로 인코딩하면 위 세 가지 정보를 복원할 수 있고, 기존 21개 핸드크래프트 피처를 보완한다.

---

## 2. 상위 아키텍처

```
        ┌──────────────────────────────┐
        │  Phase 2 기존 파이프라인      │
        │  (features.py)               │
        │  → 21-dim hand-crafted       │
        └──────────┬───────────────────┘
                   │
                   │            ┌────────────────────────────────┐
                   │            │  Phase LSTM (신규)              │
                   │            │                                │
                   │            │  per-supervoxel image stack    │
                   │            │  (T=70, C, Hs, Ws)             │
                   │            │        │                       │
                   │            │        ▼                       │
                   │            │   CNN Encoder (공유 weights)   │
                   │            │        │                       │
                   │            │        ▼                       │
                   │            │   (T, d_cnn) 시퀀스            │
                   │            │        │                       │
                   │            │        ▼                       │
                   │            │   LSTM  (bidirectional 옵션)   │
                   │            │        │                       │
                   │            │        ▼                       │
                   │            │   Last/Pool → d_lstm 임베딩    │
                   │            └──────────┬─────────────────────┘
                   │                       │
                   ▼                       ▼
        ┌──────────────────────────────────────┐
        │   concat → (21 + d_lstm) 피처 벡터   │
        └───────────────┬──────────────────────┘
                        ▼
                 VPPM-LSTM Head
                 (FC → Dropout → FC → 1)
                        ▼
               YS / UTS / UE / TE 예측
```

- **"기존 21개 피처 + 한 가지 피처 소스 추가"** 라는 요구를 만족: 22번째 피처는 단일 스칼라가 아닌 **d_lstm 차원의 임베딩 벡터** (기본 `d_lstm = 16`, `config.py`에서 조정 가능). 최종 VPPM 입력 차원은 `21 + d_lstm`.
- 임베딩을 스칼라 1개로 강제하고 싶으면 `d_lstm=1` 혹은 LSTM 출력 뒤에 `Linear(d_lstm, 1)` 사영을 추가하는 옵션을 유지.

---

## 3. 디렉토리 구조 (추가/수정 파일만)

```
Sources/vppm/
├── config.py                   # [수정] LSTM 관련 하이퍼파라미터 추가
├── features.py                 # [유지] 기존 21-feature 경로 그대로
│
├── lstm/                       # [신규] LSTM 서브패키지
│   ├── __init__.py
│   ├── image_stack.py          # Phase L1: per-supervoxel 이미지 스택 추출·캐싱
│   ├── encoder.py              # Phase L2: CNN 이미지 인코더
│   ├── sequence_model.py       # Phase L3: LSTM 시퀀스 모델
│   ├── dataset.py              # Phase L4: 21-feat + image-seq 통합 Dataset
│   ├── train_lstm.py           # Phase L5: VPPM-LSTM 학습 루프
│   └── eval_lstm.py            # Phase L6: 평가·시각화
│
├── model.py                    # [수정] VPPM_LSTM 클래스 추가
├── run_pipeline.py             # [수정] --use-lstm 플래그
└── ...

docker/                         # [신규]
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
└── .dockerignore

Sources/pipeline_outputs/               # 디스크 최소화 원칙 — 아래 §10 참조
├── features/                            # [재사용] 기존 21-feat npz 그대로 사용, 재추출 금지
│   ├── B1.*_features.npz                #   (기존, ~1.6MB 전체)
│   ├── all_features.npz                 #   (기존)
│   └── normalization.json               #   (기존)
├── models/                              # [기존 유지]
│   └── vppm_*.pt                        #   (기존 baseline, 삭제 금지)
├── lstm_embeddings/                     # [신규·소용량] per-supervoxel d_lstm 임베딩 저장
│   └── lstm_emb_{channels}_{tag}.npz    #   ≈ 29680 × 16 × 4B ≈ 2MB (전체)
├── models_lstm/                         # [신규] 모든 fold 영속화 (재평가 가능, §10.3)
│   ├── vppm_lstm_{prop}_fold{0..4}.pt   #   4 × 5 = 20개 ≈ 30MB 총합
│   └── vppm_lstm_{prop}_best.pt         #   best fold 심볼릭 포인터 (편의용)
└── results/
    └── vppm_lstm/                       # [신규] 메트릭 JSON + 요약 PNG 1~2장만
```

---

## 4. Phase L1 — 슈퍼복셀 이미지 스택 추출 (`lstm/image_stack.py`)

### 4.1 목표

각 유효 슈퍼복셀 `(ix, iy, iz)` 에 대해 모양 `(T, C, Hs, Ws)` 의 이미지 텐서를 만든다.
- `T = SV_Z_LAYERS = 70` (슈퍼복셀이 걸친 레이어 수; 빌드 상단은 pad/mask 처리)
- `C` = 선택된 채널 수 (아래 4.2 참조)
- `Hs = Ws ≈ ceil(SV_XY_PIXELS) + margin` (기본 8~10 픽셀, `config.LSTM_PATCH_PX`)

### 4.2 입력 채널 선택 옵션 (`LSTM_INPUT_CHANNELS`)

| 옵션 | 채널 구성 | 비고 |
|-----|----------|-----|
| `"raw"` | `slices/camera_data/visible/0` (용융 직후) 단일 채널 | 원본 이미지 정보 최대 보존 |
| `"raw_both"` | `visible/0` + `visible/1` (분말 도포 직후) — 2채널 | 전/후 대조 |
| `"dscnn"` | 8개 DSCNN 클래스 확률맵 | 기존 피처보다 공간 정보 많음 |
| `"raw+dscnn"` | `visible/0` + 8 DSCNN = 9채널 | 기본값. 원본·분할 정보 모두 사용 |

→ 기본값 `raw+dscnn` 으로 시작, ablation 으로 비교.

### 4.3 추출 절차

```python
# lstm/image_stack.py 의사코드
def extract_stack_for_supervoxel(f, grid, ix, iy, iz, channels):
    l0, l1 = grid.get_layer_range(iz)           # 70 레이어
    r0, r1, c0, c1 = grid.get_pixel_range(ix, iy)  # ~8×8 패치
    Hs = Ws = config.LSTM_PATCH_PX
    stack = np.zeros((config.SV_Z_LAYERS, len(channels), Hs, Ws), dtype=np.float16)
    mask  = np.zeros((config.SV_Z_LAYERS,), dtype=bool)   # 유효 레이어 플래그

    for t, layer in enumerate(range(l0, l1)):
        if layer >= grid.num_layers:
            break            # 빌드 상단 pad → mask=False
        patch_channels = []
        for ch_key in channels:
            patch = f[ch_key][layer, r0:r1, c0:c1]
            patch = resize_or_pad(patch, (Hs, Ws))
            patch_channels.append(patch)
        stack[t] = np.stack(patch_channels)
        mask[t]  = True
    return stack, mask
```

### 4.4 캐싱 전략 (디스크 절약 최우선)

> **배경**: 호스트 `/` 가 77% 사용 중(`208 GB` free). ~3.7GB 의 이미지 스택 캐시를 영속화하는 것은 실용적이지만, 사용자의 용량 제약 때문에 **가능하면 로컬 디스크에 남기지 않는다**.

**3-티어 캐시 정책** — 우선순위 높은 쪽부터 선택:

| 티어 | 위치 | 영속성 | 용량 | 선택 기준 |
|-----|-----|-------|-----|-----------|
| **T1 (기본)** | 컨테이너 내부 `/tmp/image_stacks/*.h5` | 휘발성 (컨테이너 종료 시 사라짐) | ~3.7GB, host 디스크 **0** | 기본값. tmpfs/overlay 상에만 존재 |
| **T2 (재실행 시 재활용)** | host `/tmp/vppm_image_stacks/` (bind mount, tmpfs 가능) | OS 재부팅 시 사라짐 | 동일 | 같은 컨테이너를 여러 번 돌릴 때만 사용 |
| **T3 (영속)** | host `Sources/pipeline_outputs/image_stacks/` | 영구 | 동일 | **기본적으로 사용하지 않음.** 디버깅/ablation 반복 시에만 명시적으로 켬 |

- 기본(T1)에서는 Phase L1 이 **동일 run 내에서만** stacks.h5 를 만들고, Phase L6 학습이 끝난 뒤 자동 삭제.
- 학습이 끝나면 **LSTM 이 만들어낸 d_lstm 차원 임베딩만** `Sources/pipeline_outputs/lstm_embeddings/*.npz` 에 보존 (~2MB). 이것만 있으면 재학습/ablation 비교가 가능.
- HDF5 캐시 구조(동일):
  ```
  /tmp/image_stacks/B1.2.h5
    ├── stacks      : float16  (N_sv, 70, C, Hs, Ws)
    ├── masks       : bool     (N_sv, 70)
    ├── voxel_index : int32    (N_sv, 3)
    └── attrs       : channels, patch_px, dtype
  ```
- 행 순서는 **반드시 기존 `features.npz` (B1.*_features.npz / all_features.npz) 의 슈퍼복셀 순서와 정렬**.
- 티어 전환은 `config.py` 의 `LSTM_CACHE_DIR` 로 제어:
  ```python
  LSTM_CACHE_DIR = "/tmp/image_stacks"   # T1 기본 (휘발성)
  LSTM_CACHE_PERSIST = False             # True 로 바꾸면 T3 영속화
  ```

### 4.5 CLI

```bash
python -m Sources.vppm.run_pipeline --phase image-stack --build all \
    --channels raw+dscnn --patch-px 10
```

---

## 5. Phase L2 — CNN 이미지 인코더 (`lstm/encoder.py`)

각 레이어 패치를 `d_cnn` 차원 벡터로 매핑.

```python
class PatchEncoder(nn.Module):
    """(C, Hs, Ws) → d_cnn"""
    def __init__(self, in_ch=9, d_cnn=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),   nn.ReLU(),
            nn.MaxPool2d(2),                          # 10→5
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                  # (64,1,1)
            nn.Flatten(),                             # (64)
        )
        self.proj = nn.Linear(64, d_cnn)
    def forward(self, x):
        return self.proj(self.net(x))
```

- 파라미터 수 <200K (경량). 학습 시 LSTM과 end-to-end 학습.
- 사전학습을 쓰고 싶을 경우 `ResNet18(in_channels=C)` 로 드롭인 교체 가능한 인터페이스 유지.

---

## 6. Phase L3 — LSTM 시퀀스 모델 (`lstm/sequence_model.py`)

```python
class SupervoxelLSTM(nn.Module):
    """입력: (B, T, C, Hs, Ws), mask: (B, T) → (B, d_lstm)"""
    def __init__(self, encoder, d_cnn=64, d_lstm=16,
                 bidirectional=True, num_layers=1):
        super().__init__()
        self.encoder = encoder
        hidden = d_lstm // (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=d_cnn, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
        )
    def forward(self, x, mask):
        B, T, C, H, W = x.shape
        flat = x.view(B * T, C, H, W)
        feats = self.encoder(flat).view(B, T, -1)      # (B,T,d_cnn)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths, batch_first=True, enforce_sorted=False)
        out, (h_n, _) = self.lstm(packed)
        # concat last forward/backward hidden state
        if self.lstm.bidirectional:
            emb = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, d_lstm)
        else:
            emb = h_n[-1]
        return emb
```

옵션:
- **풀링 전략**: `last_hidden` / `mean_pool` / `attention_pool` 세 가지를 config 로 선택.
- **Masking**: 빌드 상단을 걸친 슈퍼복셀은 `T<70` → `pack_padded_sequence` 로 정확히 처리.

---

## 6.5 기존 Phase 1~3 산출물 재사용 (재계산 금지)

현재 호스트에 이미 존재하는 파일들은 **읽기 전용으로 재사용**한다. 어떤 경우에도 다시 만들지 않는다.

| 재사용 파일 | 경로 | 용도 |
|------------|------|-----|
| `B1.*_features.npz` (5개) | `Sources/pipeline_outputs/features/` | 21-feat 입력 `x21` |
| `all_features.npz` | `Sources/pipeline_outputs/features/` | 전체 합본, sample_ids, voxel_index |
| `normalization.json` | `Sources/pipeline_outputs/features/` | 21-feat min/max (LSTM 학습에도 동일 scale 사용) |
| `vppm_{YS,UTS,UE,TE}_fold{0-4}.pt` | `Sources/pipeline_outputs/models/` | Baseline 비교 대상 (삭제·덮어쓰기 금지) |

구현 규칙:
1. Phase L1 은 `all_features.npz` 의 `voxel_index` 를 **먼저 로드** 한 뒤, 해당 슈퍼복셀들만 이미지 스택으로 변환 → 행 정렬 보장.
2. LSTM DataLoader 는 `X21 = np.load("all_features.npz")["features"]` 에서 직접 가져오고, **21-feat 재추출 코드는 호출하지 않는다**.
3. `run_pipeline.py --use-lstm` 은 `--phase features` 를 자동으로 **건너뛴다** (`features/all_features.npz` 존재 시 skip).
4. Baseline 모델 파일(`models/vppm_*.pt`)은 compare 용으로 필요하므로 **도커 볼륨에서 read-only 로 마운트**.

---

## 7. Phase L4 — 결합 데이터셋 (`lstm/dataset.py`)

```python
class VppmLstmDataset(Dataset):
    def __init__(self, features_npz, stacks_h5, labels, indices):
        self.X21  = features_npz["features"][indices]    # (N, 21)
        self.y    = labels[indices]
        self.h5   = h5py.File(stacks_h5, "r")            # lazy
        self.idxs = indices                              # map to stacks row
    def __getitem__(self, i):
        row = self.idxs[i]
        stack = self.h5["stacks"][row]                   # (70,C,Hs,Ws)
        mask  = self.h5["masks"][row]                    # (70,)
        return {
            "x21":   torch.from_numpy(self.X21[i]).float(),
            "img":   torch.from_numpy(stack).float(),
            "mask":  torch.from_numpy(mask),
            "y":     torch.tensor(self.y[i], dtype=torch.float32),
        }
```

- **중요**: `features.npz` 의 순서 ↔ `stacks.h5` 의 `voxel_index` 가 완전히 같아야 함. Phase L1 생성 스크립트에서 보증하고, 로드 시 `assert` 로 재확인.
- Sample-wise CV 분할은 기존 `dataset.py` 의 `sample_ids_per_supervoxel` 를 그대로 재사용.

---

## 8. Phase L5 — VPPM-LSTM 통합 모델 (`model.py` 수정)

```python
class VPPM_LSTM(nn.Module):
    def __init__(self, n_hand=21, d_lstm=16, hidden=128, dropout=0.1,
                 encoder_kwargs=None, lstm_kwargs=None):
        super().__init__()
        enc = PatchEncoder(**(encoder_kwargs or {}))
        self.seq = SupervoxelLSTM(enc, d_lstm=d_lstm, **(lstm_kwargs or {}))
        self.fc1 = nn.Linear(n_hand + d_lstm, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x21, img, mask):
        lstm_emb = self.seq(img, mask)                # (B, d_lstm)
        feat = torch.cat([x21, lstm_emb], dim=1)      # (B, 21 + d_lstm)
        h = self.drop(self.fc1(feat))
        return self.fc2(h)
```

학습 하이퍼파라미터 추가(`config.py`):
```python
# --- LSTM 업그레이드 ---
LSTM_ENABLE              = True
LSTM_INPUT_CHANNELS      = "raw+dscnn"   # raw | raw_both | dscnn | raw+dscnn
LSTM_PATCH_PX            = 10
LSTM_D_CNN               = 64
LSTM_D_EMBED             = 16            # 최종 fusion 에 더해지는 차원
LSTM_BIDIRECTIONAL       = True
LSTM_POOLING             = "last"        # last | mean | attn
LSTM_LR                  = 1e-3          # Adam
LSTM_BATCH_SIZE          = 64            # 이미지 텐서가 커서 1000 → 64
LSTM_NUM_WORKERS         = 4
```

---

## 9. Phase L6 — 학습 & 평가 (`lstm/train_lstm.py`, `lstm/eval_lstm.py`)

### 9.1 기존 학습 대비 변경점

| 항목 | 기존 VPPM | VPPM-LSTM |
|-----|----------|----------|
| 배치 크기 | 1000 | 64 (GPU 메모리) |
| 디바이스 | CPU로도 충분 | **GPU 필수** (CNN+LSTM) |
| Epoch 당 시간 | 수 초 | ~수십 초 (캐시된 stack 기준) |
| Optimizer | Adam lr=1e-3 | Adam lr=1e-3, grad-clip 1.0 |
| Loss | L1 | L1 (동일, 논문 호환) |
| Early stop | patience=50 | patience=20 (epoch이 무거움) |
| CV | 5-fold sample-wise | 동일 |

### 9.2 Ablation

논문의 Table 8 확장 버전:

| 실험 | 21-hand | LSTM | 비고 |
|-----|---------|-----|------|
| Baseline VPPM | ✔ | ✗ | 재현 결과 |
| LSTM-only | ✗ | ✔ | LSTM 단독 기여도 |
| VPPM-LSTM (raw) | ✔ | ✔ (raw 채널만) | 원본 이미지만 사용 |
| VPPM-LSTM (dscnn) | ✔ | ✔ (DSCNN 채널만) | 시간 순서 효과 측정 |
| **VPPM-LSTM (raw+dscnn)** | ✔ | ✔ | 기본 제안 |
| Unidirectional LSTM | ✔ | ✔ | 양방향 효과 검증 |

### 9.3 평가 메트릭 / 시각화

기존 `evaluate.py` 와 동일 (RMSE, naive 대비 감소율, correlation plot, per-sample min 집계) + **기여도 분석**:
- `LSTM_D_EMBED=0` (꺼짐) 과 `>0` 의 차이
- Grad-CAM 또는 integrated gradients 로 특정 예측에서 어떤 레이어 t 가 중요했는지 시각화.

---

## 10. Docker 컨테이너화 — 디스크 최소화 우선

### 10.1 설계 원칙 (용량 제약 반영)

현재 호스트 `/` 가 **77% 사용 중 (208GB free)**. "필요한 것만 볼륨 처리" 라는 요구를 최우선으로 반영한다.

- **재현성**: PyTorch+CUDA 버전, h5py, scipy 고정.
- **원본 HDF5 → read-only mount**: 230GB 를 이미지에 복사하지 않음.
- **기존 산출물 → 선별적 mount**:
  - `features/` 는 **read-only** (재사용 대상, 덮어쓰기 금지)
  - `models/` (baseline 20개 .pt) 는 **read-only** (비교용)
- **신규 산출물 → 최소 폴더만 read-write mount**:
  - `lstm_embeddings/` (~2MB): 학습 후 임베딩만 영속화
  - `models_lstm/` (~100KB): best fold 모델만 영속화 (§10.3)
  - `results/vppm_lstm/` (~수 MB): 메트릭 JSON + 요약 플롯만
- **대용량 중간 캐시는 볼륨 mount 하지 않음**:
  - `image_stacks/*.h5` (~3.7GB) 는 **컨테이너 내부 `/tmp`** 에만 쓰고, 종료 시 사라지게 함.
  - 이렇게 하면 호스트 디스크 증가분은 **약 10MB 미만**.
- **CPU/GPU 양쪽 지원**: `vppm-lstm:cpu`, `vppm-lstm:gpu`.

### 10.2 `docker/Dockerfile`

```dockerfile
# 기본 이미지: CUDA 런타임 + cuDNN
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      libhdf5-dev build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

COPY Sources/        ./Sources/
COPY CLAUDE.md       ./CLAUDE.md
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "Sources.vppm.run_pipeline", "--use-lstm", "--phase", "train"]
```

### 10.3 모델 저장 정책 (모든 fold 영속화)

재평가·Ablation·K-fold 앙상블을 위해 **baseline 과 동일하게 4 property × 5 fold = 20개 전부 저장**. VPPM-LSTM 은 CNN+LSTM 가중치 포함 ~1~2MB/fold → 총 **~30MB** 로 부담 없음.

| 저장 대상 | 파일명 | 크기 | 설명 |
|---------|-------|-----|-----|
| **fold별 모델** | `models_lstm/vppm_lstm_{YS,UTS,UE,TE}_fold{0..4}.pt` | ~30MB (20개) | 각 fold 의 state_dict, 재평가·앙상블 용 |
| **best fold 포인터** | `models_lstm/vppm_lstm_{prop}_best.json` | <1KB | `{"best_fold": 3, "val_rmse": 36.7}` 메타데이터 |
| **5-fold CV 요약 메트릭** | `results/vppm_lstm/cv_metrics.json` | <100KB | fold 별 RMSE, 평균±표준편차 |
| **학습 로그** | `results/vppm_lstm/training_log.json` | <500KB | epoch·loss·val_rmse 기록 |

학습 loop 슈도코드:
```python
fold_metrics = []
for fold in range(5):
    state, val_rmse = train_one_fold(...)
    torch.save(state, f"models_lstm/vppm_lstm_{prop}_fold{fold}.pt")
    fold_metrics.append({"fold": fold, "val_rmse": val_rmse})

best = min(fold_metrics, key=lambda x: x["val_rmse"])
save_json(f"models_lstm/vppm_lstm_{prop}_best.json", best)
```

재평가 시에는 `models_lstm/vppm_lstm_{prop}_fold{k}.pt` 를 직접 로드해 `eval_lstm.py` 로 돌리면 됨 → 전체 학습 재실행 불필요.

### 10.4 `docker/docker-compose.yml` (선별 마운트)

```yaml
services:
  vppm-lstm:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: vppm-lstm:gpu
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTHONPATH=/workspace
      - LSTM_CACHE_DIR=/tmp/image_stacks        # 컨테이너 내부, 호스트 저장 X
    volumes:
      # ─── 읽기 전용 (원본 + 재사용 산출물) ─────────────────────────
      - /home/taehoon/3DP_TensileProp_Prediction/ORNL_Data_Origin:/workspace/ORNL_Data_Origin:ro
      - /home/taehoon/3DP_TensileProp_Prediction/Sources/pipeline_outputs/features:/workspace/Sources/pipeline_outputs/features:ro
      - /home/taehoon/3DP_TensileProp_Prediction/Sources/pipeline_outputs/models:/workspace/Sources/pipeline_outputs/models:ro

      # ─── 쓰기 가능 (신규 산출물만, 소용량) ─────────────────────────
      - /home/taehoon/3DP_TensileProp_Prediction/Sources/pipeline_outputs/lstm_embeddings:/workspace/Sources/pipeline_outputs/lstm_embeddings:rw
      - /home/taehoon/3DP_TensileProp_Prediction/Sources/pipeline_outputs/models_lstm:/workspace/Sources/pipeline_outputs/models_lstm:rw
      - /home/taehoon/3DP_TensileProp_Prediction/Sources/pipeline_outputs/results/vppm_lstm:/workspace/Sources/pipeline_outputs/results/vppm_lstm:rw

      # ─── 주의: image_stacks/ 는 volume 에 없음! ────────────────
      #   컨테이너 내부 /tmp/image_stacks 에만 존재 → 호스트 디스크 증가분 0

    tmpfs:
      - /tmp:size=8g                           # 이미지 스택 캐시 장소 (RAM 기반)
    shm_size: "4gb"
    working_dir: /workspace
```

**중요 포인트**:
1. `pipeline_outputs/` 전체를 마운트하지 않고 **하위 3개 폴더만 선별 mount**. 누락 폴더(`image_stacks/`)는 컨테이너 내부 임시 공간으로만 존재.
2. `features/`, `models/` 는 `:ro` → 실수로 기존 산출물을 덮어쓸 수 없음.
3. `tmpfs /tmp` 는 RAM 기반. 이미지 스택 캐시(~3.7GB)가 RAM 에 올라가므로 **host 디스크 증가분은 0**. RAM 이 부족하면 `tmpfs` 블록을 제거해 컨테이너 overlayfs 에 쓰되 여전히 종료 시 삭제.
4. 호스트에서 사전에 다음 디렉터리를 생성해 두어야 bind mount 실패하지 않음:
   ```bash
   mkdir -p Sources/pipeline_outputs/{lstm_embeddings,models_lstm,results/vppm_lstm}
   ```
5. CPU-only 환경: `runtime: nvidia`, `NVIDIA_VISIBLE_DEVICES` 제거 후 `vppm-lstm:cpu` 빌드.

### 10.5 호스트 디스크 증가분 예상

| 항목 | 크기 | 비고 |
|-----|-----|-----|
| `lstm_embeddings/*.npz` | ~2 MB | d_lstm=16, float32, 29680 rows |
| `models_lstm/*_fold{0..4}.pt` | ~30 MB | 4 property × 5 fold (재평가용 전부 보존) |
| `models_lstm/*_best.json` | <4 KB | best fold 포인터 |
| `results/vppm_lstm/*.json` | <1 MB | cv_metrics + training_log + ablation |
| `results/vppm_lstm/*.png` | ~1~3 MB | correlation plot 등 핵심 2~3장 |
| **총 증가분** | **~35 MB** | 여전히 매우 적음 |

이와 별개로 **Docker image 자체**(CUDA+PyTorch)는 `/var/lib/docker` 에 ~6~8GB. 용량이 빠듯하면 CPU 이미지(~2GB)부터 시도 권장.

### 10.4 `docker/entrypoint.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# 결과 디렉터리가 마운트되어 있는지 확인
if [ ! -d /workspace/Sources/pipeline_outputs ]; then
  echo "[entrypoint] FATAL: pipeline_outputs not mounted" >&2
  exit 1
fi

# GPU 사용 가능 여부 로그
python - <<'PY'
import torch
print("[entrypoint] torch:", torch.__version__, "cuda:", torch.cuda.is_available())
PY

exec "$@"
```

### 10.6 실행 방법 (end-to-end, 단일 run 권장)

stack 캐시가 휘발성이므로 **"캐시 생성 → 학습 → 임베딩 저장 → 평가"** 를 한 번의 컨테이너 실행 안에서 끝낸다:

```bash
# 0) 사전: 호스트에 쓰기용 폴더 생성
mkdir -p Sources/pipeline_outputs/{lstm_embeddings,models_lstm,results/vppm_lstm}

# 1) 이미지 빌드
docker compose -f docker/docker-compose.yml build

# 2) 전체 파이프라인 (1-shot run: 캐시 → 학습 → 평가 → 임베딩 저장)
docker compose -f docker/docker-compose.yml run --rm vppm-lstm \
    python -m Sources.vppm.run_pipeline --use-lstm --phase all

# 산출물 확인 (호스트)
ls Sources/pipeline_outputs/lstm_embeddings/    # lstm_emb_*.npz
ls Sources/pipeline_outputs/models_lstm/        # vppm_lstm_*_best.pt
ls Sources/pipeline_outputs/results/vppm_lstm/  # cv_metrics.json, *.png
```

- **기존 features/** 는 read-only 로 마운트되어 재사용됨 → 21-feat 재추출 없음.
- **image_stacks** 는 컨테이너 내부 `/tmp` 에만 존재 → 종료 시 자동 삭제.
- 여러 번 돌릴 경우에도 위 명령을 반복하면 되고, 호스트 디스크에 축적되는 건 `lstm_embeddings/`, `models_lstm/`, `results/vppm_lstm/` 뿐.

---

## 11. 구현 순서 및 의존성

```
(기존 Phase 1~3 완료 전제)
        │
        ▼
L1. image_stack.py  ── 빌드당 stacks.h5 캐시 생성
        │
        ▼
L2. encoder.py      ── PatchEncoder 단위 테스트 (shape 검증)
        │
        ▼
L3. sequence_model.py ── SupervoxelLSTM 단위 테스트 (masking 검증)
        │
        ▼
L4. lstm/dataset.py ── 21-feat ↔ stacks 행 정렬 assert
        │
        ▼
L5. model.py 수정  ── VPPM_LSTM 통합
        │
        ▼
L6. train_lstm.py   ── 5-fold × 4-property 학습
        │
        ▼
L7. eval_lstm.py    ── Ablation table, correlation plot 생성
        │
        ▼
L8. Dockerfile/compose ── 컨테이너 빌드, 볼륨 마운트로 end-to-end 실행
```

---

## 12. 리스크와 대응

| 리스크 | 영향 | 대응 |
|-------|------|-----|
| `image_stacks/*.h5` 용량 과다 (>10GB) | 디스크 부족 | **호스트에 저장 안 함 (컨테이너 `/tmp` tmpfs)**, `float16` 저장, 채널 옵션, 패치 축소 |
| tmpfs RAM 부족 (~3.7GB 필요) | OOM | `LSTM_INPUT_CHANNELS="dscnn"`(8ch) 로 축소 → ~3.2GB, 또는 `LSTM_PATCH_PX=8` → ~2.4GB |
| 기존 features.npz 구조 불일치 | 재사용 실패 | Phase L1 첫 단계에서 `all_features.npz["voxel_index"]` 키 존재를 assert; 없으면 즉시 실패 |
| HDF5 원본 순차 I/O 로 stack 생성이 느림 (수 시간) | 반복 개발 정체 | Phase L1 을 **1회성 전처리**로 고정, 이후 학습은 캐시만 사용 |
| GPU 메모리 부족 (batch 64 에서도 OOM) | 학습 불가 | `LSTM_PATCH_PX=8` 축소, `grad checkpoint`, batch 32로 자동 폴백 |
| LSTM fusion 후 21-feat 신호가 희석 | 성능 저하 | `LayerNorm(21)`, `LayerNorm(d_lstm)` 를 각각 넣은 뒤 concat |
| Overfitting (작은 데이터 + CNN 추가) | 검증 RMSE 악화 | dropout 0.2, weight_decay 1e-4, early stop patience 줄임 |
| 이미지 정렬 오류 (supervoxel ↔ pixel 범위) | 피처 엉킴 | Phase L1 에 `assert` 기반 무결성 체크 + 시각화 유틸 `debug_patch_overlay()` |
| Docker GPU 패스스루 실패 | 컨테이너 학습 불가 | `nvidia-container-toolkit` 설치 확인, CPU 폴백 이미지 `vppm-lstm:cpu` 병행 제공 |

---

## 13. 검증 체크포인트

| 단계 | 검증 항목 |
|-----|---------|
| L1 | `stacks.h5` 행 수 == `features.npz` 행 수, 무작위 10 슈퍼복셀의 패치를 원본 HDF5 직접 접근과 픽셀 단위 비교 |
| L2 | `PatchEncoder((B,9,10,10)) → (B,64)` shape 및 backward 통과 |
| L3 | `mask` 가 일부 False 인 샘플에서도 packed sequence 에러 없음 |
| L4 | DataLoader 한 배치가 `{x21:(B,21), img:(B,70,C,H,W), mask:(B,70), y:(B,)}` 구조 |
| L5 | `VPPM_LSTM.forward` 가 baseline VPPM 과 같은 출력 shape `(B,1)` |
| L6 | Baseline VPPM 대비 UTS RMSE 가 **개선**되는지 (목표: 38.3 → <37 MPa) |
| L7 | Ablation 에서 `LSTM-only` 가 `DSCNN-only` 보다 좋음을 확인 (시간 순서 기여도 입증) |
| L8 | 컨테이너 종료 후 호스트의 `Sources/pipeline_outputs/models/vppm_lstm/*.pt` 존재 확인 |

---

## 14. requirements.txt 추가분

```
torch>=2.3
torchvision>=0.18
h5py>=3.10
scipy>=1.11
scikit-learn>=1.3
tqdm>=4.66
matplotlib>=3.8
```

---

## 15. 기존 계획과의 관계

- **기존 21개 피처 파이프라인(`features.py`, `dataset.py`)은 변경하지 않는다.** Phase L1~L8 는 기존 산출물(`features.npz`, sample-wise split) 에 **덧붙이는 형태**로 동작한다.
- 따라서 Baseline VPPM 재현이 완료된 상태에서 Phase L1 부터 순차 진행하면, 기존 결과와 VPPM-LSTM 결과를 동일 분할·동일 metric 으로 **fair comparison** 할 수 있다.
- 슬라이드 한 장 요약:
  > *"한 슈퍼복셀을 구성하는 70장의 레이어 이미지를 시간 순서대로 CNN+LSTM 으로 인코딩해 16차원 임베딩을 얻고, 이를 기존 21개 핸드크래프트 피처에 붙여 VPPM 의 입력을 21 → 37 차원으로 확장한다. 학습·데이터·모델은 Docker 로 컨테이너화하고, 이미지 스택 캐시와 학습 산출물은 bind-mount volume 을 통해 로컬에 영속화한다."*
