# docker/1dcnn

VPPM-1DCNN: baseline (21-feat MLP) 의 z-축 평균 압축을 **채널별 depthwise 1D CNN** 으로 교체하는 실험.
입력 `(B, 21, 70)` → 2층 DepthwiseConv1d(k=3, groups=21) → AdaptiveAvgPool1d → 21차원 → baseline `VPPM(21→128→1)` MLP 재사용.
baseline 과 MLP 후단이 동일하므로 RMSE 차이가 1D CNN 압축 자체의 효과를 직접 반영한다.

자세한 모델 설계: `Sources/vppm/1dcnn/PLAN.md`

## 사전조건

| Phase | 필요 파일 |
|:--|:--|
| `features` | `Sources/pipeline_outputs/features/all_features.npz` (baseline 피처 추출 산출물) |
| `train` / `evaluate` / `all` | 위 + `Sources/pipeline_outputs/experiments/vppm_1dcnn/features/features_seq.npz` |

의존 그래프:

```
baseline all_features.npz
    └── vppm_1dcnn features (features_seq.npz)
            └── vppm_1dcnn train / evaluate
```

`all_features.npz` 가 없을 경우:

```bash
cd docker/baseline
docker compose up --build   # 또는 run_pipeline --phase features
```

## 실행

```bash
cd docker/1dcnn

# Phase 1: layer 시퀀스 캐시 빌드 (5 빌드 HDF5 파싱, 약 60–90분/빌드)
docker compose up -d --build
docker compose logs -f
docker compose down

# Phase 2/3: 학습 + 평가 (features_seq.npz 존재 후)
VPPM_1DCNN_PHASE=train docker compose up -d --build
docker compose logs -f
docker compose down

# 전체 한번에 (features → train → evaluate)
VPPM_1DCNN_PHASE=all docker compose up -d --build
```

## 단계 / GPU / 빌드 변경

`.env` 수정 또는 인라인:

```bash
# 다른 GPU 사용
NVIDIA_VISIBLE_DEVICES=2 docker compose up -d --build

# 단계별 실행
VPPM_1DCNN_PHASE=features docker compose up -d --build
VPPM_1DCNN_PHASE=train docker compose up -d --build
VPPM_1DCNN_PHASE=evaluate docker compose up -d --build

# sanity check (빌드 1개, epoch 5, patience 3)
VPPM_1DCNN_PHASE=features VPPM_1DCNN_EXTRA=--quick docker compose up --build

# 특정 빌드만 features 추출
VPPM_1DCNN_PHASE=features VPPM_1DCNN_EXTRA="--build B1.2" docker compose up --build
```

## 환경변수 표

| 변수 | 기본값 | 설명 |
|:--|:--|:--|
| `VPPM_1DCNN_PHASE` | `features` | `features` / `train` / `evaluate` / `all` |
| `VPPM_1DCNN_EXTRA` | (빈 문자열) | 추가 CLI 인자. `--quick`, `--build B1.X` 등 |
| `NVIDIA_VISIBLE_DEVICES` | `1` (`.env` 기본) | 사용할 GPU 인덱스 |
| `UID_GID` | `1001:1001` | 호스트 UID:GID (`id -u`:`id -g` 로 확인) |

## 산출물 위치

```
Sources/pipeline_outputs/experiments/vppm_1dcnn/
├── features/
│   ├── features_seq.npz          # (N_sv, 70, 21) raw + masks + counts
│   └── normalization.json
├── models/
│   ├── vppm_1dcnn_{YS,UTS,UE,TE}_fold{0..4}.pt
│   └── training_log.json
└── results/
    ├── metrics_summary.json
    ├── predictions_{YS,UTS,UE,TE}.csv
    └── correlation_plots.png
```

## 트러블슈팅

**`FATAL: all_features.npz not found`**
- baseline 피처 추출이 완료되지 않음. `cd docker/baseline && docker compose up --build` 먼저 실행.

**`FATAL: features_seq.npz not found`**
- 1DCNN features phase 가 완료되지 않음. `VPPM_1DCNN_PHASE=features docker compose up --build` 먼저 실행.

**`FATAL: UID/권한` 또는 쓰기 실패**
- `.env` 의 `UID_GID` 를 호스트 `id -u`:`id -g` 결과로 맞춰줄 것.
- `.env` 가 맞는데도 실패하면 docker 가 첫 실행 시 mount target (`Sources/pipeline_outputs/experiments/vppm_1dcnn/`) 을 root 소유로 자동 생성한 경우다. 다음 중 하나로 해결:
  ```bash
  # (a) 비어 있다면 그냥 지우고 재실행 — compose 의 mkdir 가 컨테이너 user 로 새로 만든다
  sudo rm -rf /home/taehoon/3DP_VPPM/Sources/pipeline_outputs/experiments/vppm_1dcnn
  # (b) 산출물이 이미 있다면 소유권만 바꾼다
  sudo chown -R $(id -u):$(id -g) /home/taehoon/3DP_VPPM/Sources/pipeline_outputs/experiments/vppm_1dcnn
  ```

**GPU 미인식**
- `NVIDIA_VISIBLE_DEVICES` 인덱스 확인. `nvidia-smi` 로 사용 가능한 GPU 목록 확인 후 변경.
