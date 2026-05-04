# Docker — VPPM Hidden-Dim Sweep

[Sources/vppm/baseline_ablation_with_lstm/PLAN_hidden_dim_sweep.md](../../Sources/vppm/baseline_ablation_with_lstm/PLAN_hidden_dim_sweep.md) 의 hidden_dim sweep 실험을 단일 docker-compose 로 실행.

베이스라인 VPPM (`Linear(21 → hidden) → ReLU → Dropout → Linear(hidden → 1)`) 에서 hidden_dim 만 바꿔가며 model capacity 의 RMSE 영향을 측정한다.

| ID | hidden | 비고 |
|:--:|:------:|---|
| H1 | 1   | bottleneck — 1차원 capacity 하한 |
| H2 | 64  | plateau 진입 후보 |
| H3 | 256 | 상위 capacity 검증 |
| (E0) | 128 | 기존 baseline (`Sources/pipeline_outputs/experiments/vppm_baseline/`) — 재학습 X, summary 표에서만 비교 기준으로 인용 |

## 전제 조건

- 호스트에 NVIDIA GPU + `nvidia-docker2` (compose 의 `runtime: nvidia`)
- 호스트 `venv/` 가 이미 torch + cuda 로 설치되어 있어야 함 (컨테이너는 bind-mount 로 사용)
- `Sources/pipeline_outputs/features/all_features.npz` 생성 완료

## 구조

```
docker/hidden_sweep/
├── Dockerfile          vppm-hidden-sweep:gpu (검증/실행 로직은 compose `command:` 에 인라인)
├── docker-compose.yml  단일 service `hidden-sweep` — 환경변수로 모드 전환
├── .env                기본값 (UID_GID, NVIDIA_VISIBLE_DEVICES, HIDDEN_SWEEP_HIDDEN, HIDDEN_SWEEP_EXTRA)
└── README.md           ← 본 문서
```

## 실행

```bash
cd docker/hidden_sweep
docker compose up -d --build       # 백그라운드 풀런
docker compose logs -f             # 진행 로그
docker compose down                # 정리
```

`up` 은 (`-d` 없이) sweep 이 끝날 때까지 block. 백그라운드는 `up -d` + `docker compose logs -f`.

## 모드 변경

```bash
# 단일 hidden 만 학습:
HIDDEN_SWEEP_HIDDEN=1 docker compose up -d --build
HIDDEN_SWEEP_HIDDEN=64 docker compose up -d --build
HIDDEN_SWEEP_HIDDEN=256 docker compose up -d --build

# 다른 GPU 사용:
NVIDIA_VISIBLE_DEVICES=2 docker compose up -d --build

# Smoke 학습 (epoch=20, ~5분/run × 3 hidden × 4 prop × 5 fold):
HIDDEN_SWEEP_EXTRA=--quick docker compose up

# 학습 없이 hidden_sweep_summary.md 만 디스크 스캔으로 재생성:
HIDDEN_SWEEP_EXTRA=--rebuild-summary docker compose up
```

또는 `.env` 파일을 수정.

## 산출물

- 각 hidden run: `Sources/pipeline_outputs/experiments/baseline_ablation_with_lstm/H{1,2,3}_hidden_{1,64,256}/`
  - `models/` — fold 별 best 모델
  - `results/` — `metrics_raw.json`, 산점도/상관 plot
  - `features/normalization.json`
  - `experiment_meta.json`
- 통합 summary: `Sources/pipeline_outputs/experiments/baseline_ablation_with_lstm/hidden_sweep_summary.md` (E0 baseline 자동 인용)

## 예상 소요 시간

- 1 run (4 prop × 5 fold) ≈ 3~5 분 (CPU 기준; CUDA 면 더 짧음)
- 전체 sweep (`HIDDEN_SWEEP_HIDDEN=all`) = 3 × 4 × 5 = 60 run ≈ **3~5 시간**
- 단일 hidden = **1~1.5 시간**
- `--quick` smoke = epoch 20 으로 단축, 전체 약 30~60 분
