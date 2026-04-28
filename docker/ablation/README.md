# Docker — VPPM Feature Ablation

[Sources/vppm/ablation/PLAN.md](../../Sources/vppm/ablation/PLAN.md) 의 4개 그룹 제거 실험을 각각 **독립 도커 컨테이너** 로 실행하고, 산출물을 호스트 볼륨으로 회수한다.

## 전제 조건

- 호스트에 NVIDIA GPU + `nvidia-docker2` (compose 의 `runtime: nvidia`)
- 호스트 `venv/` 가 이미 torch + cuda 로 설치되어 있어야 함 (컨테이너는 bind-mount 로 사용)
- `Sources/pipeline_outputs/features/all_features.npz` 생성 완료 (baseline 피처 추출)

## 구조

```
docker/ablation/
├── Dockerfile          # 공용 이미지 — vppm-ablation:gpu
├── entrypoint.sh       # 마운트 검증 + torch/cuda 점검
├── run_all.sh          # 4 그룹 순차 실행
├── dscnn/              # E1 — DSCNN 8 피처 제거
│   ├── docker-compose.yml
│   └── run.sh
├── sensor/             # E2 — Temporal 센서 7 피처 제거
│   ├── docker-compose.yml
│   └── run.sh
├── cad/                # E3 — CAD/좌표 3 피처 제거
│   ├── docker-compose.yml
│   └── run.sh
├── scan/               # E4 — 스캔 3 피처 제거 (laser_module + return_delay + stripe_boundaries)
│   ├── docker-compose.yml
│   └── run.sh
├── combined/           # E13 — DSCNN+Sensor 동시 제거 (15 피처 제거, 6 피처 학습)
│   ├── docker-compose.yml
│   └── run.sh
├── dscnn_sub/          # E5~E12 + E23/E24 — DSCNN 8채널 + 2 묶음
│   ├── README.md       # 상세 — PLAN_dscnn_subablation.md 와 같이 볼 것
│   ├── docker-compose.yml   # EXPERIMENT_ID / NVIDIA_VISIBLE_DEVICES 파라미터
│   ├── run.sh               # 단일 실험
│   └── run_all.sh           # 4-GPU 3배치 병렬
├── sensor_sub/         # E14~E22 — 센서 서브 채널 9종
│   ├── README.md       # 상세 — PLAN_sensor_subablation.md 와 같이 볼 것
│   ├── docker-compose.yml   # EXPERIMENT_ID / NVIDIA_VISIBLE_DEVICES 파라미터
│   ├── run.sh               # 단일 실험
│   └── run_all.sh           # 4-GPU 3배치 병렬
└── scan_sub/           # E31~E33 — 스캔(G4) 재구현 후 ablation
    ├── README.md       # 상세 — scan 서브 채널 ablation
    ├── docker-compose.yml   # EXPERIMENT_ID / NVIDIA_VISIBLE_DEVICES 파라미터
    ├── run.sh               # 단일 실험
    └── run_all.sh           # 3-GPU 단일 배치 병렬
```

> **주의 (scan_sub)**: E31~E33 실행 전에 `scan_features.py` 구현 → 피처 재추출 → baseline v2
> 재학습이 **호스트에서** 선행되어야 한다. 상세는 [scan_sub/README.md](./scan_sub/README.md) 참조.

## 볼륨 매핑 (모든 그룹 동일)

| 호스트 | 컨테이너 | 모드 | 용도 |
|---|---|:---:|---|
| `venv/` | `/workspace/venv` | ro | torch + cuda 런타임 |
| `Sources/pipeline_outputs/features/` | 동일 | ro | `all_features.npz` 입력 |
| `Sources/pipeline_outputs/results/` | 동일 | ro | baseline `metrics_raw.json` 참조 |
| `Sources/pipeline_outputs/ablation/` | 동일 | rw | **산출물 — 호스트로 그대로 회수** |

## 실행 방법

### 단일 그룹

```bash
cd docker/ablation/dscnn
./run.sh              # 전체 학습
./run.sh --quick      # smoke test (20 epoch, 2분 이내)
```

### 4개 그룹 순차

```bash
cd docker/ablation
./run_all.sh
./run_all.sh --quick
```

### 수동 compose (원하면)

```bash
cd docker/ablation/sensor
docker compose build
docker compose run --rm ablation-sensor
```

## 산출물

각 그룹 컨테이너는 자기 그룹의 실험 폴더만 씁니다:

```
Sources/pipeline_outputs/ablation/
├── E1_no_dscnn/        # dscnn 컨테이너가 생성
│   ├── experiment_meta.json
│   ├── models/…
│   ├── results/…
│   └── features/normalization.json
├── E2_no_sensor/       # sensor 컨테이너
├── E3_no_cad/          # cad 컨테이너
├── E4_no_scan/         # scan 컨테이너
└── summary.md          # 주의: 컨테이너마다 덮어씀 (하단 참고)
```

## GPU 할당 (병렬 실행)

`run_all.sh` 는 4 컨테이너를 각자 다른 GPU 에 핀시켜 동시에 실행한다.

| 그룹 | 실험 | GPU |
|:----:|:----:|:---:|
| dscnn | E1 | 0 |
| sensor | E2 | 1 |
| cad | E3 | 2 |
| scan | E4 | 3 |

GPU 핀은 각 `<group>/docker-compose.yml` 의 `NVIDIA_VISIBLE_DEVICES=<N>` 로 설정.
GPU 4장이 없는 환경이면 compose 파일의 값을 편집해 같은 GPU 에 겹치게 하거나,
`run_all.sh` 대신 그룹별 `run.sh` 를 순차 실행하면 된다.

## summary.md 흐름

- **단일 그룹 실행**(`dscnn/run.sh` 등): 해당 실험 한 줄만 포함한 summary 가 즉시 작성됨.
- **`run_all.sh` 병렬 실행**: race 방지를 위해 각 컨테이너는 `--skip-summary` 로 요약을 건너뛰고, 모든 컨테이너 완료 후 호스트 venv 로 `--rebuild-summary` 를 호출해 4 실험을 통합한 summary.md 를 한 번에 생성.
- 언제든 결과를 다시 합치고 싶으면:
  ```bash
  ./venv/bin/python -m Sources.vppm.ablation.run --rebuild-summary
  ```

## 트러블슈팅

- **`venv not mounted`**: 호스트 `venv/bin/python` 이 실행 가능해야 함.
- **`all_features.npz 없음`**: `./venv/bin/python -m Sources.vppm.run_pipeline --phase features` 를 먼저 수행.
- **권한 에러 (UID mismatch)**: 도커 기본 root 로 실행된다. 산출물 디렉터리가 비-루트 소유면 `chown -R $(id -u):$(id -g) Sources/pipeline_outputs/ablation` 로 후처리.
- **GPU 비가시**: `runtime: nvidia` 지원 여부 확인 (`docker run --rm --gpus all nvidia/cuda:12.6.0-base nvidia-smi`).
