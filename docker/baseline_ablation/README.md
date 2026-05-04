# Docker — VPPM Feature Ablation

[Sources/vppm/baseline_ablation/PLAN.md](../../Sources/vppm/baseline_ablation/PLAN.md) 의 30 ablation 실험을 **단일 docker-compose 파일** 로 통합 실행한다. wrapper 스크립트 없이 `docker compose` 명령만으로 모든 시나리오 커버.

## 전제 조건

- 호스트에 NVIDIA GPU + `nvidia-docker2` (compose 의 `runtime: nvidia`)
- 호스트 `venv/` 가 이미 torch + cuda 로 설치되어 있어야 함 (컨테이너는 bind-mount 로 사용)
- `Sources/pipeline_outputs/features/all_features.npz` 생성 완료 (baseline 피처 추출)

## 구조

```
docker/baseline_ablation/
├── Dockerfile          공용 이미지 — vppm-ablation:gpu (검증/실행 로직은 compose `command:` 에 인라인)
├── docker-compose.yml  30 실험 + summary service. profiles 로 그룹/실험 단위 실행
├── .env                기본값 (UID_GID, ABLATION_EXTRA)
└── README.md           ← 본 문서
```

## 실험 매핑 (30개)

| Profile | 실험 | 비고 |
|---|---|---|
| `main` | E1, E2, E3, E4 | DSCNN/Sensor/CAD/Scan 그룹 통째 제거 — 4 GPU 병렬 |
| `dscnn_sub` | E5~E12, E23, E24 | DSCNN 채널 단위 제거 — 4-GPU rotation 3 batch |
| `combined` | E13 | DSCNN + Sensor 동시 제거 |
| `sensor_sub` | E14~E22 | Sensor 채널 단위 제거 — 4-GPU rotation 3 batch (4-4-1) |
| `scan_sub` | E31~E33 | Scan 채널 단위 제거 — 3 GPU 병렬 |
| `cad_sub` | E34~E36 | CAD 채널 단위 제거 — 3 GPU 병렬 |
| `summary` | — | 디스크 결과 스캔 → ablation/summary.md 통합 재생성 |
| `E<N>` | 단일 실험 | 예: `--profile E7` |

GPU pin 은 compose 안에 하드코드 (4-GPU 병렬 배치 가정). 다른 토폴로지면 compose 파일 수정.

## 사용법

### 그룹 단위 (권장 — 한 그룹 끝나면 다음 그룹 실행)

```bash
cd docker/baseline_ablation
docker compose --profile main up        # E1~E4   (~30 분)
docker compose --profile dscnn_sub up   # 10 실험 (~45 분, 내부 batching 자동)
docker compose --profile combined up    # E13     (~30 분)
docker compose --profile sensor_sub up  # 9 실험  (~45 분)
docker compose --profile scan_sub up    # 3 실험  (~30 분)
docker compose --profile cad_sub up     # 3 실험  (~30 분)
docker compose --profile summary up     # ablation/summary.md 재생성
```

`up` 은 (`-d` 없이) 해당 profile 의 모든 service 가 끝날 때까지 block. 백그라운드로 돌리려면 `up -d` + 별도 터미널에서 `docker compose logs -f`.

### 단일 실험만

```bash
docker compose --profile E7 up                          # E7 (No-Streaking) 만
docker compose --profile E7 up -d --build               # 백그라운드
docker compose logs -f E7                               # 로그 따라가기
```

### Smoke test (epochs=20, ~5분)

```bash
ABLATION_EXTRA=--quick docker compose --profile E1 up
```
또는 `.env` 의 `ABLATION_EXTRA=--quick` 수정.

### 정리

```bash
docker compose down                                     # 종료된 컨테이너 정리
```

## 내부 메커니즘

### depends_on 으로 GPU rotation batching

`dscnn_sub` profile 에서 E5~E12 + E23, E24 를 4-GPU 에 배치 스케줄링:

- **batch 1**: E5(GPU0), E6(GPU1), E7(GPU2), E8(GPU3) 동시 시작
- **batch 2**: E9~E12 — 각각 batch1 의 같은 GPU 컨테이너 완료 후 시작 (`depends_on E5/E6/E7/E8`)
- **batch 3**: E23(GPU0), E24(GPU1) — batch2 의 E9, E10 완료 후

`sensor_sub` 도 동일 패턴 (4-4-1).

`required: false` 옵션 덕분에 단일 실험 profile (`--profile E9`) 도 의존성 없이 단독 실행 가능.

### YAML anchor 로 boilerplate 공유

`x-base`, `x-env`, `x-cmd` anchor 가 27 service 공통 설정 (volumes, image, validation 명령) 을 한 번만 정의. 각 service 는 GPU/EXPERIMENT_ID 만 다름.

### 검증 인라인

이전 `entrypoint.sh` 의 검증 (venv 마운트, all_features.npz 존재, ablation 디렉터리 쓰기 권한) 이 `command:` 의 shell 명령 prefix 로 인라인됨. `set -e` + `exec python -m ...` 패턴.
