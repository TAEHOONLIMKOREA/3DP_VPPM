# Docker — VPPM 학습/평가 인프라

## 구조

```
docker/
├── baseline/           # Baseline VPPM (21-feat) 학습
│   ├── docker-compose.yml
│   └── run.sh
├── ablation/           # Feature ablation 실험 모음 (총 27개)
│   ├── README.md       # 상세
│   ├── Dockerfile      # 공용 이미지 (vppm-ablation:gpu) — baseline/ 도 재사용
│   ├── entrypoint.sh
│   ├── run_all.sh      # 27개 ablation 전체 실행 + summary 통합
│   ├── dscnn/          # E1
│   ├── sensor/         # E2
│   ├── cad/            # E3
│   ├── scan/           # E4
│   ├── combined/       # E13 (DSCNN+Sensor 조합)
│   ├── dscnn_sub/      # E5~E12 + E23/E24
│   ├── sensor_sub/     # E14~E22
│   └── scan_sub/       # E31~E33
└── lstm/               # VPPM-LSTM 변형
```

## Baseline 학습

```bash
cd docker/baseline
./run.sh             # GPU 0 기본
./run.sh --gpu 2     # GPU 지정
```

산출물: `Sources/pipeline_outputs/results/vppm_origin/`

## Ablation 전체 실행 (27개 + summary 통합)

```bash
cd docker/ablation
./run_all.sh           # 전체 (~3 시간, 4-GPU)
./run_all.sh --quick   # smoke test
```

`run_all.sh` 의 사전 검증:
1. 진행 중인 `run_pipeline --phase features` 가 있으면 종료까지 대기
2. `all_features.npz` 의 피처 #19, #20 std > 0 확인 → 0 상수면 즉시 중단
3. GPU 수 + 도커 데몬 접근성 확인

순차 실행 그룹:
1. E1~E4   main groups        (4-GPU 병렬,        ~30 min)
2. E5~E12 + E23/24  dscnn_sub (4-GPU × 3 배치,    ~45 min)
3. E13     combined           (1 GPU,             ~30 min)
4. E14~E22 sensor_sub         (4-GPU × 3 배치,    ~45 min)
5. E31~E33 scan_sub           (3-GPU 병렬,        ~30 min)
6. summary.md 통합 재생성

로그: `/tmp/ablation_full_<timestamp>/` 에 단계별 저장.

## Ablation 그룹별 단독 실행

```bash
# 주요 그룹 (E1~E4) — run_all.sh 내부에서도 자동 실행
cd docker/ablation && ./run_all.sh

# DSCNN 서브 (E5~E12, E23, E24)
cd docker/ablation/dscnn_sub && ./run_all.sh

# 조합 (E13)
cd docker/ablation/combined && ./run.sh

# 센서 서브 (E14~E22)
cd docker/ablation/sensor_sub && ./run_all.sh

# 스캔 서브 (E31~E33)
cd docker/ablation/scan_sub && ./run_all.sh
```

## 공통 사항

- 모든 컨테이너가 단일 이미지 `vppm-ablation:gpu` 공유 → 첫 실행에서만 빌드
- 호스트 `venv/` 를 read-only bind mount → 컨테이너 안에서 pip install 불필요
- `Sources/pipeline_outputs/{features,results,ablation,models}` 를 bind mount → 산출물은 호스트로 회수
- GPU 핀: `NVIDIA_VISIBLE_DEVICES=N` 환경변수로 제어
- UID/GID: `${UID_GID:-1000:1000}` — 호스트 사용자 권한 유지

## 트러블슈팅

| 증상 | 확인/조치 |
|:-----|:----|
| `venv not mounted` | `venv/bin/python` 이 실행 가능해야 함 |
| `all_features.npz 없음` | `./venv/bin/python -m Sources.vppm.run_pipeline --phase features` 먼저 |
| `scan 피처 0 상수 거부` | scan_features.py 가 적용된 features 가 필요 — 재추출 |
| 권한 에러 (UID mismatch) | `Sources/pipeline_outputs/ablation` 가 사용자 소유인지 확인 |
| `runtime: nvidia` 인식 안 됨 | `nvidia-docker2` 설치 + `docker run --rm --gpus all nvidia/cuda:12.6.0-base nvidia-smi` 로 확인 |
| GPU 4장 미만 | `run_all.sh` 의 배치 GPU ID 편집 또는 단일 GPU 순차 실행 |
