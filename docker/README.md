# Docker — VPPM 학습/평가 인프라

## 구조

```
docker/
├── baseline/           # Baseline VPPM (21-feat) 학습 — Phase 4
│   ├── docker-compose.yml
│   └── run.sh
├── ablation/           # Feature ablation 실험 모음 — Phase 5
│   ├── README.md       # 상세
│   ├── Dockerfile      # 공용 이미지 (vppm-ablation:gpu) - baseline/ 도 재사용
│   ├── entrypoint.sh
│   ├── run_all.sh      # E1~E4 4-GPU 병렬
│   ├── dscnn/          # E1
│   ├── sensor/         # E2
│   ├── cad/            # E3
│   ├── scan/           # E4
│   ├── combined/       # E13 (DSCNN+Sensor 조합)
│   ├── dscnn_sub/      # E5~E12 + E23/E24
│   ├── sensor_sub/     # E14~E22
│   └── scan_sub/       # E31~E33
├── lstm/               # VPPM-LSTM 변형
└── run_v2_all.sh       # 마스터 오케스트레이터 (Phase 4 + Phase 5)
```

## 마스터 스크립트 (Baseline + 모든 Ablation 자동 실행)

```bash
# 전체: baseline + 27 ablation
cd docker
./run_v2_all.sh

# baseline 건너뛰고 ablation 만
./run_v2_all.sh --skip-baseline

# smoke test (전체 ~20 분)
./run_v2_all.sh --quick
```

마스터 스크립트의 사전 검증:
1. **현재 진행 중인 `run_pipeline --phase features` 가 있으면 종료까지 대기**
2. `all_features.npz` 의 피처 #19, #20 std > 0 확인 → 0 상수면 즉시 중단
3. GPU 4 장 + 도커 데몬 접근성 확인

순차 실행 단계:
1. Phase 4 — Baseline v2 (`docker/baseline/`)
2. Phase 5 — 27 ablation:
   - E1~E4 (`ablation/run_all.sh` 4-GPU 병렬)
   - E5~E12 + E23/24 (`ablation/dscnn_sub/run_all.sh` 4-GPU × 3 배치)
   - E13 (`ablation/combined/run.sh` 단독)
   - E14~E22 (`ablation/sensor_sub/run_all.sh` 4-GPU × 3 배치)
   - E31~E33 (`ablation/scan_sub/run_all.sh` 3-GPU 병렬)
3. summary.md 통합 재생성

총 예상 시간: **~3~4 시간 (4-GPU)**.

로그: `/tmp/v2_full_rerun_<timestamp>/` 에 단계별 저장.

## 단독 실행 (개별 단계)

### Phase 4 — Baseline v2 만

```bash
cd docker/baseline
./run.sh             # GPU 0 기본
./run.sh --gpu 2     # GPU 지정
```

### Phase 5 — Ablation 그룹별

```bash
# 주요 그룹 (E1~E4)
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
