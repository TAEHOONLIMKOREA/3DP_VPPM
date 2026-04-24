# Docker — 센서 서브 채널 Ablation (E14~E22)

[PLAN_sensor_subablation.md](../../../Sources/vppm/ablation/PLAN_sensor_subablation.md) 의
Temporal 센서 7채널 서브 ablation 및 2개 묶음 실험을 **도커 컨테이너에서** 실행한다.

E1~E4, E13 과 동일한 공용 이미지 `vppm-ablation:gpu` 를 공유 (`docker/ablation/Dockerfile`).

## 실험 9종

| ID | drop_group | 제거 피처 idx | 의미 |
|:--:|:----------:|:-------------:|:----|
| E14 | `sensor_print_time`   | [11]          | 레이어 프린트 시간 |
| E15 | `sensor_top_flow`     | [12]          | 상단 가스 유량 |
| E16 | `sensor_bottom_flow`  | [13]          | 하단 가스 유량 |
| E17 | `sensor_oxygen`       | [14]          | **산소 농도** — 주요 관심 |
| E18 | `sensor_plate_temp`   | [15]          | 플레이트 온도 |
| E19 | `sensor_flow_temp`    | [16]          | 가스 온도 |
| E20 | `sensor_ventilator`   | [17]          | 환풍기 유량 |
| E21 | `sensor_gas_flow_all` | [12,13,17]    | 유량 3채널 묶음 |
| E22 | `sensor_thermal_all`  | [15,16]       | 온도 2채널 묶음 |

## 전제 조건

- 호스트 NVIDIA GPU 4장 (RTX 5090 기준 검증) + `nvidia-docker2` (compose `runtime: nvidia`)
- 호스트 `venv/` 에 torch + cuda 설치되어 있어야 함 (bind-mount 로 사용)
- `Sources/pipeline_outputs/features/all_features.npz` 존재 (baseline 피처 추출 완료)

## 실행 방법

### 단일 실험

```bash
cd docker/ablation/sensor_sub
./run.sh E17                   # GPU 0 기본
./run.sh E17 --gpu 2           # GPU 2 지정
./run.sh E17 --gpu 0 --quick   # smoke test
```

### 9개 실험 전체 (4-GPU 배치 병렬)

```bash
cd docker/ablation/sensor_sub
./run_all.sh              # 전체 (~30–45분)
./run_all.sh --quick      # smoke (~5분)
```

배치 스케줄:

| Batch | 병렬 실행 | GPU 배정 |
|:-----:|:---------|:--------|
| 1/3 | E14·E15·E16·E17 | 0·1·2·3 |
| 2/3 | E18·E19·E20·E21 | 0·1·2·3 |
| 3/3 | E22            | 0       |

각 배치 내부는 병렬, 배치 간은 순차. 로그는 `/tmp/sensor_sub_logs_<timestamp>/E??.log`.

### 수동 compose (필요 시)

```bash
cd docker/ablation/sensor_sub
EXPERIMENT_ID=E17 NVIDIA_VISIBLE_DEVICES=2 docker compose run --rm ablation-sensor-sub
```

## 볼륨 매핑 (E1~E4 와 동일)

| 호스트 | 컨테이너 | 모드 | 용도 |
|---|---|:---:|---|
| `venv/` | `/workspace/venv` | ro | torch + cuda |
| `Sources/pipeline_outputs/features/` | 동일 | ro | `all_features.npz` |
| `Sources/pipeline_outputs/results/` | 동일 | ro | baseline 참조 |
| `Sources/pipeline_outputs/ablation/` | 동일 | rw | **산출물** |

## 산출물

```
Sources/pipeline_outputs/ablation/
├── E14_no_sensor_print_time/
├── E15_no_sensor_top_flow/
├── E16_no_sensor_bottom_flow/
├── E17_no_sensor_oxygen/
├── E18_no_sensor_plate_temp/
├── E19_no_sensor_flow_temp/
├── E20_no_sensor_ventilator/
├── E21_no_sensor_gas_flow_all/
├── E22_no_sensor_thermal_all/
└── summary.md         # run_all.sh 마지막에 재생성됨
```

## summary.md 흐름

- 각 컨테이너는 `--skip-summary` 로 실행되어 중간 덮어쓰기 방지.
- 모든 배치 완료 후 호스트 `venv` 로 `Sources.vppm.ablation.run --rebuild-summary` 를 호출해
  기존 E1~E4, E13 까지 포함한 **통합 summary.md** 를 재생성.
- 수동 재생성:
  ```bash
  ./venv/bin/python -m Sources.vppm.ablation.run --rebuild-summary
  ```

## 트러블슈팅

- **GPU 부족**: 호스트에 GPU 4장이 아니면 `run_all.sh` 의 배치 편집 필요.
  또는 `./run.sh E14 --gpu 0 && ./run.sh E15 --gpu 0 && ...` 로 단일 GPU 순차.
- **권한 에러 (UID mismatch)**: `Sources/pipeline_outputs/ablation` 가 현재 사용자 소유인지 확인.
- **`EXPERIMENT_ID` 미지정**: compose 에서 `${EXPERIMENT_ID:?...}` 로 즉시 실패.
  항상 `run.sh` 또는 env 지정 후 호출.
- **이미지 재빌드**: `docker/ablation/Dockerfile` 수정 시 `run.sh` 가 자동으로 `compose build` 수행.
  필요하면 `docker image rm vppm-ablation:gpu` 후 재빌드.

## 후속 분석

실험 완료 후:

```bash
# 빌드별 잔차 분해 (예: E17 = No-Oxygen)
./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment E17
# → Sources/pipeline_outputs/ablation/E17_no_sensor_oxygen/per_build_analysis.md
```

9개 실험 결과 집계는 별도 리포트 작성 — `PLAN_sensor_subablation.md §3.4` 참조.
