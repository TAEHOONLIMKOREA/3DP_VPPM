# docker/lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1

VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-Sensor-1 컨테이너 실행 환경 — **4-GPU 병렬(property-parallel)**.

7-stream 모델: visible/0(d_embed=16) + visible/1(d_embed=16) + sensor LSTM(1) + dscnn(d_embed=8) + cad_patch(d_embed=8) + scan_patch(d_embed=8) → MLP 59.
fullstack(`_1dcnn_sensor_4`) 대비 sensor 인코더를 per-field 1D-CNN 에서 multi-channel LSTM 으로 교체하여 temporal 의존성 포착 방식 차이를 실험한다.
신규 캐시 빌드 없음 — CAD/Scan 패치 캐시를 fullstack 디렉터리에서 그대로 재사용. 자세한 동기는 `Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/PLAN.md` 참조.

## 4-GPU 병렬 구조

`docker compose up -d --build` 한 번으로 5 컨테이너가 자동 실행:

| 컨테이너 | GPU (`.env`) | 역할 |
|:--|:--:|:--|
| `train-ys`  | `GPU_YS=0`  | YS  property × 5 fold 학습 |
| `train-uts` | `GPU_UTS=1` | UTS property × 5 fold 학습 |
| `train-ue`  | `GPU_UE=2`  | UE  property × 5 fold 학습 |
| `train-te`  | `GPU_TE=3`  | TE  property × 5 fold 학습 |
| `evaluate`  | `GPU_EVAL=0` | 4 train 완료 후(`depends_on: service_completed_successfully`) 모든 prop 모델을 한번에 평가 → `metrics_summary.json` + plots |

**왜 evaluate 만 단독?** [`baseline/evaluate.py::save_metrics`](../../Sources/vppm/baseline/evaluate.py) 가 `metrics_summary.json`/`metrics_raw.json`/`predictions_*.csv` 를 한 디렉터리에 덮어쓰는 구조라, 4 컨테이너가 동시에 evaluate 호출하면 race condition 으로 일부 prop 결과가 누락된다.

## 사전 호스트 mkdir (한 번만)

docker daemon 이 bind mount target 디렉터리를 자동 생성하면 **root 소유** 가 되어 컨테이너의 일반 사용자 (UID 1001) 가 못 씀. 미리 호스트에서 만들어 두면 `taehoon` 소유로 생성된다:

```bash
mkdir -p /home/taehoon/3DP_VPPM/Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/{models,results,features}
```

이미 root 소유 디렉터리가 있다면:

```bash
sudo chown -R taehoon:taehoon /home/taehoon/3DP_VPPM/Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1
```

## 사전조건 캐시

- `Sources/pipeline_outputs/features/all_features.npz` (baseline 피처)
- `Sources/pipeline_outputs/experiments/vppm_lstm/cache/crop_stacks_B1.{1..5}.h5` (visible/0)
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.{1..5}.h5` (visible/1)
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7/cache/sensor_stacks_B1.{1..5}.h5` (sensor)
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7_dscnn_8/cache/dscnn_stacks_B1.{1..5}.h5` (dscnn)
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache/cad_patch_stacks_B1.{1..5}.h5`
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache/scan_patch_stacks_B1.{1..5}.h5`

사전조건 빠른 확인:

```bash
ls Sources/pipeline_outputs/features/all_features.npz
ls Sources/pipeline_outputs/experiments/vppm_lstm/cache/crop_stacks_B1.*.h5
ls Sources/pipeline_outputs/experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.*.h5
ls Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7/cache/sensor_stacks_B1.*.h5
ls Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7_dscnn_8/cache/dscnn_stacks_B1.*.h5
ls Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache/cad_patch_stacks_B1.*.h5
ls Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache/scan_patch_stacks_B1.*.h5
```

## 풀런 (4 train 동시 → evaluate 자동)

```bash
cd docker/lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1
docker compose up -d --build       # 5 컨테이너 시작 (4 train 동시 + 1 evaluate 대기)
docker compose logs -f             # 모든 service 로그 합쳐서 실시간
docker compose logs -f train-ys    # 특정 service 만
docker compose ps                  # 상태 한눈에
docker compose down                # 끝났거나 중단
```

예상 소요 시간: train 약 1h (단일 GPU 4h 가 4-way 병렬화). evaluate 는 수 분.

## 부분 실행 / 옵션 변경

`.env` 수정 또는 인라인 환경변수:

```bash
# GPU 매핑 변경 (예: GPU 4-7 사용)
GPU_YS=4 GPU_UTS=5 GPU_UE=6 GPU_TE=7 GPU_EVAL=4 docker compose up -d --build

# quick smoke (4 train 모두 epochs=20, patience=10) — 5-10분이면 끝
SENSOR_1_EXTRA=--quick docker compose up -d --build

# 단일 service 만 (예: TE 만 재학습)
docker compose up -d --build train-te
# 그 후 evaluate 만 따로
docker compose up -d --build evaluate

# evaluate 만 (모든 prop 모델 .pt 가 이미 존재할 때)
docker compose up -d --build evaluate
```

## 산출물

`Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/`

```
models/         fold별 .pt 체크포인트 (4 prop × 5 fold = 20개)
results/        metrics_summary.json, metrics_raw.json,
                predictions_{YS,UTS,UE,TE}.csv, correlation_plots.png, scatter_plot_uts.png
features/       normalization.json
experiment_meta.json
```

`models/` 안의 `training_log_{YS,UTS,UE,TE}.json` 은 4 train 컨테이너가 prop 별로 분리 저장한 fold-level 학습 로그.

## 참고

- 실험 계획: `Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/PLAN.md`
- CAD/Scan 패치 캐시 빌드 (사전조건): `docker/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/`
- 부모 실험 (단일 GPU): `docker/lstm_dual_img_4_sensor_7_dscnn_8/`
