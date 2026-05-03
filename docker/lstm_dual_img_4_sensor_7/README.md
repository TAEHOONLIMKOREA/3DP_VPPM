# docker/lstm_dual_img_4_sensor_7

VPPM-LSTM-Dual-Img-4-Sensor-7 컨테이너 실행 환경.

visible/0(d_embed=4) + visible/1(d_embed=4) 카메라 LSTM 에 sensor temporal LSTM(d_embed=7) 을 추가한 쿼드 스트림 모델.
baseline 21-feat 에서 sensor 그룹 7개를 제거하고(14-feat) 대신 sensor LSTM 임베딩으로 대체.
자세한 동기는 `Sources/vppm/lstm_dual_img_4_sensor_7/PLAN.md` 참조.

## 사전조건

- `Sources/pipeline_outputs/features/all_features.npz` (baseline 피처)
- `Sources/pipeline_outputs/experiments/vppm_lstm/cache/crop_stacks_B1.{1..5}.h5` (visible/0 캐시 — `docker/lstm` 으로 빌드)
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.{1..5}.h5` (visible/1 캐시 — `docker/lstm_dual` 의 cache_v1 단계로 빌드)
- `ORNL_Data/Co-Registered In-Situ and Ex-Situ Dataset/[baseline] (Peregrine v2023-11)/*.hdf5` (sensor 캐시 빌드 시 필요 — `--phase cache_sensor`)

사전조건 빠른 확인:

```bash
ls Sources/pipeline_outputs/features/all_features.npz
ls Sources/pipeline_outputs/experiments/vppm_lstm/cache/crop_stacks_B1.*.h5
ls Sources/pipeline_outputs/experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.*.h5
ls "ORNL_Data/Co-Registered In-Situ and Ex-Situ Dataset/[baseline] (Peregrine v2023-11)/"*.hdf5 | wc -l   # 5개여야 함
```

## 실행

```bash
cd docker/lstm_dual_img_4_sensor_7

# 풀런 (cache_sensor → train → evaluate)
docker compose up -d --build

# 로그
docker compose logs -f

# 정리
docker compose down
```

## 단계별 부분 실행

`.env` 수정 또는 인라인:

```bash
# sensor 캐시만 빌드 (ORNL raw HDF5 필요, 수십 분)
LSTM_DISH_PHASE=cache_sensor docker compose up -d --build

# 학습만 (캐시 3종 모두 존재해야 함)
LSTM_DISH_PHASE=train docker compose up -d --build

# 평가만 (모델 파일 존재해야 함)
LSTM_DISH_PHASE=evaluate docker compose up -d --build

# GPU 변경
NVIDIA_VISIBLE_DEVICES=2 docker compose up -d --build

# quick smoke (epochs=20, patience=10) — 캐시 빌드 포함 전체 파이프라인
LSTM_DISH_PHASE=train LSTM_DISH_EXTRA=--quick docker compose up -d --build
```

## 산출물

`Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7/`

```
cache/          sensor_B1.x.h5 (sensor 캐시)
models/         fold별 .pt 체크포인트
results/        metrics.json, scatter plots, correlation plots
features/       normalization.json, experiment_meta.json
```
