# docker/lstm_dual_img_4_sensor_7_dscnn_8

VPPM-LSTM-Dual-Img-4-Sensor-7-DSCNN-8 컨테이너 실행 환경.

4-stream LSTM 모델: visible/0(d_embed=4) + visible/1(d_embed=4) + sensor temporal(d_embed=7) + DSCNN segmentation(d_embed=8) → MLP 29.
sensor_7 모델에 DSCNN 분기(layer 단위 8채널 segmentation 결과, Gaussian blur 전처리)를 추가한 구조.
MLP 입력 29-feat(21 baseline에서 sensor 7개 제거 = 14-feat + 4+4+7+8 임베딩 = 37, 또는 실험 PLAN 참조)은 sensor_7 와 동일하게 controlled.
자세한 동기는 `Sources/vppm/lstm_dual_img_4_sensor_7_dscnn_8/PLAN.md` 참조.

## 사전조건

- `Sources/pipeline_outputs/features/all_features.npz` (baseline 피처)
- `Sources/pipeline_outputs/experiments/vppm_lstm/cache/crop_stacks_B1.{1..5}.h5` (visible/0 캐시 — `docker/lstm` 으로 빌드)
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.{1..5}.h5` (visible/1 캐시 — `docker/lstm_dual` 의 cache_v1 단계로 빌드)
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7/cache/sensor_stacks_B1.{1..5}.h5` (sensor 캐시 — `docker/lstm_dual_img_4_sensor_7` 의 cache_sensor 단계로 빌드)
- `ORNL_Data_Origin/*.hdf5` (DSCNN 캐시 빌드 시 필요 — `--phase cache_dscnn`)

사전조건 빠른 확인:

```bash
ls Sources/pipeline_outputs/features/all_features.npz
ls Sources/pipeline_outputs/experiments/vppm_lstm/cache/crop_stacks_B1.*.h5
ls Sources/pipeline_outputs/experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.*.h5
ls Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7/cache/sensor_stacks_B1.*.h5
ls ORNL_Data_Origin/*.hdf5 | wc -l   # 5개여야 함
```

## 빌드

```bash
cd docker/lstm_dual_img_4_sensor_7_dscnn_8
docker compose build
```

## DSCNN 캐시 빌드

ORNL raw HDF5 에서 layer 단위 8채널 segmentation 결과를 추출하고 Gaussian blur 전처리.
빌드당 10–20 분, 5 빌드 합계 약 1시간 소요.

```bash
cd docker/lstm_dual_img_4_sensor_7_dscnn_8
LSTM_DIDS_PHASE=cache_dscnn docker compose up -d --build
docker compose logs -f
docker compose down
```

## 풀런 (cache_dscnn → train → evaluate)

```bash
cd docker/lstm_dual_img_4_sensor_7_dscnn_8
docker compose up -d --build
docker compose logs -f
docker compose down
```

예상 소요 시간: DSCNN 캐시 ~1h + 학습/평가 ~3.5h = 총 약 4.5h (GPU 1장 기준).
캐시가 이미 빌드된 경우 학습+평가만 약 3.5h.

## 단계별 부분 실행

`.env` 수정 또는 인라인:

```bash
# DSCNN 캐시만 빌드 (ORNL raw HDF5 필요, 빌드당 10-20분 × 5빌드 ≈ 1h)
LSTM_DIDS_PHASE=cache_dscnn docker compose up -d --build

# 학습만 (캐시 4종 모두 존재해야 함)
LSTM_DIDS_PHASE=train docker compose up -d --build

# 평가만 (모델 파일 존재해야 함)
LSTM_DIDS_PHASE=evaluate docker compose up -d --build

# GPU 변경
NVIDIA_VISIBLE_DEVICES=2 docker compose up -d --build

# quick smoke (1 fold × YS only, 약 10분) — 전체 파이프라인 동작 확인
LSTM_DIDS_EXTRA=--quick docker compose up -d --build
```

## 산출물

`Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7_dscnn_8/`

```
cache/          dscnn_stacks_B1.x.h5 (DSCNN 캐시)
models/         fold별 .pt 체크포인트
results/        metrics.json, scatter plots, correlation plots
features/       normalization.json, experiment_meta.json
```

학습 완료 후 결과 해석: `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7_dscnn_8/results/README.md`

## 참고

- 실험 계획: `Sources/vppm/lstm_dual_img_4_sensor_7_dscnn_8/PLAN.md`
- 부모 실험 docker: `docker/lstm_dual_img_4_sensor_7/`
