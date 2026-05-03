# docker/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4

VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-1DCNN-Sensor-4 컨테이너 실행 환경.

7-stream 모델: visible/0(d_embed=16) + visible/1(d_embed=16) + sensor 1D-CNN(4 fields×1) + dscnn(d_embed=8) + cad_patch(d_embed=8) + scan_patch(d_embed=8) → MLP 86.
dscnn_8 모델에 CAD 패치(edge/overhang 근접도 8×8 패치)와 Scan 패치(return_delay/stripe 8×8 패치)를 추가한 풀-스트림 구조.
센서 인코더를 LSTM 에서 per-field 1D-CNN 으로 변경. 자세한 동기는 `Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md` 참조.

## 사전조건

- `Sources/pipeline_outputs/features/all_features.npz` (baseline 피처)
- `Sources/pipeline_outputs/experiments/vppm_lstm/cache/crop_stacks_B1.{1..5}.h5` (visible/0 캐시 — `docker/lstm` 으로 빌드)
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.{1..5}.h5` (visible/1 캐시 — `docker/lstm_dual` 의 cache_v1 단계로 빌드)
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7/cache/sensor_stacks_B1.{1..5}.h5` (sensor 캐시 — `docker/lstm_dual_img_4_sensor_7` 의 cache_sensor 단계로 빌드)
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7_dscnn_8/cache/dscnn_stacks_B1.{1..5}.h5` (dscnn 캐시 — `docker/lstm_dual_img_4_sensor_7_dscnn_8` 의 cache_dscnn 단계로 빌드)
- `ORNL_Data_Origin/*.hdf5` (CAD/Scan 패치 캐시 빌드 시 필요)

사전조건 빠른 확인:

```bash
ls Sources/pipeline_outputs/features/all_features.npz
ls Sources/pipeline_outputs/experiments/vppm_lstm/cache/crop_stacks_B1.*.h5
ls Sources/pipeline_outputs/experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.*.h5
ls Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7/cache/sensor_stacks_B1.*.h5
ls Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7_dscnn_8/cache/dscnn_stacks_B1.*.h5
ls ORNL_Data_Origin/*.hdf5 | wc -l   # 5개여야 함
```

## 빌드

```bash
cd docker/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4
docker compose build
```

## CAD 패치 캐시 빌드

ORNL raw HDF5 에서 layer 단위 CAD 패치(edge/overhang 근접도 인버전 + cad_mask 픽셀곱)를 추출.
빌드당 약 5–15분, 5 빌드 합계 약 30–60분 소요.

```bash
cd docker/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4
LSTM_FULL86_PHASE=cache_cad_patch docker compose run --rm vppm-lstm-full86
```

## Scan 패치 캐시 빌드

ORNL raw HDF5 에서 layer 단위 Scan 패치(return_delay + stripe_boundaries, NaN→0)를 추출.
빌드당 약 5–15분, 5 빌드 합계 약 30–60분 소요.

```bash
cd docker/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4
LSTM_FULL86_PHASE=cache_scan_patch docker compose run --rm vppm-lstm-full86
```

## 풀런 (cache_cad_patch → cache_scan_patch → train → evaluate)

```bash
cd docker/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4
docker compose up -d --build
docker compose logs -f
docker compose down
```

예상 소요 시간: CAD 캐시 ~1h + Scan 캐시 ~1h + 학습/평가 ~4h = 총 약 6h (GPU 1장 기준).
캐시가 모두 빌드된 경우 학습+평가만 약 4h.

## 단계별 부분 실행

`.env` 수정 또는 인라인:

```bash
# CAD 패치 캐시만 빌드 (ORNL raw HDF5 필요)
LSTM_FULL86_PHASE=cache_cad_patch docker compose up -d --build

# Scan 패치 캐시만 빌드 (ORNL raw HDF5 필요)
LSTM_FULL86_PHASE=cache_scan_patch docker compose up -d --build

# 학습만 (캐시 6종 모두 존재해야 함)
LSTM_FULL86_PHASE=train docker compose up -d --build

# 평가만 (모델 파일 존재해야 함)
LSTM_FULL86_PHASE=evaluate docker compose up -d --build

# GPU 변경
NVIDIA_VISIBLE_DEVICES=2 docker compose up -d --build

# quick smoke (epochs=20, patience=10) — 전체 파이프라인 동작 확인
LSTM_FULL86_EXTRA=--quick docker compose up -d --build
```

## 산출물

`Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/`

```
cache/          cad_patch_stacks_B1.x.h5, scan_patch_stacks_B1.x.h5
models/         fold별 .pt 체크포인트
results/        metrics.json, scatter plots, correlation plots
features/       normalization.json, experiment_meta.json
```

학습 완료 후 결과 해석: `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/results/README.md`

## 참고

- 실험 계획: `Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md`
- 부모 실험 docker: `docker/lstm_dual_img_4_sensor_7_dscnn_8/`
