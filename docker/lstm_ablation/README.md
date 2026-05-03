# docker/lstm_ablation

VPPM-LSTM-Full-Stack Ablation 실험 컨테이너 실행 환경.

풀-스택 모델 (`lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4`) 의 분기를 토글하여
각 분기의 기여도를 측정한다. 7-flag 토글 모델로 E1–E7 을 모두 지원.

### 시리즈 1 — 카메라 분기 제거 (E1-E2)

| 실험 | 설명 | MLP 입력 | kept | removed |
|:--|:--|:--:|:--|:--|
| E1 (no_v0) | visible/0 분기 제거 | 70 | feat_static, v1, sensor, dscnn, cad, scan | branch_v0 |
| E2 (no_cameras) | visible/0+v1 모두 제거 | 54 | feat_static, sensor, dscnn, cad, scan | branch_v0, branch_v1 |

### 시리즈 2 — 단일 분기 isolation (E3-E7)

| 실험 | 설명 | MLP 입력 | kept | 출력 디렉터리 |
|:--|:--|:--:|:--|:--|
| E3 (only_v0_img) | visible/0 단독 (feat_static 포함 나머지 제거) | 16 | branch_v0 | E3_only_v0_img/ |
| E4 (only_dscnn)  | DSCNN 8-class 단독 | 8 | branch_dscnn | E4_only_dscnn/ |
| E5 (only_cad)    | CAD patch 단독 | 8 | branch_cad | E5_only_cad/ |
| E6 (only_scan)   | Scan patch 단독 | 8 | branch_scan | E6_only_scan/ |
| E7 (only_sensor) | Sensor 7-field 1D-CNN 단독 | 28 | branch_sensor | E7_only_sensor/ |

풀-스택의 모든 캐시(v0, v1, sensor, dscnn, cad_patch, scan_patch)를 그대로 재사용 — 새로운 캐시 빌드 불필요.
각 실험의 출력 디렉터리는 compose 실행 시 자동으로 mkdir 됨.

## 사전조건

모든 실험 (E1-E7) 공통 — 풀-스택 캐시 전체 필요:

- `Sources/pipeline_outputs/features/all_features.npz`
- `Sources/pipeline_outputs/experiments/vppm_lstm/cache/crop_stacks_B1.{1..5}.h5` (v0 캐시)
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.{1..5}.h5` (v1 캐시)
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7/cache/sensor_stacks_B1.{1..5}.h5`
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_4_sensor_7_dscnn_8/cache/dscnn_stacks_B1.{1..5}.h5`
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache/cad_patch_stacks_B1.{1..5}.h5`
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache/scan_patch_stacks_B1.{1..5}.h5`

출력 디렉터리 (`E1_no_v0/`, `E3_only_v0_img/` 등) 는 compose 실행 시 자동 mkdir — 미리 생성 불필요.

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

## 빌드

```bash
cd docker/lstm_ablation
docker compose build
```

## 풀런 — 시리즈 1 (E1 + E2, 카메라 제거)

```bash
cd docker/lstm_ablation
docker compose up -d --build e1 e2
docker compose logs -f
docker compose down
```

E1 은 GPU 1, E2 는 GPU 2 에서 병렬 실행 (`.env` 에서 변경 가능).

## 풀런 — 시리즈 2 (E3-E7, 단일 분기 isolation)

```bash
cd docker/lstm_ablation
# 개별 실행
docker compose up -d --build e3
docker compose up -d --build e4
docker compose up -d --build e5
docker compose up -d --build e6
docker compose up -d --build e7

# 모두 동시 실행 (GPU 3-7 필요)
docker compose up -d --build e3 e4 e5 e6 e7

# 로그 확인
docker compose logs -f e3
docker compose down
```

## 단계/GPU 변경

```bash
# GPU 변경 (GPU 가 8 개 미만일 때)
NVIDIA_VISIBLE_DEVICES=0 docker compose up -d --build e3
NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm e3   # 순차 실행

# quick smoke (1 fold × YS × max_epochs=5)
LSTM_ABLATION_EXTRA=--quick docker compose run --rm e3
LSTM_ABLATION_EXTRA=--quick docker compose run --rm e4
LSTM_ABLATION_EXTRA=--quick docker compose run --rm e5
LSTM_ABLATION_EXTRA=--quick docker compose run --rm e6
LSTM_ABLATION_EXTRA=--quick docker compose run --rm e7

# 평가만 (모델 .pt 파일이 이미 존재할 때)
docker compose run --rm e3 /workspace/venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E3 --phase evaluate
```

## 산출물

`Sources/pipeline_outputs/experiments/lstm_ablation/`

```
E1_no_v0/            models/ results/ features/
E2_no_cameras/       models/ results/ features/
E3_only_v0_img/      models/ results/ features/   (MLP 입력 16-d, branch_v0 단독)
E4_only_dscnn/       models/ results/ features/   (MLP 입력  8-d, branch_dscnn 단독)
E5_only_cad/         models/ results/ features/   (MLP 입력  8-d, branch_cad 단독)
E6_only_scan/        models/ results/ features/   (MLP 입력  8-d, branch_scan 단독)
E7_only_sensor/      models/ results/ features/   (MLP 입력 28-d, branch_sensor 단독)
```

각 실험: `vppm_lstm_ablation_E{n}_{YS,UTS,UE,TE}_fold{0-4}.pt` (fold당 .pt × 4 props × 5 folds = 20 파일)

예상 소요 시간: 각 실험 약 3–4시간 (4 props × 5 folds × max_epochs=5000, early_stop_patience=50).

## GPU 핀 정책

기본값: e1=1, e2=2, e3=3, e4=4, e5=5, e6=6, e7=7.
GPU 가 부족하면 환경 변수로 오버라이드:

```bash
NVIDIA_VISIBLE_DEVICES=0 docker compose up -d --build e3
```

또는 `.env` 의 `NVIDIA_VISIBLE_DEVICES=0` 으로 일괄 변경 (모든 서비스 같은 GPU — 순차 실행 시만 안전).

## 참고

- 공통 계획: `Sources/vppm/lstm_ablation/PLAN.md`
- 시리즈 1: `Sources/vppm/lstm_ablation/PLAN_E1_no_v0.md`, `PLAN_E2_no_cameras.md`
- 시리즈 2: `Sources/vppm/lstm_ablation/PLAN_E3_only_v0_img.md` ~ `PLAN_E7_only_sensor.md`
- 베이스 모델 docker: `docker/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/`
