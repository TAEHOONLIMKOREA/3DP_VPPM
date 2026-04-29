# docker/lstm_dual

VPPM-LSTM-Dual (visible/0 + visible/1 두 채널 CNN+LSTM → 23-feat MLP) 컨테이너 실행 환경.

## 사전조건

- `Sources/pipeline_outputs/features/all_features.npz` (baseline 피처 추출 산출물)
- `Sources/pipeline_outputs/experiments/vppm_lstm/cache/crop_stacks_B1.{1..5}.h5` (visible/0 캐시 — `docker/lstm` 으로 빌드)

## 실행

```bash
cd docker/lstm_dual

# 한 번에 (cache_v1 → train → evaluate)
docker compose up -d --build

# 로그
docker compose logs -f

# 정리
docker compose down
```

## 단계 / GPU 변경

`.env` 수정 또는 인라인:

```bash
NVIDIA_VISIBLE_DEVICES=2 docker compose up -d --build
LSTM_DUAL_PHASE=cache_v1 docker compose up -d --build
LSTM_DUAL_PHASE=train LSTM_DUAL_EXTRA=--quick docker compose up -d --build
```

산출물 → `Sources/pipeline_outputs/experiments/vppm_lstm_dual/`
