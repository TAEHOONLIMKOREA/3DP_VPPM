# docker/lstm_dual_4

VPPM-LSTM-Dual-4 (visible/0 + visible/1, **d_embed=4**, → 29-feat MLP) 컨테이너 실행 환경.

projection 통로를 dual 의 16→1 에서 16→4 로 확장. 자세한 동기는 `Sources/vppm/lstm_dual_4/PLAN.md` 참조.

## 사전조건

- `Sources/pipeline_outputs/features/all_features.npz` (baseline 피처 추출 산출물)
- `Sources/pipeline_outputs/experiments/vppm_lstm/cache/crop_stacks_B1.{1..5}.h5` (visible/0 캐시 — `docker/lstm` 으로 빌드)
- `Sources/pipeline_outputs/experiments/vppm_lstm_dual/cache/crop_stacks_v1_B1.{1..5}.h5` (visible/1 캐시 — `docker/lstm_dual` 의 cache_v1 단계로 빌드)

> **캐시는 dual 과 공유합니다 — 재추출 안 함.**

## 실행

```bash
cd docker/lstm_dual_4

# 한 번에 (train → evaluate)
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
LSTM_DUAL_4_PHASE=evaluate docker compose up -d --build
LSTM_DUAL_4_PHASE=train LSTM_DUAL_4_EXTRA=--quick docker compose up -d --build
```

산출물 → `Sources/pipeline_outputs/experiments/vppm_lstm_dual_4/`
