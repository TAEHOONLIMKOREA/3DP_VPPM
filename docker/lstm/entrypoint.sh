#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Sample-LSTM v2 container starting..."

VENV_PY=/workspace/venv/bin/python
if [ ! -x "$VENV_PY" ]; then
  echo "[entrypoint] FATAL: $VENV_PY 없음 — 호스트 venv 가 마운트되지 않았습니다" >&2
  exit 1
fi

# Phase 별 검증
PHASE="${LSTM_PHASE:-unknown}"
case "$PHASE" in
  cache)
    if [ ! -d /workspace/ORNL_Data_Origin ]; then
      echo "[entrypoint] FATAL: ORNL_Data_Origin/ 가 ro 마운트되지 않음" >&2
      exit 1
    fi
    if [ ! -d /workspace/Sources/pipeline_outputs/sample_stacks ]; then
      echo "[entrypoint] FATAL: sample_stacks/ rw 마운트 필요" >&2
      exit 1
    fi
    ;;
  train|extract)
    if [ ! -f /workspace/Sources/pipeline_outputs/sample_stacks/normalization.json ]; then
      echo "[entrypoint] FATAL: 캐시 (Phase L1) 가 먼저 완료되어야 합니다" >&2
      exit 1
    fi
    if [ -z "${LSTM_MODE:-}" ]; then
      echo "[entrypoint] FATAL: LSTM_MODE (fwd|bidir) 환경변수 필요" >&2
      exit 1
    fi
    ;;
  vppm)
    if [ -z "${LSTM_MODE:-}" ]; then
      echo "[entrypoint] FATAL: LSTM_MODE (fwd|bidir) 환경변수 필요" >&2
      exit 1
    fi
    NPZ="/workspace/Sources/pipeline_outputs/features/all_features_with_lstm_${LSTM_MODE}.npz"
    if [ ! -f "$NPZ" ]; then
      echo "[entrypoint] FATAL: $NPZ 없음 — extract (Phase L3 --mode $LSTM_MODE) 먼저 실행" >&2
      exit 1
    fi
    ;;
esac

# torch / cuda 상태
$VENV_PY - <<'PY'
import torch, sys
print(f"[entrypoint] python={sys.version.split()[0]}  torch={torch.__version__}  "
      f"cuda_avail={torch.cuda.is_available()}  "
      f"devices={torch.cuda.device_count()}")
PY

echo "[entrypoint] phase=$PHASE  launching: $*"
exec "$@"
