#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] VPPM-LSTM container starting..."

# 필수 마운트 포인트 검증
for p in \
    /workspace/Sources/pipeline_outputs/features \
    /workspace/Sources/pipeline_outputs/lstm_embeddings \
    /workspace/Sources/pipeline_outputs/models_lstm \
    /workspace/Sources/pipeline_outputs/results/vppm_lstm ; do
  if [ ! -d "$p" ]; then
    echo "[entrypoint] FATAL: $p 가 마운트되지 않음" >&2
    exit 1
  fi
done

# 필수 재사용 파일
if [ ! -f /workspace/Sources/pipeline_outputs/features/all_features.npz ]; then
  echo "[entrypoint] FATAL: all_features.npz 없음 — baseline features 추출을 먼저 해야 합니다" >&2
  exit 1
fi

# /tmp/image_stacks 생성 (tmpfs)
mkdir -p "${LSTM_CACHE_DIR:-/tmp/image_stacks}"

# GPU / torch 상태 로그
python - <<'PY'
import torch, sys
print(f"[entrypoint] python={sys.version.split()[0]}  torch={torch.__version__}  "
      f"cuda_avail={torch.cuda.is_available()}  "
      f"devices={torch.cuda.device_count()}")
PY

echo "[entrypoint] launching: $*"
exec "$@"
