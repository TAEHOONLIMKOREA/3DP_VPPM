#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] VPPM Ablation container starting..."

VENV_PY=/workspace/venv/bin/python
if [ ! -x "$VENV_PY" ]; then
  echo "[entrypoint] FATAL: $VENV_PY 없음 — 호스트 venv 가 마운트되지 않았습니다" >&2
  exit 1
fi

# 필수 입력 검증
if [ ! -f /workspace/Sources/pipeline_outputs/features/all_features.npz ]; then
  echo "[entrypoint] FATAL: all_features.npz 없음 — 먼저 baseline 피처 추출(run_pipeline --phase features)을 마쳐야 합니다" >&2
  exit 1
fi

# 출력 마운트 검증
if [ ! -d /workspace/Sources/pipeline_outputs/ablation ]; then
  echo "[entrypoint] FATAL: /workspace/Sources/pipeline_outputs/ablation 가 마운트되지 않음 (rw 필요)" >&2
  exit 1
fi

# 쓰기 권한 확인 (UID mismatch 시 조기 실패)
if ! touch /workspace/Sources/pipeline_outputs/ablation/.write_test 2>/dev/null; then
  echo "[entrypoint] FATAL: ablation 볼륨에 쓰기 실패 — UID/권한을 확인하세요" >&2
  exit 1
fi
rm -f /workspace/Sources/pipeline_outputs/ablation/.write_test

# GPU / torch 상태 로그
$VENV_PY - <<'PY'
import torch, sys
print(f"[entrypoint] python={sys.version.split()[0]}  torch={torch.__version__}  "
      f"cuda_avail={torch.cuda.is_available()}  "
      f"devices={torch.cuda.device_count()}")
PY

echo "[entrypoint] launching: $*"
exec "$@"
