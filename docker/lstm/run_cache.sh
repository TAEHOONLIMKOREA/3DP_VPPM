#!/usr/bin/env bash
# Phase L1 — sample_stacks/ 캐시 빌드.
# Usage:
#   ./run_cache.sh                     # 5 빌드 모두 (~30분)
#   ./run_cache.sh --builds B1.2       # 특정 빌드만
set -euo pipefail
cd "$(dirname "$0")"

EXTRA="${*:-}"
export UID_GID="$(id -u):$(id -g)"
export LSTM_EXTRA="$EXTRA"

# 산출물 폴더 사전 생성
mkdir -p ../../Sources/pipeline_outputs/sample_stacks

DC=(docker compose)
if ! docker info >/dev/null 2>&1; then
  echo "[run_cache] docker daemon 접근 불가 → sudo -E 로 재시도"
  DC=(sudo -E docker compose)
fi

echo "[run_cache] building image..."
"${DC[@]}" build lstm-cache
echo "[run_cache] launching cache builder (extra='$EXTRA')..."
"${DC[@]}" run --rm lstm-cache
echo "[run_cache] ✓ done."
