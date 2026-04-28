#!/usr/bin/env bash
# Phase L3 — 임베딩 추출 + 21→{22,37} npz 통합.
# Usage:
#   ./run_extract.sh --mode {fwd1|bidir1|fwd16|bidir16}
set -euo pipefail
cd "$(dirname "$0")"

MODE=""
GPU="0"
EXTRA=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --gpu)  GPU="$2"; shift 2 ;;
    *) EXTRA="$EXTRA $1"; shift ;;
  esac
done

case "$MODE" in
  fwd1|bidir1|fwd16|bidir16) ;;
  *) echo "[run_extract] FATAL: --mode {fwd1|bidir1|fwd16|bidir16} 필수 (입력: '$MODE')" >&2; exit 1 ;;
esac

export UID_GID="$(id -u):$(id -g)"
export NVIDIA_VISIBLE_DEVICES="$GPU"
export LSTM_MODE="$MODE"
export LSTM_EXTRA="$(echo "$EXTRA" | xargs || true)"

mkdir -p ../../Sources/pipeline_outputs/lstm_embeddings/"$MODE"

DC=(docker compose)
if ! docker info >/dev/null 2>&1; then
  echo "[run_extract] docker daemon 접근 불가 → sudo -E 로 재시도"
  DC=(sudo -E docker compose)
fi

echo "[run_extract] mode=$MODE  GPU=$GPU"
"${DC[@]}" build lstm-extract
"${DC[@]}" run --rm lstm-extract
echo "[run_extract] ✓ done."
