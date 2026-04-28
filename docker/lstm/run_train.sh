#!/usr/bin/env bash
# Phase L2 — Sample-LSTM 5-Fold CV 학습.
# Usage:
#   ./run_train.sh --mode fwd1                # forward, d_embed=1 (22 차원)
#   ./run_train.sh --mode bidir1              # bidirectional, d_embed=1 (22 차원)
#   ./run_train.sh --mode fwd16               # forward, d_embed=16 (37 차원)
#   ./run_train.sh --mode bidir16             # bidirectional, d_embed=16 (37 차원, 이전 설계)
#   ./run_train.sh --mode bidir1 --gpu 2 --quick
#   ./run_train.sh --mode fwd1 --folds 0 1
set -euo pipefail
cd "$(dirname "$0")"

MODE=""
GPU="0"
EXTRA=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)  MODE="$2"; shift 2 ;;
    --gpu)   GPU="$2"; shift 2 ;;
    --quick) EXTRA="$EXTRA --quick"; shift ;;
    --folds)
      shift
      EXTRA="$EXTRA --folds"
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        EXTRA="$EXTRA $1"; shift
      done
      ;;
    *) EXTRA="$EXTRA $1"; shift ;;
  esac
done

case "$MODE" in
  fwd1|bidir1|fwd16|bidir16) ;;
  *) echo "[run_train] FATAL: --mode {fwd1|bidir1|fwd16|bidir16} 필수 (입력: '$MODE')" >&2; exit 1 ;;
esac

export UID_GID="$(id -u):$(id -g)"
export NVIDIA_VISIBLE_DEVICES="$GPU"
export LSTM_MODE="$MODE"
export LSTM_EXTRA="$(echo "$EXTRA" | xargs || true)"

mkdir -p ../../Sources/pipeline_outputs/models_lstm/"$MODE"

DC=(docker compose)
if ! docker info >/dev/null 2>&1; then
  echo "[run_train] docker daemon 접근 불가 → sudo -E 로 재시도"
  DC=(sudo -E docker compose)
fi

echo "[run_train] mode=$MODE  GPU=$GPU  extra='$LSTM_EXTRA'"
"${DC[@]}" build lstm-train
"${DC[@]}" run --rm lstm-train
echo "[run_train] ✓ done. → models_lstm/$MODE/"
