#!/usr/bin/env bash
# Phase L4 — VPPM 재학습 (22 또는 37 차원, mode 에 따름).
# Usage:
#   ./run_vppm.sh --mode {fwd1|bidir1|fwd16|bidir16}
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
  *) echo "[run_vppm] FATAL: --mode {fwd1|bidir1|fwd16|bidir16} 필수 (입력: '$MODE')" >&2; exit 1 ;;
esac

export UID_GID="$(id -u):$(id -g)"
export NVIDIA_VISIBLE_DEVICES="$GPU"
export LSTM_MODE="$MODE"
export LSTM_EXTRA="$(echo "$EXTRA" | xargs || true)"

mkdir -p ../../Sources/pipeline_outputs/results/vppm_lstm_"$MODE"
mkdir -p ../../Sources/pipeline_outputs/models_lstm/"$MODE"

DC=(docker compose)
if ! docker info >/dev/null 2>&1; then
  echo "[run_vppm] docker daemon 접근 불가 → sudo -E 로 재시도"
  DC=(sudo -E docker compose)
fi

echo "[run_vppm] mode=$MODE  GPU=$GPU"
"${DC[@]}" build lstm-vppm
"${DC[@]}" run --rm lstm-vppm
echo "[run_vppm] ✓ done. → results/vppm_lstm_$MODE/"
