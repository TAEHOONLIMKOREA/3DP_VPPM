#!/usr/bin/env bash
# Baseline VPPM 학습 — 도커.
#
# Usage:
#   ./run.sh             # GPU 0 기본
#   ./run.sh --gpu 2     # GPU 지정
#
# 전제: features/all_features.npz 존재.
set -euo pipefail
cd "$(dirname "$0")"

GPU="0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    *)     echo "[run.sh] 알 수 없는 옵션: $1" >&2; exit 1 ;;
  esac
done

export UID_GID="$(id -u):$(id -g)"
export NVIDIA_VISIBLE_DEVICES="$GPU"

DC=(docker compose)
if ! docker info >/dev/null 2>&1; then
  echo "[run.sh] docker 데몬 접근 불가 → sudo -E 로 재시도"
  DC=(sudo -E docker compose)
fi

echo "[run.sh] Baseline train+evaluate on GPU $GPU"
"${DC[@]}" build
"${DC[@]}" run --rm baseline
echo "[run.sh] Done."
