#!/usr/bin/env bash
# E13 (DSCNN+Sensor 조합) ablation 도커 실행.
#
# Usage:
#   ./run.sh           # 전체 학습 (GPU 0 기본)
#   ./run.sh --gpu 2   # GPU 2
#   ./run.sh --quick   # smoke test
set -euo pipefail
cd "$(dirname "$0")"

GPU="0"
EXTRA=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)   GPU="$2"; shift 2 ;;
    --quick) EXTRA="--quick"; shift ;;
    *)       echo "[run.sh] 알 수 없는 옵션: $1" >&2; exit 1 ;;
  esac
done

export UID_GID="$(id -u):$(id -g)"
export NVIDIA_VISIBLE_DEVICES="$GPU"
export ABLATION_EXTRA="$EXTRA"

DC=(docker compose)
if ! docker info >/dev/null 2>&1; then
  echo "[run.sh] docker 데몬 접근 불가 → sudo -E 로 재시도"
  DC=(sudo -E docker compose)
fi

echo "[run.sh] E13 on GPU $GPU  extra='$EXTRA'"
"${DC[@]}" build
"${DC[@]}" run --rm ablation-combined
