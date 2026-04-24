#!/usr/bin/env bash
# 단일 센서 서브 실험을 도커로 실행.
#
# Usage:
#   ./run.sh E17                  # E17 (No-Oxygen) 를 GPU 0 에서 실행
#   ./run.sh E17 --gpu 2          # GPU 2 에서 실행
#   ./run.sh E17 --gpu 0 --quick  # smoke test (20 epoch)
#
# 지원 실험: E14 ~ E22
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <EXPERIMENT_ID> [--gpu N] [--quick]" >&2
  echo "  EXPERIMENT_ID: E14 ~ E22" >&2
  exit 1
fi

EXP="$1"; shift
GPU="0"
EXTRA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)   GPU="$2"; shift 2 ;;
    --quick) EXTRA="--quick"; shift ;;
    *)       echo "[run.sh] 알 수 없는 옵션: $1" >&2; exit 1 ;;
  esac
done

case "$EXP" in
  E14|E15|E16|E17|E18|E19|E20|E21|E22) ;;
  *) echo "[run.sh] EXPERIMENT_ID 는 E14 ~ E22 여야 합니다 (입력: $EXP)" >&2; exit 1 ;;
esac

export UID_GID="$(id -u):$(id -g)"
export EXPERIMENT_ID="$EXP"
export NVIDIA_VISIBLE_DEVICES="$GPU"
export ABLATION_EXTRA="$EXTRA"

DC=(docker compose)
if ! docker info >/dev/null 2>&1; then
  echo "[run.sh] docker 데몬 접근 불가 → sudo -E 로 재시도"
  DC=(sudo -E docker compose)
fi

echo "[run.sh] $EXP on GPU $GPU  extra='$EXTRA'"
"${DC[@]}" build
"${DC[@]}" run --rm ablation-sensor-sub
