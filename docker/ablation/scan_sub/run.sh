#!/usr/bin/env bash
# 단일 스캔 서브 실험(E31~E33)을 도커로 실행.
#
# Usage:
#   ./run.sh E32                  # GPU 0 기본
#   ./run.sh E32 --gpu 1          # GPU 1 지정
#   ./run.sh E32 --gpu 0 --quick  # smoke test
#
# 전제: scan_features.py 구현 + all_features.npz v2 재추출 완료 (README 참조).
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <EXPERIMENT_ID> [--gpu N] [--quick]" >&2
  echo "  EXPERIMENT_ID: E31 ~ E33" >&2
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
  E31|E32|E33) ;;
  *) echo "[run.sh] EXPERIMENT_ID 는 E31 ~ E33 여야 합니다 (입력: $EXP)" >&2; exit 1 ;;
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
"${DC[@]}" run --rm ablation-scan-sub
