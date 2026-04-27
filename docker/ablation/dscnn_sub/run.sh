#!/usr/bin/env bash
# 단일 DSCNN 서브 실험(E5~E12, E23, E24)을 도커로 실행.
#
# Usage:
#   ./run.sh E7                   # GPU 0 기본 (recoater_streaking)
#   ./run.sh E12 --gpu 2          # GPU 2 지정 (excessive_melting)
#   ./run.sh E23 --gpu 0 --quick  # smoke test (defects_all)
#
# 지원 실험:
#   E5  — No-Powder
#   E6  — No-Printed
#   E7  — No-RecoaterStreaking
#   E8  — No-EdgeSwelling
#   E9  — No-Debris
#   E10 — No-SuperElevation
#   E11 — No-Soot
#   E12 — No-ExcessiveMelting
#   E23 — No-DefectsAll (6 채널 묶음)
#   E24 — No-DSCNNNormal (2 채널 묶음)
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <EXPERIMENT_ID> [--gpu N] [--quick]" >&2
  echo "  EXPERIMENT_ID: E5 ~ E12, E23, E24" >&2
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
  E5|E6|E7|E8|E9|E10|E11|E12|E23|E24) ;;
  *) echo "[run.sh] EXPERIMENT_ID 는 E5~E12, E23, E24 여야 합니다 (입력: $EXP)" >&2; exit 1 ;;
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
"${DC[@]}" run --rm ablation-dscnn-sub
