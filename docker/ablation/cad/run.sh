#!/usr/bin/env bash
# E3 (No-CAD) ablation 실험을 도커로 실행.
# Usage:
#   ./run.sh              # 전체 학습
#   ./run.sh --quick      # smoke test
set -euo pipefail
cd "$(dirname "$0")"

EXTRA=""
if [[ "${1:-}" == "--quick" ]]; then
  EXTRA="--quick"
fi

export UID_GID="$(id -u):$(id -g)"
export ABLATION_EXTRA="$EXTRA"

DC=(docker compose)
if ! docker info >/dev/null 2>&1; then
  echo "[run.sh] docker 데몬 접근 불가 → sudo -E 로 재시도"
  DC=(sudo -E docker compose)
fi

"${DC[@]}" build
"${DC[@]}" run --rm ablation-cad
