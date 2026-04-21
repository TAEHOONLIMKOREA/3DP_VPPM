#!/usr/bin/env bash
# 4개 그룹을 각자 다른 GPU 에서 병렬로 실행.
#
# GPU 배정 (compose env NVIDIA_VISIBLE_DEVICES):
#   dscnn  → GPU 0   (E1 No-DSCNN)
#   sensor → GPU 1   (E2 No-Sensor)
#   cad    → GPU 2   (E3 No-CAD)
#   scan   → GPU 3   (E4 No-Scan)
#
# 흐름:
#   1) 공용 이미지 vppm-ablation:gpu 을 한 번만 빌드
#   2) 4 컨테이너를 백그라운드로 동시에 띄움 (summary.md race 방지를 위해 각자 --skip-summary)
#   3) 모두 완료되면 호스트 venv 로 --rebuild-summary 호출 → ablation/summary.md 통합 재생성
#
# Usage:
#   ./run_all.sh           # 전체 학습
#   ./run_all.sh --quick   # smoke test
set -euo pipefail
cd "$(dirname "$0")"

EXTRA=""
if [[ "${1:-}" == "--quick" ]]; then
  EXTRA="--quick"
fi

export UID_GID="$(id -u):$(id -g)"
# 컨테이너 안의 run.py 에 --skip-summary 를 붙여 race 방지
export ABLATION_EXTRA="--skip-summary ${EXTRA}"

DC=(docker compose)
if ! docker info >/dev/null 2>&1; then
  echo "[run_all] docker 데몬 접근 불가 → sudo -E 로 재시도"
  DC=(sudo -E docker compose)
fi

# ── 1) 공용 이미지 1회 빌드 ────────────────────────────────
echo "[run_all] building shared image vppm-ablation:gpu ..."
( cd dscnn && "${DC[@]}" build )

# ── 2) 4 컨테이너 병렬 실행 ────────────────────────────────
LOG_DIR="/tmp/ablation_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[run_all] logs: $LOG_DIR"

declare -A pids
for grp in dscnn sensor cad scan; do
  echo "[run_all] launching $grp (GPU 자동 핀 by compose)"
  (
    cd "$grp" && "${DC[@]}" run --rm "ablation-$grp"
  ) > "$LOG_DIR/${grp}.log" 2>&1 &
  pids[$grp]=$!
  echo "[run_all]   PID=${pids[$grp]}  log=$LOG_DIR/${grp}.log"
done

# ── 3) 대기 + 상태 점검 ────────────────────────────────────
fail=0
for grp in dscnn sensor cad scan; do
  if wait "${pids[$grp]}"; then
    echo "[run_all] ✓ $grp OK"
  else
    echo "[run_all] ✗ $grp FAILED — $LOG_DIR/${grp}.log 확인"
    fail=1
  fi
done

if [ $fail -ne 0 ]; then
  echo "[run_all] 일부 그룹 실패 — summary 재생성을 건너뜁니다."
  exit 1
fi

# ── 4) summary.md 통합 재생성 (호스트 venv) ────────────────
echo ""
echo "[run_all] 모든 컨테이너 완료 — summary.md 재생성"
cd ../..
if [ -x ./venv/bin/python ]; then
  ./venv/bin/python -m Sources.vppm.ablation.run --rebuild-summary
else
  echo "[run_all] venv 없음 — summary 재생성 건너뜀. 컨테이너로 재생성하려면:"
  echo "    (cd docker/ablation/scan && sudo -E docker compose run --rm ablation-scan /workspace/venv/bin/python -m Sources.vppm.ablation.run --rebuild-summary)"
fi

echo "[run_all] done."
