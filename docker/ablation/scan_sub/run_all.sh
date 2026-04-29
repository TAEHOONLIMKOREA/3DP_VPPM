#!/usr/bin/env bash
# 스캔 서브 3개 실험(E31~E33)을 3-GPU 병렬로 단일 배치 실행.
#
# GPU 배정:
#   E31 (No-Scan)            → GPU 0
#   E32 (No-ReturnDelay)     → GPU 1
#   E33 (No-StripeBoundary)  → GPU 2
#
# 각 컨테이너는 --skip-summary 로 실행되어 summary.md race 를 막고, 전체 완료 후
# 호스트 venv 로 --rebuild-summary 를 호출해 통합 summary.md 를 생성한다.
#
# 전제: scan_features.py 구현 + all_features.npz 재추출 완료 (README 참조).
#
# Usage:
#   ./run_all.sh           # 전체 학습 (GPU 3장 병렬, ~30분)
#   ./run_all.sh --quick   # smoke test (~2분)
set -euo pipefail
cd "$(dirname "$0")"

EXTRA=""
if [[ "${1:-}" == "--quick" ]]; then
  EXTRA="--quick"
fi

export UID_GID="$(id -u):$(id -g)"
export ABLATION_EXTRA="--skip-summary ${EXTRA}"

DC=(docker compose)
if ! docker info >/dev/null 2>&1; then
  echo "[run_all] docker 데몬 접근 불가 → sudo -E 로 재시도"
  DC=(sudo -E docker compose)
fi

# ── 1) 공용 이미지 1회 빌드 ────────────────────────────────
echo "[run_all] building shared image vppm-ablation:gpu ..."
EXPERIMENT_ID=E31 NVIDIA_VISIBLE_DEVICES=0 "${DC[@]}" build

LOG_DIR="/tmp/scan_sub_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[run_all] logs: $LOG_DIR"

# ── 2) 3 컨테이너 병렬 실행 ────────────────────────────────
declare -A pids
declare -A GPU_MAP=([E31]=0 [E32]=1 [E33]=2)

for exp in E31 E32 E33; do
  gpu="${GPU_MAP[$exp]}"
  echo "[run_all] launching $exp on GPU $gpu"
  (
    EXPERIMENT_ID="$exp" \
    NVIDIA_VISIBLE_DEVICES="$gpu" \
    "${DC[@]}" run --rm ablation-scan-sub
  ) > "$LOG_DIR/${exp}.log" 2>&1 &
  pids[$exp]=$!
  echo "[run_all]   PID=${pids[$exp]}  log=$LOG_DIR/${exp}.log"
done

# ── 3) 대기 + 상태 점검 ────────────────────────────────────
fail=0
for exp in E31 E32 E33; do
  if wait "${pids[$exp]}"; then
    echo "[run_all] ✓ $exp OK"
  else
    echo "[run_all] ✗ $exp FAILED — $LOG_DIR/${exp}.log 확인"
    fail=1
  fi
done

if [[ $fail -ne 0 ]]; then
  echo "[run_all] 일부 실험 실패 — summary 재생성을 건너뜁니다."
  exit 1
fi

# ── 4) summary.md 통합 재생성 ────────────────────────────
echo ""
echo "[run_all] 모든 컨테이너 완료 — summary.md 재생성"
cd ../../..
if [ -x ./venv/bin/python ]; then
  ./venv/bin/python -m Sources.vppm.ablation.run --rebuild-summary
else
  echo "[run_all] venv 없음 — summary 재생성 건너뜀. 컨테이너로 하려면:"
  echo "    (cd docker/ablation/scan_sub && \\"
  echo "      EXPERIMENT_ID=E31 NVIDIA_VISIBLE_DEVICES=0 sudo -E docker compose run --rm \\"
  echo "      ablation-scan-sub /workspace/venv/bin/python -m Sources.vppm.ablation.run --rebuild-summary)"
fi

echo ""
echo "[run_all] done. 로그: $LOG_DIR"
