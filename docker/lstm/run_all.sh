#!/usr/bin/env bash
# Phase L1 → (L2 → L3 → L4) 순차 실행.
# Usage:
#   ./run_all.sh --mode fwd1                # 단일 모드 (~3시간)
#   ./run_all.sh --mode bidir16             # 이전 37 차원 설계
#   ./run_all.sh --mode all                 # 4 모드 모두 (L1 1회, L2~L4 각 모드별 ~12시간)
#   ./run_all.sh --mode all --quick         # smoke test (4 모드 모두)
#   ./run_all.sh --mode bidir16 --skip-cache
set -euo pipefail
cd "$(dirname "$0")"

MODE=""
GPU="0"
QUICK=""
SKIP_CACHE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)        MODE="$2"; shift 2 ;;
    --gpu)         GPU="$2"; shift 2 ;;
    --quick)       QUICK="--quick"; shift ;;
    --skip-cache)  SKIP_CACHE=1; shift ;;
    *) echo "[run_all] unknown opt: $1" >&2; exit 1 ;;
  esac
done

case "$MODE" in
  fwd1|bidir1|fwd16|bidir16|all) ;;
  *) echo "[run_all] FATAL: --mode {fwd1|bidir1|fwd16|bidir16|all} 필수" >&2; exit 1 ;;
esac

LOG_DIR="/tmp/lstm_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[run_all] mode=$MODE  GPU=$GPU  logs=$LOG_DIR"

# ── Phase L1 (mode 무관, 1 회) ──────────────────────────
if [[ $SKIP_CACHE -eq 0 ]]; then
  echo ""
  echo "============================================================"
  echo "  Phase L1 — sample_stacks 캐시"
  echo "============================================================"
  ./run_cache.sh 2>&1 | tee "$LOG_DIR/L1_cache.log"
fi

# ── 모드별 L2~L4 ────────────────────────────────────────
run_mode() {
  local m="$1"
  echo ""
  echo "============================================================"
  echo "  [$m]  Phase L2 — LSTM 5-Fold 학습"
  echo "============================================================"
  ./run_train.sh --mode "$m" --gpu "$GPU" $QUICK 2>&1 | tee "$LOG_DIR/L2_${m}_train.log"

  echo ""
  echo "============================================================"
  echo "  [$m]  Phase L3 — 임베딩 추출 + npz 통합"
  echo "============================================================"
  ./run_extract.sh --mode "$m" --gpu "$GPU" 2>&1 | tee "$LOG_DIR/L3_${m}_extract.log"

  echo ""
  echo "============================================================"
  echo "  [$m]  Phase L4 — VPPM 재학습"
  echo "============================================================"
  ./run_vppm.sh --mode "$m" --gpu "$GPU" 2>&1 | tee "$LOG_DIR/L4_${m}_vppm.log"
}

if [[ "$MODE" == "all" ]]; then
  for m in fwd1 bidir1 fwd16 bidir16; do
    run_mode "$m"
  done
else
  run_mode "$MODE"
fi

echo ""
echo "============================================================"
echo "  Sample-LSTM v2 완료 (mode=$MODE)"
echo "  결과: Sources/pipeline_outputs/results/vppm_lstm_{fwd1,bidir1,fwd16,bidir16}/"
echo "  로그: $LOG_DIR"
echo "============================================================"
