#!/usr/bin/env bash
# Phase 5 복구 — E1~E12 까지 완료된 상태에서 남은 15개 (E13, E23, E24, E14~E22, E31~E33) 실행.
#
# Batch A (3-GPU 동시): E13 (combined) + E23/E24 (dscnn_sub categories)
# Batch B (4-GPU × 3): E14~E22 sensor_sub (run_all.sh)
# Batch C (3-GPU × 1): E31~E33 scan_sub (run_all.sh)
# 그 후: --rebuild-summary 로 27 실험 통합 summary.md 갱신
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCKER_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/tmp/v2_resume_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[resume] 로그: $LOG_DIR"

DC=(docker compose)
if ! docker info >/dev/null 2>&1; then
  DC=(sudo -E docker compose)
fi
export UID_GID="$(id -u):$(id -g)"
export ABLATION_EXTRA="--skip-summary"

# ── Batch A — E13 + E23 + E24 동시 (각각 GPU 0/1/2) ─────────
echo ""
echo "============================================================"
echo "  Batch A — E13(combined) + E23/E24(dscnn_sub) 병렬"
echo "============================================================"

# E13: combined (GPU 0)
(
  cd "$DOCKER_DIR/ablation/combined"
  EXPERIMENT_ID=E13 NVIDIA_VISIBLE_DEVICES=0 \
    "${DC[@]}" build > "$LOG_DIR/E13.build.log" 2>&1
  EXPERIMENT_ID=E13 NVIDIA_VISIBLE_DEVICES=0 \
    "${DC[@]}" run --rm ablation-combined > "$LOG_DIR/E13.log" 2>&1
) &
PID_E13=$!
echo "[resume] E13 PID=$PID_E13 → $LOG_DIR/E13.log"

# E23, E24: dscnn_sub (GPU 1, 2)
(
  cd "$DOCKER_DIR/ablation/dscnn_sub"
  EXPERIMENT_ID=E23 NVIDIA_VISIBLE_DEVICES=1 \
    "${DC[@]}" run --rm ablation-dscnn-sub > "$LOG_DIR/E23.log" 2>&1
) &
PID_E23=$!
echo "[resume] E23 PID=$PID_E23 → $LOG_DIR/E23.log"

(
  cd "$DOCKER_DIR/ablation/dscnn_sub"
  EXPERIMENT_ID=E24 NVIDIA_VISIBLE_DEVICES=2 \
    "${DC[@]}" run --rm ablation-dscnn-sub > "$LOG_DIR/E24.log" 2>&1
) &
PID_E24=$!
echo "[resume] E24 PID=$PID_E24 → $LOG_DIR/E24.log"

fail=0
for tag in E13:$PID_E13 E23:$PID_E23 E24:$PID_E24; do
  exp="${tag%%:*}"; pid="${tag##*:}"
  if wait "$pid"; then
    echo "[resume] ✓ $exp OK"
  else
    echo "[resume] ✗ $exp FAILED — $LOG_DIR/$exp.log 확인" >&2
    fail=1
  fi
done
[[ $fail -eq 0 ]] || echo "[resume] WARN: Batch A 일부 실패 — 계속 진행"

# ── Batch B — E14~E22 sensor_sub ────────────────────────────
echo ""
echo "============================================================"
echo "  Batch B — E14~E22 sensor_sub"
echo "============================================================"
(cd "$DOCKER_DIR/ablation/sensor_sub" && ./run_all.sh) 2>&1 | \
  tee "$LOG_DIR/sensor_sub.log" || echo "[resume] WARN: sensor_sub 일부 실패"

# ── Batch C — E31~E33 scan_sub ──────────────────────────────
echo ""
echo "============================================================"
echo "  Batch C — E31~E33 scan_sub"
echo "============================================================"
(cd "$DOCKER_DIR/ablation/scan_sub" && ./run_all.sh) 2>&1 | \
  tee "$LOG_DIR/scan_sub.log" || echo "[resume] WARN: scan_sub 일부 실패"

# ── Phase 6 — 통합 summary 재생성 ──────────────────────────
echo ""
echo "============================================================"
echo "  통합 summary.md 재생성 (27 실험)"
echo "============================================================"
cd "$PROJECT_ROOT"
"$PROJECT_ROOT/venv/bin/python" -m Sources.vppm.ablation.run --rebuild-summary 2>&1 | \
  tee "$LOG_DIR/summary.log"

echo ""
echo "============================================================"
echo "  Resume 완료 — 로그: $LOG_DIR"
echo "============================================================"
