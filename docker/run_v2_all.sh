#!/usr/bin/env bash
# V2 Full Re-run 마스터 오케스트레이터 — Phase 4 + 5 자동 실행.
#
# 흐름:
#   0) 사전 검증 (Phase 3 완료 / v2 피처 / GPU 4장 / 도커)
#   1) Phase 4: Baseline v2 학습 (GPU 0, ~30 min)
#   2) Phase 5: 27개 ablation 병렬 실행
#         - E1~E4 (4-GPU 병렬, ~30 min)
#         - E5~E12 + E23 + E24 (4-GPU × 3 배치, ~45 min)
#         - E13 단독 (1 GPU, ~30 min — 다른 그룹과 동시 가능)
#         - E14~E22 (4-GPU × 3 배치, ~45 min)
#         - E31~E33 (3-GPU 병렬, ~30 min)
#   3) summary.md 통합 재생성
#
# 총 예상 시간: ~3 ~ 4 시간 (4-GPU)
#
# Usage:
#   ./run_v2_all.sh                 # 전체 자동 실행
#   ./run_v2_all.sh --skip-baseline # Phase 4 건너뛰고 Phase 5 만
#   ./run_v2_all.sh --quick         # 모든 단계 smoke test (~20 min)
set -euo pipefail
cd "$(dirname "$0")"

SKIP_BASELINE=0
QUICK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-baseline) SKIP_BASELINE=1; shift ;;
    --quick)         QUICK="--quick";  shift ;;
    *) echo "[v2_all] 알 수 없는 옵션: $1" >&2; exit 1 ;;
  esac
done

PROJECT_ROOT="$(cd .. && pwd)"
FEATURES_NPZ="$PROJECT_ROOT/Sources/pipeline_outputs/features/all_features.npz"
VENV_PY="$PROJECT_ROOT/venv/bin/python"

# ── 0) 사전 검증 ────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  V2 Full Re-run — 사전 검증"
echo "============================================================"

# 0a. 진행 중인 Phase 3 (run_pipeline --phase features) 가 있으면 대기.
#    - bash wrapper / pgrep 자체 / 셸 자식 을 걸러내고 실제 python 프로세스만 찾는다.
#    - 형태: `<py> -m Sources.vppm.run_pipeline --phase features`
EXTRACT_PID=$(pgrep -af "run_pipeline.*--phase features" 2>/dev/null \
              | grep -v "/bin/bash" \
              | grep -v "pgrep" \
              | awk '{print $1}' \
              | head -1 || true)
if [[ -n "$EXTRACT_PID" ]]; then
  echo "[v2_all] Phase 3 (features 재추출) 이 진행 중 (python PID=$EXTRACT_PID)"
  echo "[v2_all] 완료까지 대기 중..."
  while kill -0 "$EXTRACT_PID" 2>/dev/null; do
    sleep 60
    echo "  [$(date '+%H:%M:%S')] PID $EXTRACT_PID 아직 실행 중 — 60초 후 재확인"
  done
  echo "[v2_all] Phase 3 종료 감지 (PID $EXTRACT_PID 종료). 결과 검증으로 진행."
  # 파일 시스템 sync 보장 (HDF5 close + npz write flush)
  sync
  sleep 5
fi

# 0b. all_features.npz 가 v2 (실 구현) 인지 확인
if [[ ! -f "$FEATURES_NPZ" ]]; then
  echo "[v2_all] FATAL: $FEATURES_NPZ 가 없습니다." >&2
  echo "         Phase 3 (./venv/bin/python -m Sources.vppm.run_pipeline --phase features) 를 먼저 실행하세요." >&2
  exit 1
fi

if [[ ! -x "$VENV_PY" ]]; then
  echo "[v2_all] FATAL: $VENV_PY 가 실행 가능하지 않습니다." >&2
  exit 1
fi

V2_CHECK=$("$VENV_PY" -c "
import numpy as np
d = np.load('$FEATURES_NPZ')
f19 = d['features'][:, 19]
f20 = d['features'][:, 20]
s19 = float(np.nanstd(f19))
s20 = float(np.nanstd(f20))
n_voxels = len(f19)
n_zero_19 = int((f19 == 0).sum())
print(f'std_19={s19:.4f} std_20={s20:.4f} n={n_voxels} zero_19={n_zero_19}')
if s19 < 0.001 and s20 < 0.001:
    print('VERDICT: placeholder')
else:
    print('VERDICT: real')
")
echo "[v2_all] features 검증: $V2_CHECK"
if echo "$V2_CHECK" | grep -q "VERDICT: placeholder"; then
  echo "[v2_all] FATAL: all_features.npz 의 scan 피처(#19, #20) 가 0 상수입니다." >&2
  echo "         scan_features.py 적용 후 features 재추출이 필요합니다." >&2
  exit 1
fi
echo "[v2_all] ✓ scan 피처 실 구현 확인됨"

# 0c. GPU 4장 확인
N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [[ $N_GPUS -lt 4 ]]; then
  echo "[v2_all] WARN: GPU $N_GPUS 장 — 일부 병렬 실행 불가. 직렬 fallback 으로 진행."
fi

# 0d. 도커 데몬 접근
if ! docker info >/dev/null 2>&1; then
  if ! sudo -n docker info >/dev/null 2>&1; then
    echo "[v2_all] FATAL: docker 데몬 접근 불가 (sudo 도 비밀번호 요구)." >&2
    exit 1
  fi
  echo "[v2_all] docker 는 sudo -E 로 호출됩니다."
fi

# 로그 디렉터리
LOG_DIR="/tmp/v2_full_rerun_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[v2_all] 로그 디렉터리: $LOG_DIR"

# ── 1) Phase 4 — Baseline v2 ─────────────────────────────────
if [[ $SKIP_BASELINE -eq 0 ]]; then
  echo ""
  echo "============================================================"
  echo "  Phase 4 — Baseline v2 학습 (GPU 0)"
  echo "============================================================"
  cd "$(dirname "$0")/baseline"
  if ! ./run.sh --gpu 0 2>&1 | tee "$LOG_DIR/phase4_baseline.log"; then
    echo "[v2_all] FATAL: Phase 4 실패 — 로그: $LOG_DIR/phase4_baseline.log" >&2
    exit 1
  fi
  cd "$(dirname "$0")"
  echo "[v2_all] ✓ Phase 4 완료"
else
  echo "[v2_all] --skip-baseline: Phase 4 건너뜀"
fi

# Ablation 실행 시 기존 결과 정리 (백업 없이 삭제 — 덮어쓰기 방지)
ABL_DIR="$PROJECT_ROOT/Sources/pipeline_outputs/ablation"
if [[ -d "$ABL_DIR" ]] && ls "$ABL_DIR"/E*_no_*/ >/dev/null 2>&1; then
  echo "[v2_all] 기존 ablation 결과 정리 (덮어쓰기 방지)"
  rm -rf "$ABL_DIR"/E*_no_*
fi
mkdir -p "$ABL_DIR"

# ── 2) Phase 5 — 모든 ablation 병렬 실행 ──────────────────────
echo ""
echo "============================================================"
echo "  Phase 5 — 27 개 ablation 재실행"
echo "============================================================"

run_group() {
  local label="$1"; local cmd="$2"
  echo ""
  echo "[v2_all] === $label ==="
  if eval "$cmd" 2>&1 | tee "$LOG_DIR/phase5_${label// /_}.log"; then
    echo "[v2_all] ✓ $label OK"
  else
    echo "[v2_all] ✗ $label FAILED — $LOG_DIR/phase5_${label// /_}.log 확인" >&2
    return 1
  fi
}

cd "$(dirname "$0")"

# E1~E4 (4-GPU 병렬 1배치)
run_group "E1-E4 main groups" \
  "(cd ablation && ./run_all.sh ${QUICK:-})" || true

# E5~E12 + E23/24 (4-GPU × 3배치)
run_group "E5-E12 + E23/24 dscnn sub" \
  "(cd ablation/dscnn_sub && ./run_all.sh ${QUICK:-})" || true

# E13 (단독, GPU 0 사용 — 다른 그룹 끝난 후 안전하게 실행)
run_group "E13 combined" \
  "(cd ablation/combined && ./run.sh --gpu 0 ${QUICK:-})" || true

# E14~E22 (4-GPU × 3배치)
run_group "E14-E22 sensor sub" \
  "(cd ablation/sensor_sub && ./run_all.sh ${QUICK:-})" || true

# E31~E33 (3-GPU 병렬)
run_group "E31-E33 scan sub" \
  "(cd ablation/scan_sub && ./run_all.sh ${QUICK:-})" || true

# ── 3) Summary 통합 재생성 ───────────────────────────────────
echo ""
echo "============================================================"
echo "  Phase 6 — summary.md 통합 재생성"
echo "============================================================"
cd "$PROJECT_ROOT"
if "$VENV_PY" -m Sources.vppm.ablation.run --rebuild-summary 2>&1 | tee "$LOG_DIR/phase6_summary.log"; then
  echo "[v2_all] ✓ summary.md 갱신 완료"
else
  echo "[v2_all] WARN: summary.md 재생성 실패 — $LOG_DIR/phase6_summary.log 확인"
fi

echo ""
echo "============================================================"
echo "  V2 Full Re-run 완료"
echo "============================================================"
echo "  로그: $LOG_DIR"
echo "  결과:"
echo "    - Sources/pipeline_outputs/results/vppm_origin/   (baseline v2)"
echo "    - Sources/pipeline_outputs/ablation/E*_no_*/      (27 ablation)"
echo "    - Sources/pipeline_outputs/ablation/summary.md    (통합 요약)"
echo ""
echo "  다음 단계: FULL_REPORT.md 수동 갱신 (Phase 6)."
