#!/usr/bin/env bash
# Ablation Full Run — 모든 ablation 실험 (총 27개) 자동 실행 + summary 통합 재생성.
#
# 실행 그룹:
#   1) E1~E4   main groups        (4-GPU 병렬,        ~30 min)
#   2) E5~E12 + E23/24  dscnn_sub (4-GPU × 3 배치,    ~45 min)
#   3) E13     combined           (1 GPU,             ~30 min)
#   4) E14~E22 sensor_sub         (4-GPU × 3 배치,    ~45 min)
#   5) E31~E33 scan_sub           (3-GPU 병렬,        ~30 min)
#   6) summary.md 통합 재생성
#
# 총 예상 시간: ~3 시간 (4-GPU)
#
# 전제: features/all_features.npz 존재 (Phase 3 완료).
#
# Usage:
#   ./run_all.sh           # 전체 학습
#   ./run_all.sh --quick   # smoke test
set -euo pipefail
cd "$(dirname "$0")"

QUICK=""
if [[ "${1:-}" == "--quick" ]]; then
  QUICK="--quick"
fi

PROJECT_ROOT="$(cd ../.. && pwd)"
FEATURES_NPZ="$PROJECT_ROOT/Sources/pipeline_outputs/features/all_features.npz"
VENV_PY="$PROJECT_ROOT/venv/bin/python"

# ── 0) 사전 검증 ────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Ablation Full Run — 사전 검증"
echo "============================================================"

# 0a. 진행 중인 features 재추출 대기 (있을 경우)
EXTRACT_PID=$(pgrep -af "run_pipeline.*--phase features" 2>/dev/null \
              | grep -v "/bin/bash" \
              | grep -v "pgrep" \
              | awk '{print $1}' \
              | head -1 || true)
if [[ -n "$EXTRACT_PID" ]]; then
  echo "[ablation] features 재추출 진행 중 (python PID=$EXTRACT_PID) — 완료까지 대기"
  while kill -0 "$EXTRACT_PID" 2>/dev/null; do
    sleep 60
    echo "  [$(date '+%H:%M:%S')] PID $EXTRACT_PID 아직 실행 중 — 60초 후 재확인"
  done
  echo "[ablation] features 추출 종료 감지 — 결과 검증으로 진행"
  sync
  sleep 5
fi

# 0b. all_features.npz 존재 + scan 피처 실 구현 확인
if [[ ! -f "$FEATURES_NPZ" ]]; then
  echo "[ablation] FATAL: $FEATURES_NPZ 가 없습니다." >&2
  echo "          ./venv/bin/python -m Sources.vppm.run_pipeline --phase features 를 먼저 실행하세요." >&2
  exit 1
fi
if [[ ! -x "$VENV_PY" ]]; then
  echo "[ablation] FATAL: $VENV_PY 가 실행 가능하지 않습니다." >&2
  exit 1
fi

CHECK=$("$VENV_PY" -c "
import numpy as np
d = np.load('$FEATURES_NPZ')
s19 = float(np.nanstd(d['features'][:, 19]))
s20 = float(np.nanstd(d['features'][:, 20]))
print(f'std_19={s19:.4f} std_20={s20:.4f}')
if s19 < 0.001 and s20 < 0.001:
    print('VERDICT: placeholder')
else:
    print('VERDICT: real')
")
echo "[ablation] features 검증: $CHECK"
if echo "$CHECK" | grep -q "VERDICT: placeholder"; then
  echo "[ablation] FATAL: scan 피처(#19, #20) 가 0 상수 — features 재추출이 필요합니다." >&2
  exit 1
fi
echo "[ablation] ✓ scan 피처 실 구현 확인됨"

# 0c. GPU 수 확인 (정보용)
N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [[ $N_GPUS -lt 4 ]]; then
  echo "[ablation] WARN: GPU $N_GPUS 장 — 일부 4-GPU 병렬 단계는 직렬 fallback 됨."
fi

# 0d. docker 데몬 접근
DC=(docker compose)
if ! docker info >/dev/null 2>&1; then
  if ! sudo -n docker info >/dev/null 2>&1; then
    echo "[ablation] FATAL: docker 데몬 접근 불가 (sudo 도 비밀번호 요구)." >&2
    exit 1
  fi
  echo "[ablation] docker 는 sudo -E 로 호출됩니다."
  DC=(sudo -E docker compose)
fi

export UID_GID="$(id -u):$(id -g)"
# 컨테이너 안의 run.py 에 --skip-summary 를 붙여 race 방지 (마지막에 일괄 재생성)
export ABLATION_EXTRA="--skip-summary ${QUICK}"

# 기존 ablation 결과 정리 (덮어쓰기 방지)
ABL_DIR="$PROJECT_ROOT/Sources/pipeline_outputs/ablation"
if [[ -d "$ABL_DIR" ]] && ls "$ABL_DIR"/E*_no_*/ >/dev/null 2>&1; then
  echo "[ablation] 기존 ablation 결과 정리 (덮어쓰기 방지)"
  rm -rf "$ABL_DIR"/E*_no_*
fi
mkdir -p "$ABL_DIR"

LOG_DIR="/tmp/ablation_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[ablation] 로그 디렉터리: $LOG_DIR"

run_group() {
  local label="$1"; local cmd="$2"
  echo ""
  echo "[ablation] === $label ==="
  if eval "$cmd" 2>&1 | tee "$LOG_DIR/${label// /_}.log"; then
    echo "[ablation] ✓ $label OK"
  else
    echo "[ablation] ✗ $label FAILED — $LOG_DIR/${label// /_}.log 확인" >&2
    return 1
  fi
}

# ── 1) E1~E4 main groups (4-GPU 병렬) ──────────────────────
echo ""
echo "============================================================"
echo "  E1~E4 main groups (4-GPU 병렬)"
echo "============================================================"

# 공용 이미지 vppm-ablation:gpu 1회 빌드
echo "[ablation] building shared image vppm-ablation:gpu ..."
( cd dscnn && "${DC[@]}" build )

LOG_MAIN="$LOG_DIR/main"
mkdir -p "$LOG_MAIN"
declare -A pids
for grp in dscnn sensor cad scan; do
  echo "[ablation] launching $grp (GPU 자동 핀 by compose)"
  (
    cd "$grp" && "${DC[@]}" run --rm "ablation-$grp"
  ) > "$LOG_MAIN/${grp}.log" 2>&1 &
  pids[$grp]=$!
  echo "[ablation]   PID=${pids[$grp]}  log=$LOG_MAIN/${grp}.log"
done

main_fail=0
for grp in dscnn sensor cad scan; do
  if wait "${pids[$grp]}"; then
    echo "[ablation] ✓ main/$grp OK"
  else
    echo "[ablation] ✗ main/$grp FAILED — $LOG_MAIN/${grp}.log 확인" >&2
    main_fail=1
  fi
done
[[ $main_fail -eq 0 ]] || echo "[ablation] WARN: main groups 일부 실패 — 다음 단계로 계속 진행"

# ── 2) E5~E12 + E23/24 dscnn_sub ──────────────────────────
run_group "E5-E12 + E23/24 dscnn_sub" \
  "(cd dscnn_sub && ./run_all.sh ${QUICK})" || true

# ── 3) E13 combined ───────────────────────────────────────
run_group "E13 combined" \
  "(cd combined && ./run.sh --gpu 0 ${QUICK})" || true

# ── 4) E14~E22 sensor_sub ─────────────────────────────────
run_group "E14-E22 sensor_sub" \
  "(cd sensor_sub && ./run_all.sh ${QUICK})" || true

# ── 5) E31~E33 scan_sub ───────────────────────────────────
run_group "E31-E33 scan_sub" \
  "(cd scan_sub && ./run_all.sh ${QUICK})" || true

# ── 6) summary.md 통합 재생성 ─────────────────────────────
echo ""
echo "============================================================"
echo "  summary.md 통합 재생성"
echo "============================================================"
cd "$PROJECT_ROOT"
if "$VENV_PY" -m Sources.vppm.ablation.run --rebuild-summary 2>&1 | tee "$LOG_DIR/summary.log"; then
  echo "[ablation] ✓ summary.md 갱신 완료"
else
  echo "[ablation] WARN: summary.md 재생성 실패 — $LOG_DIR/summary.log 확인"
fi

echo ""
echo "============================================================"
echo "  Ablation Full Run 완료"
echo "============================================================"
echo "  로그: $LOG_DIR"
echo "  결과:"
echo "    - Sources/pipeline_outputs/ablation/E*_no_*/    (27 ablation)"
echo "    - Sources/pipeline_outputs/ablation/summary.md  (통합 요약)"
