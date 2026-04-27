#!/usr/bin/env bash
# DSCNN 서브 10개 실험(E5~E12, E23, E24)을 4-GPU 에 배치 스케줄로 병렬 실행.
#
# 호스트 GPU 구성: 0,1,2,3 (4장)
#   Batch 1: E5→GPU0  E6→GPU1  E7→GPU2  E8→GPU3    (powder, printed, streaking, edge_swelling)
#   Batch 2: E9→GPU0  E10→GPU1  E11→GPU2  E12→GPU3 (debris, super_elev, soot, excessive_melt)
#   Batch 3: E23→GPU0  E24→GPU1                     (defects_all, normal_all)
#
# 각 컨테이너는 --skip-summary 로 실행되어 summary.md race 를 막고,
# 전체 완료 후 호스트 venv 로 --rebuild-summary 를 호출해 통합 summary.md 를 생성한다.
#
# Usage:
#   ./run_all.sh           # 전체 학습 (GPU 4장 병렬, ~45~60분 예상)
#   ./run_all.sh --quick   # smoke test (~5분)
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
EXPERIMENT_ID=E5 NVIDIA_VISIBLE_DEVICES=0 "${DC[@]}" build

LOG_DIR="/tmp/dscnn_sub_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[run_all] logs: $LOG_DIR"

# ── 2) 배치 실행 헬퍼 ────────────────────────────────────
# $1: 배치 라벨, $2+: "EXP:GPU" 쌍 목록
run_batch() {
  local label="$1"; shift
  echo ""
  echo "[run_all] === $label ==="
  declare -A pids
  for pair in "$@"; do
    local exp="${pair%%:*}"
    local gpu="${pair##*:}"
    echo "[run_all] launching $exp on GPU $gpu"
    (
      EXPERIMENT_ID="$exp" \
      NVIDIA_VISIBLE_DEVICES="$gpu" \
      "${DC[@]}" run --rm ablation-dscnn-sub
    ) > "$LOG_DIR/${exp}.log" 2>&1 &
    pids[$exp]=$!
    echo "[run_all]   PID=${pids[$exp]}  log=$LOG_DIR/${exp}.log"
  done

  local fail=0
  for exp in "${!pids[@]}"; do
    if wait "${pids[$exp]}"; then
      echo "[run_all] ✓ $exp OK"
    else
      echo "[run_all] ✗ $exp FAILED — $LOG_DIR/${exp}.log 확인"
      fail=1
    fi
  done
  return $fail
}

# ── 3) 3개 배치 순차 실행 (각 배치는 내부 병렬) ─────────────
overall_fail=0
run_batch "Batch 1/3 — E5~E8 (powder/printed/streaking/edge_swelling)" \
  E5:0 E6:1 E7:2 E8:3 || overall_fail=1

run_batch "Batch 2/3 — E9~E12 (debris/super_elev/soot/excessive_melt)" \
  E9:0 E10:1 E11:2 E12:3 || overall_fail=1

run_batch "Batch 3/3 — E23, E24 (defects_all, normal)" \
  E23:0 E24:1 || overall_fail=1

if [[ $overall_fail -ne 0 ]]; then
  echo ""
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
  echo "    (cd docker/ablation/dscnn_sub && \\"
  echo "      EXPERIMENT_ID=E5 NVIDIA_VISIBLE_DEVICES=0 sudo -E docker compose run --rm \\"
  echo "      ablation-dscnn-sub /workspace/venv/bin/python -m Sources.vppm.ablation.run --rebuild-summary)"
fi

echo ""
echo "[run_all] done. 로그: $LOG_DIR"
