"""VPPM-1DCNN 전용 하이퍼파라미터.

baseline 의 학습 하이퍼파라미터(LR, batch, optimizer, early stop 등) 는
``common.config`` 의 baseline 값을 그대로 재사용한다 — 본 모듈에서는
1DCNN 블록 자체의 설계값과 P1-P4 채널 인덱스, 출력 디렉터리만 정의한다.
"""
from __future__ import annotations

from ..common import config


# ============================================================
# 1D CNN 블록 설계
# ============================================================
N_CHANNELS = config.N_FEATURES        # = 21
SEQ_LENGTH = config.SV_Z_LAYERS       # = 70
KERNEL_SIZE = 3
CONV_LAYERS = 2

# ============================================================
# 채널 인덱스 (0-based, FEATURES.md §평균 처리 방식별 분류 와 일치)
# ============================================================
# P1: 픽셀 평균 + 레이어 CAD-가중평균 (10개)
#     #1 distance_from_edge, #2 distance_from_overhang, #4-#11 DSCNN 8채널
P1_INDICES = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]

# P2: 픽셀 평균 + 레이어 단순평균 (2개)
#     #20 laser_return_delay, #21 laser_stripe_boundaries
P2_INDICES = [19, 20]

# P3: 레이어 단순평균만 (7개) — 픽셀 처리 없음, z-block 내 SV 동일값
#     #12-#18 프린터 센서 7채널
P3_INDICES = [11, 12, 13, 14, 15, 16, 17]

# P4: 평균 없음 (2개) — layer-invariant 스칼라
#     #3 build_height, #19 laser_module
P4_INDICES = [2, 18]

# ============================================================
# 산출물 경로 — Sources/pipeline_outputs/experiments/vppm_1dcnn/
# ============================================================
EXPERIMENT_DIR = config.OUTPUT_DIR / "experiments" / "vppm_1dcnn"
FEATURES_DIR   = EXPERIMENT_DIR / "features"
MODELS_DIR     = EXPERIMENT_DIR / "models"
RESULTS_DIR    = EXPERIMENT_DIR / "results"

FEATURES_SEQ_NPZ    = FEATURES_DIR / "features_seq.npz"
NORMALIZATION_JSON  = FEATURES_DIR / "normalization.json"

# baseline 캐시 (검증용 비교 대상)
BASELINE_FEATURES_NPZ = config.FEATURES_DIR / "all_features.npz"
