"""
VPPM 재구현 설정
논문: Scime et al., Materials 2023, 16, 7293
"""
from pathlib import Path

# ============================================================
# 경로
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
HDF5_DIR = PROJECT_ROOT / "ORNL_Data_Origin"
OUTPUT_DIR = PROJECT_ROOT / "Sources" / "pipeline_outputs"
FEATURES_DIR = OUTPUT_DIR / "features"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# ============================================================
# 빌드 정보
# ============================================================
BUILDS = {
    "B1.1": "2021-07-13 TCR Phase 1 Build 1.hdf5",
    "B1.2": "2021-04-16 TCR Phase 1 Build 2.hdf5",
    "B1.3": "2021-04-28 TCR Phase 1 Build 3.hdf5",
    "B1.4": "2021-08-03 TCR Phase 1 Build 4.hdf5",
    "B1.5": "2021-08-23 TCR Phase 1 Build 5.hdf5",
}

def hdf5_path(build_id: str) -> Path:
    return HDF5_DIR / BUILDS[build_id]

# ============================================================
# 슈퍼복셀 파라미터 (논문 Section 2.10)
# ============================================================
SV_XY_MM = 1.0            # x-y 평면 크기 (mm)
SV_Z_MM = 3.5             # z 방향 크기 (mm)
LAYER_THICKNESS_MM = 0.05  # 레이어 두께 (mm)
SV_Z_LAYERS = int(SV_Z_MM / LAYER_THICKNESS_MM)  # 70 레이어

# 이미지 파라미터 (HDF5 attrs에서 확인한 값)
IMAGE_PIXELS = 1842        # 이미지 크기 (pixels)
REAL_SIZE_MM = 245.0       # 물리적 크기 (mm)
PIXEL_SIZE_MM = REAL_SIZE_MM / IMAGE_PIXELS  # ~0.1330 mm/pixel
SV_XY_PIXELS = SV_XY_MM / PIXEL_SIZE_MM     # ~7.52 pixels

# 가우시안 블러 (논문 Appendix D)
GAUSSIAN_KERNEL_MM = 1.0
GAUSSIAN_STD_MM = 0.5
GAUSSIAN_STD_PIXELS = GAUSSIAN_STD_MM / PIXEL_SIZE_MM  # ~3.76 pixels

# ============================================================
# DSCNN 클래스 매핑 (HDF5 12클래스 → 논문 8클래스)
# ============================================================
DSCNN_FEATURE_MAP = {
    # feature_index: (hdf5_class_id, paper_class_name)
    0: (0, "Powder"),
    1: (1, "Printed"),
    2: (3, "Recoater Streaking"),
    3: (5, "Edge Swelling"),
    4: (6, "Debris"),
    5: (7, "Super-Elevation"),
    6: (8, "Soot"),
    7: (10, "Excessive Melting"),
}

# ============================================================
# Temporal 피처 매핑 (논문 Table A4)
# ============================================================
TEMPORAL_FEATURES = [
    "layer_times",                  # 피처 12
    "top_flow_rate",                # 피처 13
    "bottom_flow_rate",             # 피처 14
    "module_oxygen",                # 피처 15
    "build_plate_temperature",      # 피처 16
    "bottom_flow_temperature",      # 피처 17
    "actual_ventilator_flow_rate",  # 피처 18
]

# ============================================================
# VPPM 학습 하이퍼파라미터 (논문 Section 2.11)
# ============================================================
N_FEATURES = 21
HIDDEN_DIM = 128
DROPOUT_RATE = 0.1
LEARNING_RATE = 1e-3  # 논문은 1e-8이나, [-1,1] 정규화된 타겟에서는 수렴 불가. 실용적 값 사용
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-4
BATCH_SIZE = 1000
N_FOLDS = 5
MAX_EPOCHS = 5000
EARLY_STOP_PATIENCE = 50
WEIGHT_INIT_STD = 0.1
RANDOM_SEED = 42

# 예측 타겟
TARGET_PROPERTIES = [
    "yield_strength",
    "ultimate_tensile_strength",
    "uniform_elongation",
    "total_elongation",
]

TARGET_SHORT = {
    "yield_strength": "YS",
    "ultimate_tensile_strength": "UTS",
    "uniform_elongation": "UE",
    "total_elongation": "TE",
}

# 내재 측정 오차 (논문 Section 2.9)
MEASUREMENT_ERROR = {
    "yield_strength": 16.6,            # MPa
    "ultimate_tensile_strength": 15.6,  # MPa
    "uniform_elongation": 1.73,         # %
    "total_elongation": 2.92,           # %
}

# CAD 피처 파라미터
DIST_EDGE_SATURATION_MM = 3.0
DIST_OVERHANG_SATURATION_LAYERS = 71
SAMPLE_OVERLAP_THRESHOLD = 0.10  # 10%

# ============================================================
# Feature Ablation 그룹 (Sources/vppm/ablation/PLAN.md)
# ============================================================
# 0-based 인덱스. origin/features.py 의 FEATURE_NAMES 와 일치해야 함.
FEATURE_GROUPS = {
    "cad":    [0, 1, 2],                       # G3: distance_edge / distance_overhang / build_height
    "dscnn":  [3, 4, 5, 6, 7, 8, 9, 10],        # G1: DSCNN 8 classes
    "sensor": [11, 12, 13, 14, 15, 16, 17],     # G2: temporal sensors (7)
    "scan":   [18, 19, 20],                     # G4: laser_module / return_delay / stripe_boundaries
}

# 조합 ablation (PLAN_E13_combined.md)
FEATURE_GROUPS["dscnn_sensor"] = FEATURE_GROUPS["dscnn"] + FEATURE_GROUPS["sensor"]  # 15개

# 센서 서브 ablation (PLAN_sensor_subablation.md) — 개별 7채널 + 묶음 2개
FEATURE_GROUPS_SENSOR_SUB = {
    "sensor_print_time":    [11],          # E14
    "sensor_top_flow":      [12],          # E15
    "sensor_bottom_flow":   [13],          # E16
    "sensor_oxygen":        [14],          # E17
    "sensor_plate_temp":    [15],          # E18
    "sensor_flow_temp":     [16],          # E19
    "sensor_ventilator":    [17],          # E20
    "sensor_gas_flow_all":  [12, 13, 17],  # E21: 유량 3채널
    "sensor_thermal_all":   [15, 16],       # E22: 온도 2채널
}
FEATURE_GROUPS.update(FEATURE_GROUPS_SENSOR_SUB)

# DSCNN 서브 ablation (PLAN_dscnn_subablation.md) — 개별 8채널 + 묶음 2개
FEATURE_GROUPS_DSCNN_SUB = {
    # 1단계: 개별 채널 (E5~E12)
    "dscnn_powder":             [3],   # E5
    "dscnn_printed":            [4],   # E6
    "dscnn_recoater_streaking": [5],   # E7  — B1.5 리코터 손상 핵심 후보
    "dscnn_edge_swelling":      [6],   # E8
    "dscnn_debris":             [7],   # E9  — B1.4 스패터 관련
    "dscnn_super_elevation":    [8],   # E10
    "dscnn_soot":               [9],   # E11
    "dscnn_excessive_melting":  [10],  # E12 — B1.2 Keyhole 핵심 후보
    # 2단계: 카테고리 묶음
    "dscnn_defects_all":        [5, 6, 7, 8, 9, 10],  # E23: 결함 6채널 (Normal 2개만 남김)
    "dscnn_normal":             [3, 4],                # E24: Normal 2채널 (Defect 6개만 남김)
}
FEATURE_GROUPS.update(FEATURE_GROUPS_DSCNN_SUB)

# 스캔(G4) 서브 ablation — laser_module / return_delay / stripe_boundaries 단독
FEATURE_GROUPS_SCAN_SUB = {
    "scan_return_delay":       [19],  # E32: #20 (0-based idx 19) 단독 제거
    "scan_stripe_boundaries":  [20],  # E33: #21 (0-based idx 20) 단독 제거
}
FEATURE_GROUPS.update(FEATURE_GROUPS_SCAN_SUB)

# ============================================================
# Sample-LSTM 업그레이드 (PLAN_LSTM_v2.md)
# ============================================================
# 입력: 샘플별 raw camera 이미지 시퀀스 (channel 0 = 용융 직후 단일).
# 출력: 샘플 단위 LSTM 임베딩 — d_embed 차원 vector. 21 차원 핸드크래프트 피처에 concat (broadcast).
# 최종 입력 차원: 21 + LSTM_D_EMBED.
#
# 4 가지 모드를 지원 — 단/양방향 × 임베딩 1-dim/16-dim 조합:
#   fwd1    : forward,       d_embed=1   → 22 차원
#   bidir1  : bidirectional, d_embed=1   → 22 차원
#   fwd16   : forward,       d_embed=16  → 37 차원
#   bidir16 : bidirectional, d_embed=16  → 37 차원   (= 이전 v1 설계와 동일)
# 각 모드의 산출물은 별도 폴더에 분리됨.
import os as _os

# 카메라 채널 — 0(용융 직후) 만 사용 (DSCNN 채널 없음)
LSTM_RAW_CHANNEL = 0
# 샘플 bbox crop 후 resize 사이즈 (정사각)
LSTM_CROP_SIZE = 64
# CNN 인코더 출력 차원
LSTM_D_CNN = 64
# LSTM 내부 hidden — bidirectional 이면 실효 출력은 2*LSTM_D_HIDDEN. 모드 간 비교 일관성 위해 고정.
LSTM_D_HIDDEN = 8
LSTM_POOLING = "last"           # last | mean
LSTM_NUM_LAYERS = 1

# 학습 하이퍼파라미터 (가변 길이 시퀀스, GPU 필수)
LSTM_LR = 1e-3
LSTM_BATCH_SIZE = 8             # 샘플 단위라 배치 작게
LSTM_MAX_EPOCHS = 200
LSTM_EARLY_STOP_PATIENCE = 20
LSTM_NUM_WORKERS = 2
LSTM_GRAD_CLIP = 1.0
LSTM_WEIGHT_DECAY = 1e-4

# 캐시 디렉터리: 샘플별 (T, H, W) 시퀀스 텐서 (mode 무관)
LSTM_CACHE_DIR = _os.environ.get(
    "LSTM_CACHE_DIR", str(OUTPUT_DIR / "sample_stacks")
)
LSTM_CACHE_PERSIST = True

# 학습 산출물 — mode 별로 분리.
LSTM_EMBEDDINGS_DIR = OUTPUT_DIR / "lstm_embeddings"
LSTM_MODELS_DIR = OUTPUT_DIR / "models_lstm"
LSTM_RESULTS_DIR = OUTPUT_DIR / "results"

# 모드 사양 — 모든 모드는 hidden=8 공유. proj 만 다르게.
LSTM_MODES = {
    "fwd1":    {"bidirectional": False, "d_embed": 1},
    "bidir1":  {"bidirectional": True,  "d_embed": 1},
    "fwd16":   {"bidirectional": False, "d_embed": 16},
    "bidir16": {"bidirectional": True,  "d_embed": 16},
}
LSTM_VALID_MODES = tuple(LSTM_MODES.keys())


def lstm_paths(mode: str) -> dict:
    """Mode 별 산출물 path + 모델 설정."""
    if mode not in LSTM_MODES:
        raise ValueError(f"mode must be one of {LSTM_VALID_MODES}, got {mode!r}")
    spec = LSTM_MODES[mode]
    return {
        **spec,
        "n_feats":         N_FEATURES + spec["d_embed"],
        "models_dir":      LSTM_MODELS_DIR / mode,
        "embeddings_dir":  LSTM_EMBEDDINGS_DIR / mode,
        "embeddings_npz":  LSTM_EMBEDDINGS_DIR / mode / "embeddings.npz",
        "features_npz":    FEATURES_DIR / f"all_features_with_lstm_{mode}.npz",
        "results_dir":     LSTM_RESULTS_DIR / f"vppm_lstm_{mode}",
    }
