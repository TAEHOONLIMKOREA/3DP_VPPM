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

# ============================================================
# VPPM-LSTM 업그레이드 (IMPLEMENTATION_PLAN_LSTM.md)
# ============================================================
# 이미지 스택 채널 구성: "raw" | "raw_both" | "dscnn" | "raw+dscnn"
LSTM_INPUT_CHANNELS = "raw+dscnn"
# 슈퍼복셀 패치 크기 (픽셀). sv_xy_pixels(=8)와 맞추는 것이 기본.
LSTM_PATCH_PX = 8
# CNN 인코더 출력 차원
LSTM_D_CNN = 64
# LSTM 임베딩 차원 — 최종 피처는 21 + LSTM_D_EMBED
LSTM_D_EMBED = 16
LSTM_BIDIRECTIONAL = True
LSTM_POOLING = "last"           # last | mean
LSTM_NUM_LAYERS = 1

# 학습 하이퍼파라미터 (이미지 텐서 대비 작은 배치, GPU 필수)
LSTM_LR = 1e-3
LSTM_BATCH_SIZE = 64
LSTM_MAX_EPOCHS = 200
LSTM_EARLY_STOP_PATIENCE = 20
LSTM_NUM_WORKERS = 2
LSTM_GRAD_CLIP = 1.0
LSTM_WEIGHT_DECAY = 1e-4

# 캐시 디렉터리: 프로젝트 내부 영속 저장 (리부팅해도 남음)
import os as _os
LSTM_CACHE_DIR = _os.environ.get(
    "LSTM_CACHE_DIR", str(OUTPUT_DIR / "image_stacks")
)
LSTM_CACHE_PERSIST = True

# 학습 산출물 — plan §3 디렉터리 구조와 일치
LSTM_EMBEDDINGS_DIR = OUTPUT_DIR / "lstm_embeddings"
LSTM_MODELS_DIR = OUTPUT_DIR / "models_lstm"
LSTM_RESULTS_DIR = OUTPUT_DIR / "results" / "vppm_lstm"
