"""
VPPM 재구현 설정
논문: Scime et al., Materials 2023, 16, 7293
"""
from pathlib import Path

# ============================================================
# 경로
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ORNL_DATA_ROOT = PROJECT_ROOT / "ORNL_Data" / "Co-Registered In-Situ and Ex-Situ Dataset"
HDF5_DIR = ORNL_DATA_ROOT / "[baseline] (Peregrine v2023-11)"
HDF5_DIR_NEW_V1 = ORNL_DATA_ROOT / "[new_v1] (Peregrine v2023-09)"
HDF5_DIR_NEW_V2 = ORNL_DATA_ROOT / "[new_v2] (Peregrine v2023-10)"
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
# VPPM-LSTM 확장 (Sources/vppm/lstm/PLAN.md)
# ============================================================
# Per-supervoxel 이미지 시퀀스 (가변 길이) → CNN per-frame → LSTM → 1-dim 임베딩
# 21 baseline 피처에 concat → 22-dim → 기존 VPPM MLP

# 입력 — 카메라 채널: 0 = 용융 직후
LSTM_CAMERA_CHANNEL = 0

# 시퀀스 / 크롭
LSTM_T_MAX = SV_Z_LAYERS         # 70 — 최대 시퀀스 길이 (실제 T_sv 는 가변)
LSTM_CROP_H = 8                  # SV xy 픽셀 = 8 (1842 // 8 = 230 그리드)
LSTM_CROP_W = 8

# CNN 인코더 — Conv 3×3 두 층 + GAP + Linear
LSTM_CNN_CH1 = 16
LSTM_CNN_CH2 = 32
LSTM_CNN_KERNEL = 3
LSTM_D_CNN = 32                  # 프레임당 임베딩 차원 (LSTM 입력)

# LSTM
LSTM_D_HIDDEN = 16               # hidden state 차원 (= cell state)
LSTM_NUM_LAYERS = 1
LSTM_BIDIRECTIONAL = False       # forward 만. CLI flag 로 override 가능

# 최종 임베딩
LSTM_D_EMBED = 1                 # 22 = 21 + 1 (사용자 지정)

# 학습 하이퍼파라미터 (baseline 동일 골격)
LSTM_LR = 1e-3
LSTM_BATCH_SIZE = 256
LSTM_MAX_EPOCHS = 5000
LSTM_EARLY_STOP_PATIENCE = 50
LSTM_NUM_WORKERS = 0             # in-memory 라 worker 불필요
LSTM_GRAD_CLIP = 1.0
LSTM_WEIGHT_DECAY = 0.0          # baseline 도 무 weight decay

# 산출물 경로 — Sources/pipeline_outputs/experiments/vppm_lstm/
LSTM_EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / "vppm_lstm"
LSTM_CACHE_DIR = LSTM_EXPERIMENT_DIR / "cache"
LSTM_MODELS_DIR = LSTM_EXPERIMENT_DIR / "models"
LSTM_RESULTS_DIR = LSTM_EXPERIMENT_DIR / "results"
LSTM_FEATURES_DIR = LSTM_EXPERIMENT_DIR / "features"  # normalization.json 저장용

# ============================================================
# VPPM-LSTM-Dual 확장 (Sources/vppm/lstm_dual/PLAN.md)
# ============================================================
# visible/0 + visible/1 두 채널 각각 CNN+LSTM 임베딩 (1-dim) → 21 + 1 + 1 = 23 features
# visible/0 캐시는 기존 LSTM_CACHE_DIR (vppm_lstm/cache/) 재사용,
# visible/1 캐시만 lstm_dual 전용 디렉터리에 새로 빌드.

LSTM_DUAL_CAMERA_CHANNEL_V1 = 1   # 추가 채널 (분말 도포 후)

# 채널별 임베딩 차원 (둘 다 1 → 23 = 21 + 1 + 1)
LSTM_DUAL_D_EMBED_V0 = 1
LSTM_DUAL_D_EMBED_V1 = 1

# CNN/LSTM 가중치 공유 여부 — 기본 False (채널별 독립 브랜치)
LSTM_DUAL_SHARE_CNN = False
LSTM_DUAL_SHARE_LSTM = False

# 산출물 경로 — Sources/pipeline_outputs/experiments/vppm_lstm_dual/
LSTM_DUAL_EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / "vppm_lstm_dual"
LSTM_DUAL_CACHE_DIR = LSTM_DUAL_EXPERIMENT_DIR / "cache"     # visible/1 캐시 전용
LSTM_DUAL_MODELS_DIR = LSTM_DUAL_EXPERIMENT_DIR / "models"
LSTM_DUAL_RESULTS_DIR = LSTM_DUAL_EXPERIMENT_DIR / "results"
LSTM_DUAL_FEATURES_DIR = LSTM_DUAL_EXPERIMENT_DIR / "features"

# ============================================================
# VPPM-LSTM-Dual-4 확장 (Sources/vppm/lstm_dual_4/PLAN.md)
# ============================================================
# d_embed_v0/v1 = 4 → 21 + 4 + 4 = 29 features.
# proj 통로를 16→1 에서 16→4 로 확장 (LSTM hidden 16-dim 출력은 그대로).
# 캐시는 dual 와 동일 (재추출 안 함): v0=LSTM_CACHE_DIR, v1=LSTM_DUAL_CACHE_DIR 재사용.

LSTM_DUAL_4_D_EMBED_V0 = 4
LSTM_DUAL_4_D_EMBED_V1 = 4

# 산출물 경로 — Sources/pipeline_outputs/experiments/vppm_lstm_dual_4/
LSTM_DUAL_4_EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / "vppm_lstm_dual_4"
LSTM_DUAL_4_MODELS_DIR = LSTM_DUAL_4_EXPERIMENT_DIR / "models"
LSTM_DUAL_4_RESULTS_DIR = LSTM_DUAL_4_EXPERIMENT_DIR / "results"
LSTM_DUAL_4_FEATURES_DIR = LSTM_DUAL_4_EXPERIMENT_DIR / "features"
# 캐시는 dual 과 공유 — 재추출 안 함
LSTM_DUAL_4_CACHE_V0_DIR = LSTM_CACHE_DIR
LSTM_DUAL_4_CACHE_V1_DIR = LSTM_DUAL_CACHE_DIR

# ============================================================
# VPPM-LSTM-Dual-Img-4-Sensor-7 (Sources/vppm/lstm_dual_img_4_sensor_7/PLAN.md)
# ============================================================
# 카메라 v0/v1 LSTM(d_embed=4) + sensor LSTM(d_embed=7) 3-분기.
# G2(센서 7-feat 평균) 제거 → 14-feat baseline 사용 → MLP 입력 14+4+4+7 = 29 (dual_4 와 동일).

LSTM_DUAL_IMG_4_SENSOR_7_N_CHANNELS = len(TEMPORAL_FEATURES)   # = 7
LSTM_DUAL_IMG_4_SENSOR_7_D_HIDDEN_S = 16                        # sensor LSTM hidden
LSTM_DUAL_IMG_4_SENSOR_7_D_EMBED_S = 7                          # sensor proj 출력
LSTM_DUAL_IMG_4_SENSOR_7_NUM_LAYERS = 1
LSTM_DUAL_IMG_4_SENSOR_7_BIDIRECTIONAL = False

# 산출물 경로
LSTM_DUAL_IMG_4_SENSOR_7_EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / "vppm_lstm_dual_img_4_sensor_7"
LSTM_DUAL_IMG_4_SENSOR_7_CACHE_DIR    = LSTM_DUAL_IMG_4_SENSOR_7_EXPERIMENT_DIR / "cache"
LSTM_DUAL_IMG_4_SENSOR_7_MODELS_DIR   = LSTM_DUAL_IMG_4_SENSOR_7_EXPERIMENT_DIR / "models"
LSTM_DUAL_IMG_4_SENSOR_7_RESULTS_DIR  = LSTM_DUAL_IMG_4_SENSOR_7_EXPERIMENT_DIR / "results"
LSTM_DUAL_IMG_4_SENSOR_7_FEATURES_DIR = LSTM_DUAL_IMG_4_SENSOR_7_EXPERIMENT_DIR / "features"
# 카메라 캐시는 lstm/lstm_dual 과 공유
LSTM_DUAL_IMG_4_SENSOR_7_CACHE_V0_DIR = LSTM_CACHE_DIR
LSTM_DUAL_IMG_4_SENSOR_7_CACHE_V1_DIR = LSTM_DUAL_CACHE_DIR

# ============================================================
# VPPM-LSTM-Dual-Img-4-Sensor-7-DSCNN-8 (Sources/vppm/lstm_dual_img_4_sensor_7_dscnn_8/PLAN.md)
# ============================================================
# 카메라 v0/v1 LSTM(d_embed=4) + sensor LSTM(d_embed=7) + DSCNN LSTM(d_embed=8) 4-분기.
# G1(DSCNN 8-feat 가중평균) + G2(센서 7-feat 평균) 둘 다 제거 → 6-feat baseline (G3+G4)
# → MLP 입력 6 + 4 + 4 + 7 + 8 = 29 (sensor_7 / dual_4 와 동일).

LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_N_CHANNELS = len(DSCNN_FEATURE_MAP)   # = 8
LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_D_HIDDEN_D = 16                       # DSCNN LSTM hidden
LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_D_EMBED_D = 8                         # DSCNN proj 출력
LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_NUM_LAYERS = 1
LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_BIDIRECTIONAL = False

# 산출물 경로
LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / "vppm_lstm_dual_img_4_sensor_7_dscnn_8"
LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_DIR    = LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_EXPERIMENT_DIR / "cache"
LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_MODELS_DIR   = LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_EXPERIMENT_DIR / "models"
LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_RESULTS_DIR  = LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_EXPERIMENT_DIR / "results"
LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_FEATURES_DIR = LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_EXPERIMENT_DIR / "features"
# 카메라 캐시는 lstm/lstm_dual 과 공유, sensor 캐시는 sensor_7 디렉터리 재사용
LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_V0_DIR     = LSTM_CACHE_DIR
LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_V1_DIR     = LSTM_DUAL_CACHE_DIR
LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_SENSOR_DIR = LSTM_DUAL_IMG_4_SENSOR_7_CACHE_DIR

# ============================================================
# VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-1DCNN-Sensor-4
# (Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md)
# 짧은 별칭 prefix: LSTM_FULL86_  (정식 이름이 너무 길어서 86-feat MLP 입력 차원 기반)
# ============================================================
# 카메라 v0/v1 — d_embed 16 (dscnn_8 의 4 보다 4× 확장)
LSTM_FULL86_D_HIDDEN_CAM = 16
LSTM_FULL86_D_EMBED_V0   = 16
LSTM_FULL86_D_EMBED_V1   = 16

# Sensor — 필드별 1D-CNN, 필드당 4-dim (sensor_7 의 7-ch LSTM 과 표현력 4×)
LSTM_FULL86_N_SENSOR_FIELDS    = len(TEMPORAL_FEATURES)            # = 7
LSTM_FULL86_D_PER_SENSOR_FIELD = 4
LSTM_FULL86_SENSOR_HIDDEN_CH   = 16
LSTM_FULL86_SENSOR_KERNEL      = 5

# DSCNN — 8-ch LSTM (dscnn_8 와 동일)
LSTM_FULL86_N_DSCNN_CH = len(DSCNN_FEATURE_MAP)                    # = 8
LSTM_FULL86_D_HIDDEN_D = 16
LSTM_FULL86_D_EMBED_D  = 8

# CAD — spatial-CNN+LSTM, 8×8 패치 보존 (in_channels=2)
# 채널 0 = edge_proximity (mm, inversion + cad_mask 픽셀곱), 채널 1 = overhang_proximity (layers, 동일)
LSTM_FULL86_N_CAD_CH      = 2
LSTM_FULL86_CAD_PATCH_H   = LSTM_CROP_H                            # = 8 (카메라와 동일)
LSTM_FULL86_CAD_PATCH_W   = LSTM_CROP_W
LSTM_FULL86_D_CNN_C       = 32                                     # cad spatial CNN proj 출력
LSTM_FULL86_D_HIDDEN_C    = 16
LSTM_FULL86_D_EMBED_C     = 8                                      # 채널당 4-dim
LSTM_FULL86_CAD_INVERSION_APPLIED = True                           # 3.0 - dist / 71 - dist
LSTM_FULL86_CAD_MASK_APPLIED      = True                           # cad_mask 픽셀곱

# Scan — spatial-CNN+LSTM, 8×8 패치 보존 (in_channels=2)
# 채널 0 = return_delay (s, raw + NaN→0), 채널 1 = stripe_boundaries (a.u., raw)
LSTM_FULL86_N_SCAN_CH     = 2
LSTM_FULL86_SCAN_PATCH_H  = LSTM_CROP_H                            # = 8
LSTM_FULL86_SCAN_PATCH_W  = LSTM_CROP_W
LSTM_FULL86_D_CNN_SC      = 32
LSTM_FULL86_D_HIDDEN_SC   = 16
LSTM_FULL86_D_EMBED_SC    = 8
LSTM_FULL86_SCAN_INVERSION_APPLIED = False                         # raw 그대로 (이미 0=nominal)
LSTM_FULL86_SCAN_MASK_APPLIED      = False                         # baseline 처리가 미용융=0 부여

# 정적 피처 (P4) — 21-feat 에서 추출
LSTM_FULL86_STATIC_IDX = [2, 18]                                   # build_height, laser_module

# 결합 MLP — 86 → 256 → 128 → 64 → 1 (4 fc layer, baseline 의 1-layer 보다 깊음)
LSTM_FULL86_MLP_HIDDEN = (256, 128, 64)

# 산출물 경로
LSTM_FULL86_EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / "vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4"
LSTM_FULL86_CACHE_DIR      = LSTM_FULL86_EXPERIMENT_DIR / "cache"
LSTM_FULL86_MODELS_DIR     = LSTM_FULL86_EXPERIMENT_DIR / "models"
LSTM_FULL86_RESULTS_DIR    = LSTM_FULL86_EXPERIMENT_DIR / "results"
LSTM_FULL86_FEATURES_DIR   = LSTM_FULL86_EXPERIMENT_DIR / "features"

# 카메라 / sensor / dscnn 캐시는 기존 디렉터리 재사용
LSTM_FULL86_CACHE_V0_DIR     = LSTM_CACHE_DIR
LSTM_FULL86_CACHE_V1_DIR     = LSTM_DUAL_CACHE_DIR
LSTM_FULL86_CACHE_SENSOR_DIR = LSTM_DUAL_IMG_4_SENSOR_7_CACHE_DIR
LSTM_FULL86_CACHE_DSCNN_DIR  = LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_CACHE_DIR
# cad_patch / scan_patch 는 본 실험 디렉터리에 신규 빌드
LSTM_FULL86_CACHE_CAD_DIR    = LSTM_FULL86_CACHE_DIR
LSTM_FULL86_CACHE_SCAN_DIR   = LSTM_FULL86_CACHE_DIR

# ============================================================
# LSTM Ablation (Sources/vppm/lstm_ablation/PLAN.md)
# 풀-스택 LSTM 의 카메라 분기 ablation 산출물 디렉터리.
# 캐시는 LSTM_FULL86_* 를 그대로 재사용.
# ============================================================
LSTM_ABLATION_EXPERIMENT_BASE_DIR = OUTPUT_DIR / "experiments" / "lstm_ablation"
LSTM_ABLATION_E1_DIR              = LSTM_ABLATION_EXPERIMENT_BASE_DIR / "E1_no_v0"
LSTM_ABLATION_E2_DIR              = LSTM_ABLATION_EXPERIMENT_BASE_DIR / "E2_no_cameras"
