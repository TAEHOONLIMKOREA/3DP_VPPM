# VPPM 데이터 파이프라인 설명

> **논문**: Scime et al., "A Data-Driven Framework for Direct Local Tensile Property Prediction of Laser Powder Bed Fusion Parts", *Materials* 2023, 16, 7293

이 문서는 `Sources/vppm/` 의 **데이터 파이프라인** — HDF5 원본 → 슈퍼복셀 분할 → 21 피처 추출 → 정규화/K-fold 분할 — 에 대한 설명입니다.

**모델 설명은 분리되어 있습니다:**
- `origin/MODEL.md` — 원본 VPPM (21 피처 → MLP)
- `lstm/MODEL.md` — VPPM-LSTM 확장 (21 피처 + 이미지 시퀀스 CNN+LSTM)

---

## 데이터 파이프라인 전체 흐름

```
ORNL_Data_Origin/  (5 빌드 HDF5, ~230GB)
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ Phase 1  common/supervoxel.py  빌드 볼륨 → 슈퍼복셀 격자 분할    │
│ Phase 2  origin/features.py    슈퍼복셀당 21개 피처 추출          │
│          ↓ {build}_features.npz 저장                              │
│ Merge    run_pipeline.py       빌드별 피처 병합 → all_features.npz│
│ Phase 3  common/dataset.py     정규화, 필터링, Sample-wise K-Fold │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
pipeline_outputs/features/
  ├── B1.1_features.npz ~ B1.5_features.npz   # 빌드별 피처
  ├── all_features.npz                         # 전체 병합
  └── normalization.json                       # [-1,1] 정규화 파라미터
```

(LSTM 확장의 경우 추가로 Phase L1: `lstm/image_stack.py` 로 슈퍼복셀당 이미지 시퀀스 HDF5 가 생성됨 — 세부 내용은 `lstm/MODEL.md` 참조)

---

## 파일별 상세 설명

### `common/config.py` — 전역 설정

파이프라인 전체에서 참조하는 상수와 경로.

| 항목 | 내용 |
|------|------|
| 경로 | `HDF5_DIR`, `OUTPUT_DIR`, `FEATURES_DIR`, `MODELS_DIR`, `RESULTS_DIR` |
| 빌드 매핑 | `BUILDS` dict — 빌드 ID(`B1.1`~`B1.5`) → HDF5 파일명 |
| 슈퍼복셀 파라미터 | xy 1.0mm, z 3.5mm (70레이어), 픽셀 해상도 ~0.133mm/px |
| 이미지 파라미터 | `IMAGE_PIXELS=1842`, `REAL_SIZE_MM=245`, `PIXEL_SIZE_MM=REAL/IMAGE` |
| DSCNN 클래스 매핑 | HDF5 12클래스 → 논문 8클래스 인덱스 변환 (`DSCNN_FEATURE_MAP`) |
| Temporal 피처 키 | `TEMPORAL_FEATURES` — 7개 센서 데이터 키 목록 |
| 타겟 속성 | `TARGET_PROPERTIES` — YS, UTS, UE, TE (4개 인장 특성) |
| 측정 오차 | `MEASUREMENT_ERROR` — 논문 Section 2.9의 내재 오차값 |
| 피처 그룹 | `FEATURE_GROUPS` — ablation 용 CAD/DSCNN/Sensor/Scan 4 그룹 인덱스 |

(학습 하이퍼파라미터 — `HIDDEN_DIM`, `DROPOUT_RATE`, `LEARNING_RATE`, `N_FOLDS` 등 — 은 `origin/MODEL.md` 와 `lstm/MODEL.md` 에서 설명)

### `common/supervoxel.py` — Phase 1: 슈퍼복셀 그리드

빌드 볼륨(245×245mm, 수천 레이어)을 **1.0×1.0×3.5mm 직육면체 격자** 로 분할.

- **`SuperVoxelGrid` 클래스**
  - `from_hdf5(path)`: HDF5 에서 이미지 크기/레이어 수를 읽어 그리드 자동 생성
  - `get_pixel_range(ix, iy)`: 슈퍼복셀의 2D 픽셀 범위 (row_start, row_end, col_start, col_end)
  - `get_layer_range(iz)`: 슈퍼복셀의 z 방향 레이어 범위
  - `get_z_center_mm(iz)`: 슈퍼복셀 중심 높이(mm)

- **`find_valid_supervoxels(grid, hdf5_path)`**: 학습에 사용할 유효 슈퍼복셀 식별
  - 조건 1: CAD 형상(`part_ids > 0`) 과 교차
  - 조건 2: SS-J3 게이지 섹션(`sample_ids > 0`) 과 10% 이상 교차 (`SAMPLE_OVERLAP_THRESHOLD`)
  - 반환: `voxel_indices (N,3)`, `sample_ids`, `part_ids`, `cad_ratio`

### `origin/features.py` — Phase 2: 21개 피처 엔지니어링

각 유효 슈퍼복셀에 대해 **21개 엔지니어링 피처** 를 추출.

- **`FeatureExtractor` 클래스**
  - `extract_features(valid_voxels)`: z-block 단위로 메모리 효율적 처리

| 피처 번호 | 이름 | 소스 | 설명 |
|:---------:|------|------|------|
| #1  | `distance_from_edge`     | CAD    | CAD 경계까지 거리 (3.0mm 포화) |
| #2  | `distance_from_overhang` | CAD    | 오버행까지 거리 (71레이어 포화) |
| #3  | `build_height`           | 좌표   | 슈퍼복셀 중심 z좌표 (mm) |
| #4-11 | `seg_*` (8개)          | DSCNN  | Powder, Printed, Recoater Streaking, Edge Swelling, Debris, Super-Elevation, Soot, Excessive Melting 픽셀 평균 확률 |
| #12-18 | temporal (7개)        | 센서   | layer_print_time, gas flow rates, O₂, 온도 등 |
| #19-21 | laser (3개)           | 스캔   | laser_module, return_delay, stripe_boundaries |

처리 방식:
- 가우시안 블러(σ=0.5mm) 적용 후 CAD 영역 내 평균
- z-block 내 레이어별 가중 평균으로 3D→1D 축소

### `common/dataset.py` — Phase 3: 데이터셋 구성

추출된 피처를 학습 입력 형태로 가공.

- **`build_dataset(features, sample_ids, targets, build_ids)`**: 데이터 전처리 파이프라인
  1. 무효 샘플 제거: UTS < 50 MPa, NaN 타겟/피처
  2. **Min-max 정규화 → [-1, 1]** 범위로 스케일링 (빌드 레벨 min/max 기반)
  3. 정규화 파라미터(min/max) 반환 (`norm_params`)

- **`create_cv_splits(sample_ids, n_folds=5)`**: **샘플 단위** K-Fold 분할
  - 핵심: 같은 시편의 모든 슈퍼복셀은 **반드시 같은 fold** 에 배정 → 데이터 누출 방지

- **`VPPMDataset(Dataset)`**: PyTorch Dataset 래퍼 (21 피처 + 타겟)
- **`normalize()` / `denormalize()`**: [-1, 1] 정규화 및 역변환
- **`save_norm_params()`**: 정규화 파라미터 JSON 저장 → 추론 시 역정규화에 재사용

### `run_pipeline.py` — 파이프라인 진입점

전체 단계를 연결하는 **메인 실행 스크립트**.

- **`extract_features_for_build(build_id)`**: 단일 빌드의 피처 추출 (Phase 1-2)
- **`merge_all_builds(build_ids)`**: 빌드별 피처 병합 (빌드 간 sample ID 충돌 방지를 위해 오프셋 적용)
- **`run_train()`**: 데이터셋 구축 → 학습 → 즉시 평가 (Phase 3-6, 모델 단)

```bash
# 데이터 파이프라인 실행 예시
python -m Sources.vppm.run_pipeline --quick-test          # B1.2 만으로 빠른 테스트
python -m Sources.vppm.run_pipeline --all                 # 전체 빌드 (features → train → eval)
python -m Sources.vppm.run_pipeline --phase features --builds B1.1 B1.2
python -m Sources.vppm.run_pipeline --phase train         # 학습만 (features.npz 필요)
python -m Sources.vppm.run_pipeline --phase evaluate      # 평가만
python -m Sources.vppm.run_pipeline --all --n-feats 11    # ablation: 피처 11개만
```

LSTM 분기:
```bash
python -m Sources.vppm.run_pipeline --use-lstm --phase image-stack   # 이미지 스택 캐시 빌드
python -m Sources.vppm.run_pipeline --use-lstm --phase train-lstm    # LSTM 학습
python -m Sources.vppm.run_pipeline --use-lstm --phase eval-lstm     # 평가
```

---

## 디렉토리 구조

```
Sources/vppm/
├── run_pipeline.py
├── regen_plots.py                  # origin/lstm 결과 PNG 재생성 유틸
├── common/                         # 공용 모듈
│   ├── config.py                   # ★ 전역 설정
│   ├── dataset.py                  # ★ 데이터셋, 정규화, K-Fold
│   ├── model.py                    # VPPM, VPPM_LSTM (모델 — origin/lstm MODEL.md 참조)
│   └── supervoxel.py               # ★ 슈퍼복셀 그리드
├── origin/                         # 원본 VPPM
│   ├── MODEL.md                    # VPPM 모델 설명
│   ├── features.py                 # ★ 21 피처 추출
│   ├── train.py
│   └── evaluate.py
├── lstm/                           # VPPM-LSTM 확장
│   ├── MODEL.md                    # VPPM-LSTM 모델 설명
│   ├── image_stack.py              # (Phase L1) HDF5 → stacks_all.h5
│   ├── encoder.py / sequence_model.py
│   ├── dataset.py / train_lstm.py / eval_lstm.py
├── tools/                          # 시각화/포맷 변환 유틸 (LSTM 무관)
│   ├── split_stacks_by_build.py    # stacks_all.h5 → 빌드별 분리 (H5Web 뷰용)
│   ├── view_stacks_example.py
│   ├── view_per_build.py
│   └── export_crop_png.py
└── ablation/
    └── run.py                      # 피처 그룹별 ablation 실험
```

★ 표시: 본 README 에서 다루는 데이터 파이프라인 핵심 모듈.

---

## 출력물 구조 (데이터 레이어)

```
Sources/pipeline_outputs/
├── features/
│   ├── B1.{1..5}_features.npz              # 빌드별 (features, sample_ids, targets, voxel_indices)
│   ├── all_features.npz                    # 전체 병합 (features, sample_ids, build_ids, targets)
│   └── normalization.json                  # 정규화 파라미터 (min/max per feature/target/build)
├── image_stacks/                           # (LSTM 경로 — lstm/MODEL.md 참조)
│   ├── stacks_all.h5                       # (36047, 70, 9, 8, 8) float16
│   └── per_build/stacks_B1.{1..5}.h5       # H5Web 뷰용 분리본 (float32+gzip)
├── models/, results/                       # 원본 VPPM 산출물 → origin/MODEL.md
└── models_lstm/, lstm_embeddings/, results/vppm_lstm/  # → lstm/MODEL.md
```

---

## 의존 패키지

| 패키지 | 용도 |
|--------|------|
| `h5py`         | HDF5 파일 읽기 |
| `numpy`        | 수치 연산 |
| `scipy`        | 가우시안 블러, 거리 변환 (features 추출) |
| `scikit-learn` | K-Fold 교차검증 분할 |
| `torch`        | 모델 정의/학습/추론 (→ origin/MODEL.md, lstm/MODEL.md) |
| `matplotlib`   | 평가 시각화 |
| `tqdm`         | 진행률 표시 |
