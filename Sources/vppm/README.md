# VPPM 재구현 코드 설명

> **논문**: Scime et al., "A Data-Driven Framework for Direct Local Tensile Property Prediction of Laser Powder Bed Fusion Parts", *Materials* 2023, 16, 7293

이 디렉토리(`Sources/vppm/`)는 위 논문의 핵심 파이프라인 — **슈퍼복셀 기반 피처 엔지니어링 + VPPM(Voxelized Property Prediction Model)** — 을 재구현한 코드입니다.

---

## 파이프라인 전체 흐름

```
HDF5 원본 (ORNL_Data_Origin/)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 1  supervoxel.py   빌드 볼륨 → 슈퍼복셀 격자 분할     │
│  Phase 2  features.py     슈퍼복셀당 21개 피처 추출           │
│           ↓ .npz 저장                                        │
│  Merge    run_pipeline.py 빌드별 피처 병합 → all_features.npz │
│  Phase 3  dataset.py      정규화, 필터링, K-Fold 분할         │
│  Phase 4  model.py        VPPM 모델 정의                      │
│  Phase 5  train.py        5-Fold CV 학습 + Early Stopping     │
│  Phase 6  evaluate.py     RMSE 평가 + 시각화                  │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
pipeline_outputs/
  ├── features/       추출된 피처 (.npz)
  ├── models/         학습된 모델 가중치 (.pt)
  └── results/        평가 메트릭 + 플롯
```

---

## 파일별 상세 설명

### `config.py` — 전역 설정

파이프라인 전체에서 참조하는 상수와 경로를 한곳에 정의합니다.

| 항목 | 내용 |
|------|------|
| 경로 | `HDF5_DIR`, `OUTPUT_DIR`, `FEATURES_DIR`, `MODELS_DIR`, `RESULTS_DIR` |
| 빌드 매핑 | `BUILDS` dict — 빌드 ID(`B1.1`~`B1.5`) → HDF5 파일명 |
| 슈퍼복셀 파라미터 | xy 1.0mm, z 3.5mm (70레이어), 픽셀 해상도 ~0.133mm/px |
| DSCNN 클래스 매핑 | HDF5 12클래스 → 논문 8클래스 인덱스 변환 |
| Temporal 피처 키 | 7개 센서 데이터 키 목록 |
| 학습 하이퍼파라미터 | hidden=128, dropout=0.1, lr=1e-3, batch=1000, 5-fold, patience=50 |
| 타겟 속성 | YS, UTS, UE, TE (4개 인장 특성) |
| 측정 오차 | 논문 Section 2.9의 내재 오차값 |

### `supervoxel.py` — Phase 1: 슈퍼복셀 그리드

빌드 볼륨(245×245mm, 수천 레이어)을 **1.0×1.0×3.5mm 직육면체 격자**로 분할합니다.

- **`SuperVoxelGrid` 클래스**
  - `from_hdf5(path)`: HDF5에서 이미지 크기/레이어 수를 읽어 그리드 자동 생성
  - `get_pixel_range(ix, iy)`: 슈퍼복셀의 2D 픽셀 범위 (row_start, row_end, col_start, col_end)
  - `get_layer_range(iz)`: 슈퍼복셀의 z 방향 레이어 범위
  - `get_z_center_mm(iz)`: 슈퍼복셀 중심의 높이(mm)

- **`find_valid_supervoxels(grid, hdf5_path)`**: 학습에 사용할 유효 슈퍼복셀 식별
  - 조건 1: CAD 형상(`part_ids > 0`)과 교차
  - 조건 2: SS-J3 게이지 섹션(`sample_ids > 0`)과 10% 이상 교차
  - 반환: `voxel_indices`, `sample_ids`, `part_ids`, `cad_ratio`

### `features.py` — Phase 2: 21개 피처 엔지니어링

각 유효 슈퍼복셀에 대해 **21개 엔지니어링 피처**를 추출합니다.

- **`FeatureExtractor` 클래스**
  - `extract_features(valid_voxels)`: z-block 단위로 메모리 효율적 처리

| 피처 번호 | 이름 | 소스 | 설명 |
|:---------:|------|------|------|
| #1 | `distance_from_edge` | CAD | CAD 경계까지 거리 (3.0mm 포화) |
| #2 | `distance_from_overhang` | CAD | 오버행까지 거리 (71레이어 포화) |
| #3 | `build_height` | 좌표 | 슈퍼복셀 중심 z좌표 (mm) |
| #4-11 | `seg_*` (8개) | DSCNN | Powder, Printed, Recoater Streaking, Edge Swelling, Debris, Super-Elevation, Soot, Excessive Melting |
| #12-18 | temporal (7개) | 센서 | layer_print_time, gas flow rates, O₂, 온도 등 |
| #19-21 | laser (3개) | 스캔 | laser_module, return_delay, stripe_boundaries |

처리 방식:
- 가우시안 블러(σ=0.5mm) 적용 후 CAD 영역 내 평균
- z-block 내 레이어별 가중 평균으로 3D→1D 축소

### `dataset.py` — Phase 3: 데이터셋 구성

추출된 피처를 학습에 사용할 수 있는 형태로 가공합니다.

- **`build_dataset(features, sample_ids, targets)`**: 데이터 전처리 파이프라인
  1. 무효 샘플 제거: UTS < 50 MPa, NaN 타겟/피처
  2. **Min-max 정규화 → [-1, 1]** 범위로 스케일링
  3. 정규화 파라미터(min/max) 저장

- **`create_cv_splits(sample_ids)`**: **샘플 단위** K-Fold 분할
  - 핵심: 같은 시편의 모든 슈퍼복셀은 **반드시 같은 fold**에 배정 (데이터 누출 방지)

- **`VPPMDataset(Dataset)`**: PyTorch Dataset 래퍼
- **`normalize()` / `denormalize()`**: [-1, 1] 정규화 및 역변환

### `model.py` — Phase 4: VPPM 모델 정의

논문 Table 6의 아키텍처를 그대로 구현한 **2-layer MLP**입니다.

```
Input(21) → FC(128) → ReLU → Dropout(0.1) → FC(1) → Output
```

- **`VPPM(nn.Module)` 클래스**
  - 가중치 초기화: `N(0, 0.1)` 가우시안
  - 파라미터 수: 21×128 + 128 + 128×1 + 1 = **2,945**

### `train.py` — Phase 5: 학습 파이프라인

4개 인장 특성 × 5-fold = **총 20개 모델**을 학습합니다.

- **`EarlyStopper`**: 검증 손실이 50 에포크 동안 개선되지 않으면 학습 중단
  - 최적 모델 상태를 자동 저장/복원

- **`train_single_fold()`**: 단일 fold 학습
  - 손실 함수: **L1 Loss** (MAE)
  - 옵티마이저: **Adam** (lr=1e-3, betas=(0.9, 0.999), eps=1e-4)
  - 배치 크기: 1000
  - 최대 에포크: 5000

- **`train_all(dataset)`**: 전체 학습 오케스트레이션
  - 각 fold의 모델 가중치를 `.pt` 파일로 저장
  - 학습 로그를 `training_log.json`으로 기록

### `evaluate.py` — Phase 6: 평가 및 시각화

학습된 모델의 성능을 정량적·정성적으로 평가합니다.

- **`evaluate_fold()`**: 단일 fold 평가
  - 슈퍼복셀별 예측 → **샘플별 최소값** 취합 (보수적 추정, 논문 Section 3.1)
  - RMSE 계산

- **`evaluate_all(dataset)`**: 5-fold CV 평균 RMSE
  - Naive baseline(전체 평균 예측)과 비교하여 개선율 계산

- **`plot_correlation(results)`**: 예측 vs 실측 **2D 히스토그램** (논문 Figure 17 재현)
- **`plot_scatter_uts(results)`**: UTS **산점도** (논문 Figure 18 재현)
- **`save_metrics(results)`**: 메트릭 요약을 JSON으로 저장

### `run_pipeline.py` — 전체 파이프라인 진입점

모든 단계를 연결하는 **메인 실행 스크립트**입니다.

- **`extract_features_for_build(build_id)`**: 단일 빌드의 피처 추출 (Phase 1-2)
- **`merge_all_builds(build_ids)`**: 빌드별 피처 병합 (빌드 간 sample ID 충돌 방지를 위해 오프셋 적용)
- **`run_train()`**: 데이터셋 구축 → 학습 → 즉시 평가 (Phase 3-6)

```bash
# 실행 예시
python -m Sources.vppm.run_pipeline --quick-test        # B1.2만으로 빠른 테스트
python -m Sources.vppm.run_pipeline --all                # 전체 빌드 파이프라인
python -m Sources.vppm.run_pipeline --phase features --builds B1.1 B1.2  # 피처만
python -m Sources.vppm.run_pipeline --phase train        # 학습만
python -m Sources.vppm.run_pipeline --phase evaluate     # 평가만
python -m Sources.vppm.run_pipeline --all --n-feats 11   # ablation (피처 11개만)
```

### `__init__.py` — 패키지 초기화

빈 파일. `Sources.vppm`을 Python 패키지로 인식하게 합니다.

---

## 핵심 설계 결정

| 결정 | 이유 |
|------|------|
| 학습률 1e-3 (논문은 1e-8) | [-1,1] 정규화된 타겟에서 1e-8은 수렴 불가. 실용적 값으로 조정 |
| L1 Loss | 논문 원문 그대로. 이상치에 강건한 손실 함수 |
| 샘플 단위 K-Fold | 같은 시편의 슈퍼복셀이 train/val에 걸치면 데이터 누출 발생 |
| 샘플별 최소값 취합 | 가장 취약한 슈퍼복셀이 파단을 결정한다는 물리적 직관 반영 |
| z-block 단위 처리 | HDF5 전체 로드 시 메모리 초과 방지 (빌드당 ~50GB) |
| 피처 #19-21 placeholder | 스캔 경로 기반 피처는 구현 복잡도가 높아 현재 0으로 고정 |

---

## 출력물 구조

```
Sources/pipeline_outputs/
├── features/
│   ├── B1.1_features.npz       # 빌드별 피처 (features, sample_ids, targets 포함)
│   ├── B1.2_features.npz
│   ├── ...
│   ├── all_features.npz        # 전체 병합 피처
│   └── normalization.json      # 정규화 파라미터 (min/max)
├── models/
│   ├── vppm_YS_fold0.pt        # YS 모델 (fold 0~4)
│   ├── vppm_UTS_fold0.pt       # UTS 모델
│   ├── vppm_UE_fold0.pt        # UE 모델
│   ├── vppm_TE_fold0.pt        # TE 모델
│   └── training_log.json       # fold별 val loss, 에포크 수
└── results/
    ├── metrics_summary.json    # RMSE 요약 (VPPM vs Naive)
    ├── correlation_plots.png   # 4개 속성 2D 히스토그램
    └── scatter_plot_uts.png    # UTS 산점도
```

---

## 의존 패키지

| 패키지 | 용도 |
|--------|------|
| `torch` | 모델 정의, 학습, 추론 |
| `h5py` | HDF5 파일 읽기 |
| `numpy` | 수치 연산 |
| `scipy` | 가우시안 블러, 거리 변환 |
| `scikit-learn` | K-Fold 교차검증 분할 |
| `matplotlib` | 평가 시각화 |
| `tqdm` | 진행률 표시 |
