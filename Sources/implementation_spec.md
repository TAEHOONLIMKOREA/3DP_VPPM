# VPPM 재구현 스펙 (Implementation Specification)

> 이 문서는 논문 "A Data-Driven Framework for Direct Local Tensile Property Prediction of L-PBF Parts"의
> 핵심 파이프라인(N5: 슈퍼복셀 + N6: VPPM)을 재구현하기 위한 단계별 명세서입니다.
> **에이전트는 이 문서를 파싱하여, STATUS가 TODO인 단계를 순서대로 수행합니다.**

---

## 공통 규칙

- Python 환경: `./venv/bin/python`
- 소스 코드 위치: `Sources/` 하위
- 데이터 로더: `Sources/hdf5_parser/ornl_data_loader.py`의 `ORNLDataLoader` 사용
- HDF5 원본: `ORNL_Data_Origin/` (각 빌드별 ~50GB)
- 중간 결과물: `Sources/pipeline_outputs/` 하위에 저장
- 모든 코드는 메모리 효율적으로 작성 (HDF5를 전체 로드하지 않고 레이어 단위 처리)
- 에러 발생 시 해당 단계의 STATUS를 BLOCKED로 변경하고 에러 내용을 기록

---

## STEP 0: 환경 준비
- **STATUS**: TODO
- **OUTPUT**: `requirements.txt` 업데이트, 패키지 설치 확인
- **DESCRIPTION**:
  추가 필요한 패키지를 설치한다.

### 작업 내용
1. 기존 venv에 아래 패키지 추가 설치:
   - `torch` (PyTorch) - VPPM 모델 학습용
   - `scikit-learn` - 교차검증 split, 메트릭 계산용
   - `tqdm` - 진행률 표시
   - `scipy` - 가우시안 블러 등 신호처리
2. `requirements.txt` 업데이트
3. 설치 확인 스크립트 실행

---

## STEP 1: 좌표계 및 슈퍼복셀 그리드 정의
- **STATUS**: TODO
- **OUTPUT**: `Sources/supervoxel/grid.py`
- **DESCRIPTION**:
  빌드 볼륨을 1.0 x 1.0 x 3.5mm 슈퍼복셀 격자로 분할하는 모듈을 구현한다.

### 설계 상세

#### 좌표계
- HDF5 이미지 크기: (num_layers, H, W) - 대략 2048 x 2048 픽셀
- 물리적 크기: 245 x 245 mm (프린트 영역)
- 픽셀 해상도: `slices/origin`과 이미지 크기로부터 계산
  - 대략 245mm / 1842px ≈ 0.133 mm/pixel
- z축: 레이어 번호 x 레이어 두께(0.05mm)
- 좌표 원점: `slices/origin` 속성에서 읽어옴

#### 슈퍼복셀 그리드
- x-y 크기: 1.0 x 1.0 mm → 약 7.5 x 7.5 픽셀
- z 크기: 3.5 mm = 70 레이어
- z 방향 stride: 70 레이어 (슬라이딩 윈도우, stride = super-voxel height)

#### 클래스 인터페이스
```python
class SuperVoxelGrid:
    def __init__(self, xy_size_mm=1.0, z_size_mm=3.5, layer_thickness_mm=0.05,
                 pixel_size_mm=None, origin=None, image_shape=None):
        """
        Args:
            xy_size_mm: x-y 평면 슈퍼복셀 크기 (mm)
            z_size_mm: z 방향 슈퍼복셀 크기 (mm)
            layer_thickness_mm: 레이어 두께 (mm)
            pixel_size_mm: 픽셀 해상도 (mm/pixel), None이면 HDF5에서 계산
            origin: 좌표 원점 (mm), None이면 HDF5에서 읽기
            image_shape: (H, W) 이미지 크기
        """

    def setup_from_hdf5(self, hdf5_path: str):
        """HDF5 파일에서 좌표 정보를 읽어 그리드 초기화"""

    def get_voxel_indices(self) -> np.ndarray:
        """모든 슈퍼복셀의 (ix, iy, iz) 인덱스 배열 반환"""

    def get_pixel_range(self, ix, iy) -> tuple:
        """슈퍼복셀 (ix, iy)에 해당하는 픽셀 범위 반환
        Returns: (row_start, row_end, col_start, col_end)"""

    def get_layer_range(self, iz) -> tuple:
        """슈퍼복셀 iz에 해당하는 레이어 범위 반환
        Returns: (layer_start, layer_end)"""

    def filter_by_cad_geometry(self, part_ids_layer: np.ndarray, ix, iy) -> float:
        """해당 슈퍼복셀 내 CAD 형상(part_ids > 0)의 비율 반환
        비율이 0이면 해당 슈퍼복셀은 무효"""
```

---

## STEP 2: 피처 추출기 구현
- **STATUS**: TODO
- **OUTPUT**: `Sources/supervoxel/features.py`
- **DESCRIPTION**:
  21개 엔지니어링 피처를 슈퍼복셀 단위로 추출하는 모듈을 구현한다.

### DSCNN 클래스 매핑 (HDF5 12클래스 → 논문 8클래스)

| 논문 피처# | 논문 클래스명 | HDF5 class_id | HDF5 클래스명 |
|-----------|------------|---------------|-------------|
| 4 | Powder | 0 | Powder |
| 5 | Printed | 1 | Printed |
| 6 | Recoater streaking | 3 | Recoater Streaking |
| 7 | Edge swelling | 5 | Swelling |
| 8 | Debris | 6 | Debris |
| 9 | Super-elevation | 7 | Super-Elevation |
| 10 | Soot | 8 | Spatter |
| 11 | Excessive melting | 10 | Over Melting |

### 피처별 추출 알고리즘

#### 피처 1: Distance from edge
- **입력**: `slices/part_ids` (레이어별)
- **처리**:
  1. CAD 형상 바이너리 마스크 생성 (part_ids > 0)
  2. `scipy.ndimage.distance_transform_edt()` 적용
  3. 거리값을 mm 단위로 변환 (pixel_size_mm 곱셈)
  4. 3.0mm에서 saturation (cap)
  5. 1.0mm 커널, 0.5mm std 가우시안 블러 적용
  6. 슈퍼복셀 내 CAD 영역 픽셀만 평면 평균
  7. z 방향: 70레이어에 걸쳐 평균

#### 피처 2: Distance from overhang
- **입력**: `slices/part_ids` (레이어별)
- **처리**:
  1. 각 레이어에서 CAD 형상의 수직 열(column)별로 오버행 표면까지 거리 계산
  2. 오버행 = 현재 레이어에 CAD 있지만, 바로 아래 71레이어 이내에 CAD 없는 영역
  3. 거리를 레이어 수로 계산, 71레이어에서 saturation
  4. 1.0mm 커널, 0.5mm std 가우시안 블러 적용
  5. 슈퍼복셀 내 평균

#### 피처 3: Build height
- **입력**: 레이어 번호
- **처리**: `layer_number * layer_thickness_mm (0.05)`
- 슈퍼복셀의 중심 레이어 높이 사용

#### 피처 4-11: DSCNN 세그멘테이션 비율
- **입력**: `slices/segmentation_results/{class_id}` (레이어별)
- **처리** (각 클래스 동일):
  1. 해당 클래스의 바이너리 마스크 획득
  2. 1.0mm 커널, 0.5mm std 가우시안 블러 적용
  3. 슈퍼복셀 내 CAD 영역 픽셀만 평면 평균
  4. z 방향 70레이어에 걸쳐 평균
  - 결과: 해당 클래스의 distance-weighted 면적 비율

#### 피처 12: Layer print time
- **입력**: `temporal/layer_times`
- **처리**: 슈퍼복셀이 포함하는 70레이어의 시간 평균

#### 피처 13: Top gas flow rate
- **입력**: `temporal/top_flow_rate`
- **처리**: 70레이어 평균

#### 피처 14: Bottom gas flow rate
- **입력**: `temporal/bottom_flow_rate`
- **처리**: 70레이어 평균

#### 피처 15: Module oxygen
- **입력**: `temporal/module_oxygen`
- **처리**: 70레이어 평균

#### 피처 16: Build plate temperature
- **입력**: `temporal/build_plate_temperature`
- **처리**: 70레이어 평균

#### 피처 17: Bottom flow temperature
- **입력**: `temporal/bottom_flow_temperature`
- **처리**: 70레이어 평균

#### 피처 18: Actual ventilator flow rate
- **입력**: `temporal/actual_ventilator_flow_rate`
- **처리**: 70레이어 평균

#### 피처 19: Laser module
- **입력**: `scans/{layer}` 스캔 경로 데이터 + `parts/process_parameters/laser_module`
- **처리**:
  1. 해당 슈퍼복셀 위치를 용융한 레이저 모듈 식별
  2. 모듈 1이면 0.0, 모듈 2이면 1.0
  3. 70레이어 평균 (대부분 동일값)
- **대안**: `parts/process_parameters/laser_module`에서 파트별 레이저 모듈 직접 조회

#### 피처 20: Laser return delay
- **입력**: `scans/{layer}` 스캔 경로 데이터
- **처리**:
  1. 레이어 시작 이후 melt time 맵 생성 (1.0mm 커널 max/min 필터)
  2. 픽셀별 max-min 차이 계산
  3. saturation 값 적용 (스트라이프 경계 제외)
  4. 슈퍼복셀 내 평균
- **참고**: 스캔 경로 데이터가 없는 레이어는 건너뜀

#### 피처 21: Laser stripe boundaries
- **입력**: `scans/{layer}` 스캔 경로 데이터
- **처리**:
  1. 레이저 melt time 맵에 양축 Sobel 필터 적용
  2. 두 Sobel 응답의 제곱합의 제곱근 (RMS)
  3. 슈퍼복셀 내 평균

### 클래스 인터페이스
```python
class FeatureExtractor:
    def __init__(self, grid: SuperVoxelGrid, hdf5_path: str):
        """
        Args:
            grid: SuperVoxelGrid 인스턴스
            hdf5_path: HDF5 파일 경로
        """

    def extract_all_features(self, sample_ids_to_track: list = None) -> dict:
        """
        모든 슈퍼복셀에 대해 21개 피처 추출

        Args:
            sample_ids_to_track: 추적할 샘플 ID 목록 (None이면 전체)

        Returns:
            {
                'features': np.ndarray (N_voxels, 21),
                'voxel_indices': np.ndarray (N_voxels, 3),  # (ix, iy, iz)
                'sample_ids': np.ndarray (N_voxels,),  # 각 복셀의 대표 샘플 ID
                'part_ids': np.ndarray (N_voxels,),
                'cad_overlap_ratio': np.ndarray (N_voxels,),
            }
        """

    def _extract_cad_features(self, ...) -> np.ndarray:
        """피처 1-3 (CAD 기반) 추출"""

    def _extract_dscnn_features(self, ...) -> np.ndarray:
        """피처 4-11 (DSCNN 기반) 추출"""

    def _extract_temporal_features(self, ...) -> np.ndarray:
        """피처 12-18 (로그 파일 기반) 추출"""

    def _extract_scan_features(self, ...) -> np.ndarray:
        """피처 19-21 (스캔 경로 기반) 추출"""
```

---

## STEP 3: 슈퍼복셀-샘플 매핑 및 학습 데이터 구축
- **STATUS**: TODO
- **OUTPUT**: `Sources/supervoxel/dataset.py`, `Sources/pipeline_outputs/features/` 하위 파일
- **DESCRIPTION**:
  슈퍼복셀을 SS-J3 게이지 섹션과 매핑하고, 학습 데이터셋을 구축한다.

### 매핑 규칙
1. `slices/sample_ids`를 사용하여 각 슈퍼복셀이 어떤 SS-J3 샘플의 게이지 섹션과 교차하는지 확인
2. 교차 면적이 슈퍼복셀 면적의 10% 미만이면 무효 처리
3. in situ 데이터가 누락된 슈퍼복셀도 무효 처리
4. 유효한 슈퍼복셀의 피처 벡터와 해당 샘플의 인장 특성(YS, UTS, UE, TE)을 매핑

### 정규화
- 피처: zero-center 정규화 [-1, 1] (min-max 기반)
  - `X_norm = 2 * (X - X_min) / (X_max - X_min) - 1`
- 타겟(인장 특성): 동일하게 zero-center 정규화 [-1, 1]
- min/max 값은 학습+검증 데이터 전체에서 계산하여 저장

### 출력 파일 형식
```
Sources/pipeline_outputs/features/
├── B1.1_features.npz   # features, sample_ids, voxel_indices, ...
├── B1.2_features.npz
├── B1.3_features.npz
├── B1.4_features.npz
├── B1.5_features.npz
├── all_features.npz    # 5개 빌드 합산
└── normalization.json  # min/max 값 (피처 21개 + 타겟 4개)
```

---

## STEP 4: VPPM 모델 정의
- **STATUS**: TODO
- **OUTPUT**: `Sources/model/vppm.py`
- **DESCRIPTION**:
  Voxelized Property Prediction Model (퍼셉트론)을 PyTorch로 구현한다.

### 아키텍처
```
Input (21) → Linear(21, 128) → ReLU → Dropout(0.1) → Linear(128, 1)
```

| Layer | Type | Input | Output |
|-------|------|-------|--------|
| 1 | Fully Connected | 21 | 128 |
| - | ReLU activation | 128 | 128 |
| 2 | Dropout (10%) | 128 | 128 |
| 3 | Fully Connected | 128 | 1 |

### 구현 사항
```python
class VPPM(nn.Module):
    def __init__(self, n_features=21, hidden_size=128, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 초기화
- 가중치: 정규분포 (mean=0, std=0.1)
- 바이어스: 0

---

## STEP 5: 학습 파이프라인 구현
- **STATUS**: TODO
- **OUTPUT**: `Sources/model/train.py`
- **DESCRIPTION**:
  5-fold 교차검증 기반 VPPM 학습 파이프라인을 구현한다.

### 학습 설정
- **옵티마이저**: Adam
  - learning_rate = 1e-8
  - betas = (0.9, 0.999)
  - eps = 1e-4
- **손실 함수**: L1Loss (MAE)
- **배치 크기**: 1000 (슈퍼복셀 단위)
- **교차검증**: 5-fold
  - 분할 단위: **샘플 단위** (같은 샘플의 슈퍼복셀은 같은 fold에)
  - 학습 80%, 검증 20%
- **조기 종료**: 검증 오차 plateau 감지 (patience 기반)
- **모델 수**: 4개 (YS, UTS, UE, TE 각각 별도)

### 학습 루프 의사 코드
```python
for target_property in ['YS', 'UTS', 'UE', 'TE']:
    for fold in range(5):
        model = VPPM(n_features=21)
        init_weights(model, std=0.1)
        optimizer = Adam(model.parameters(), lr=1e-8, betas=(0.9, 0.999), eps=1e-4)
        criterion = L1Loss()

        train_loader = DataLoader(train_voxels, batch_size=1000, shuffle=True)
        val_loader = DataLoader(val_voxels, batch_size=1000)

        best_val_loss = inf
        patience_counter = 0

        for epoch in range(max_epochs):
            # Train
            model.train()
            for batch_features, batch_targets in train_loader:
                loss = criterion(model(batch_features), batch_targets)
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            val_loss = evaluate(model, val_loader)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
```

### 출력 파일
```
Sources/pipeline_outputs/models/
├── vppm_YS_fold0.pt ... vppm_YS_fold4.pt
├── vppm_UTS_fold0.pt ... vppm_UTS_fold4.pt
├── vppm_UE_fold0.pt ... vppm_UE_fold4.pt
├── vppm_TE_fold0.pt ... vppm_TE_fold4.pt
└── training_log.json  # 학습 이력 (epoch별 loss)
```

---

## STEP 6: 평가 및 시각화
- **STATUS**: TODO
- **OUTPUT**: `Sources/model/evaluate.py`, `Sources/pipeline_outputs/results/`
- **DESCRIPTION**:
  학습된 모델을 평가하고 논문과 동일한 메트릭/시각화를 생성한다.

### 평가 메트릭
1. **RMS Validation Error**: 5-fold 평균 ± 표준편차
   - 슈퍼복셀 예측 → 샘플별 최소값 취합 → 실제값과 비교
2. **Naive 예측 대비 감소율**: (RMSE_naive - RMSE_vppm) / RMSE_naive
3. **상대 오차**: RMSE / 관측 범위

### 예측 후처리
- 각 SS-J3 게이지 섹션에 여러 슈퍼복셀 예측이 존재
- 보수적 추정을 위해 **최소값(minimum)** 취합

### Out-of-Distribution 탐지
- 각 피처의 학습셋 min/max 범위를 벗어나면 → NaN 처리
- OOD 비율 보고

### 시각화 (논문 Figure 재현)
1. **Correlation plot** (Figure 17): 예측 vs 실측 2D 히스토그램
2. **Scatter plot** (Figure 18): 예측 vs 실측 산점도 (빌드별 색상)
3. **ROC-like curve** (Figure 16): 오차 임계값별 통과 비율
4. **Spatial prediction map** (Figure 20): 빌드 단면의 UTS 컬러맵

### 출력 파일
```
Sources/pipeline_outputs/results/
├── metrics_summary.json        # Table 7 재현
├── ablation_results.json       # Table 8 재현
├── correlation_plots.png       # Figure 17
├── scatter_plot_uts.png        # Figure 18
├── roc_curves.png              # Figure 16
└── spatial_prediction_B1.2.png # Figure 20
```

---

## STEP 7: 통합 실행 스크립트
- **STATUS**: TODO
- **OUTPUT**: `Sources/run_pipeline.py`
- **DESCRIPTION**:
  전체 파이프라인을 단일 스크립트로 실행할 수 있는 진입점을 만든다.

### CLI 인터페이스
```bash
# 전체 파이프라인 실행
python Sources/run_pipeline.py --all

# 특정 단계만 실행
python Sources/run_pipeline.py --step features --builds B1.2
python Sources/run_pipeline.py --step train --target UTS
python Sources/run_pipeline.py --step evaluate

# 단일 빌드 빠른 테스트
python Sources/run_pipeline.py --quick-test --builds B1.2
```

---

## 부록: 핵심 수치 참조

### 슈퍼복셀 통계 (논문)
- 총 6,299 샘플 → 29,680 슈퍼복셀 (평균 4.7 복셀/샘플)
- OOD 비율: ~0.05%

### 기대 성능 (논문 Table 7)
| 모델 | YS (MPa) | UTS (MPa) | UE (%) | TE (%) |
|------|----------|-----------|--------|--------|
| Full VPPM | 24.7±1.0 | 38.3±0.9 | 9.0±0.3 | 11.9±0.1 |
| Naive | 35.4±1.4 | 74.2±1.8 | 16.1±0.4 | 19.7±0.2 |

### 가우시안 블러 파라미터
- 커널 크기: 1.0 mm (≈ 7-8 픽셀, 정확히는 pixel_size_mm로 계산)
- 표준편차: 0.5 mm (≈ 3-4 픽셀)
