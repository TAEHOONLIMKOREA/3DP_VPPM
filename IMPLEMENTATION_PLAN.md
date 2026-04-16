# VPPM 재구현 설계 계획서

> **논문**: Scime, L. et al. "A Data-Driven Framework for Direct Local Tensile Property Prediction of Laser Powder Bed Fusion Parts" Materials 2023, 16, 7293

---

## 1. 재구현 범위

논문의 AIR (Augmented Intelligence Relay) 중 **N5 (슈퍼복셀 구축 + 피처 엔지니어링)** 및 **N6 (VPPM 모델)**을 재구현한다.

- N1~N4 (센서 데이터 공간 매핑, 이미지 보정, DSCNN 세그멘테이션)는 HDF5에 이미 전처리되어 저장되어 있으므로 **해당 결과를 직접 활용**
- N7~N8 (물리 모델, 응용 시험)은 범위 외

### 재구현 파이프라인 요약

```
HDF5 원본 데이터
    │
    ├─ slices/part_ids          ─┐
    ├─ slices/sample_ids         │
    ├─ slices/segmentation_results ─┤── N5: 슈퍼복셀 구축
    ├─ temporal/*                 │      + 21개 피처 벡터 생성
    ├─ scans/{layer}             │
    └─ samples/test_results     ─┘
                │
                ▼
        VPPM (Perceptron)  ── N6: 인장 특성 예측
                │
                ▼
        YS, UTS, UE, TE 예측값
```

---

## 2. 디렉토리 구조

```
Sources/
├── hdf5_parser/                  # 기존 HDF5 파서 (유지)
│   ├── ornl_data_loader.py
│   ├── example_usage.py
│   └── export_to_readable.py
│
└── vppm/                         # 새로 생성할 재구현 코드
    ├── config.py                 # 하이퍼파라미터 및 경로 설정
    ├── supervoxel.py             # Phase 1: 슈퍼복셀 그리드 구축
    ├── features.py               # Phase 2: 21개 피처 엔지니어링
    ├── dataset.py                # Phase 3: 데이터셋 구성 (정규화, CV 분할)
    ├── model.py                  # Phase 4: VPPM 퍼셉트론 모델
    ├── train.py                  # Phase 5: 학습 루프
    ├── evaluate.py               # Phase 6: 평가 및 시각화
    └── run_pipeline.py           # 전체 파이프라인 실행 진입점
```

---

## 3. Phase 1: 슈퍼복셀 그리드 구축 (`supervoxel.py`)

### 3.1 슈퍼복셀 정의

| 파라미터 | 값 | 근거 |
|---------|---|------|
| XY 크기 | 1.0 × 1.0 mm | 논문 Section 2.10 |
| Z 크기 | 3.5 mm (70 레이어) | 레이어 두께 50μm × 70 |
| 이미지 해상도 | ~130 μm/pixel | 논문 Section 2.2 |
| XY 픽셀 수 | ~7.7 pixels/mm | 1.0mm / 0.13mm |
| 빌드 영역 | 245 × 245 mm | 1842 × 1842 pixels |

### 3.2 구현 단계

```python
# 1. 빌드 볼륨을 고정 격자로 분할
#    - XY: 245mm / 1.0mm = 245 그리드 셀
#    - Z: num_layers / 70 = z_blocks 개
#    - 이미지 좌표계: slices/origin, slices/x-axis 참조

# 2. 각 슈퍼복셀이 CAD 형상(part_ids > 0)과 교차하는지 판별
#    - 슈퍼복셀 내 part_ids > 0인 픽셀이 존재해야 유효

# 3. 학습 대상: sample_ids > 0인 영역과 교차하는 슈퍼복셀만
#    - SS-J3 게이지 섹션과 10% 이상 겹쳐야 유효
#    - 각 슈퍼복셀에 대응하는 sample_id 기록
```

### 3.3 핵심 고려사항

- **메모리**: 각 레이어의 part_ids/sample_ids를 한 번에 전부 로드하지 않고, 70레이어씩 청크로 처리
- **좌표 변환**: 이미지 픽셀 좌표 ↔ 물리 좌표(mm) 변환 필수
  - `slices/origin`: 이미지 좌표 원점의 물리적 위치
  - 해상도: 물리 크기(245mm) / 픽셀 수(1842)

---

## 4. Phase 2: 21개 피처 엔지니어링 (`features.py`)

### 4.1 피처 목록 및 계산 방법

#### 그룹 A: CAD 기반 피처 (3개)

| # | 피처명 | 계산 방법 |
|---|--------|---------|
| 1 | `distance_from_edge` | 각 레이어에서 part_ids 바이너리 마스크에 대해 **거리 변환(distance transform)** 수행. 슈퍼복셀 내 평균 |
| 2 | `distance_from_overhang` | 현재 레이어에서 파트가 존재하지만 **아래 레이어에서는 없는** 픽셀까지의 거리. 거리 변환 후 슈퍼복셀 내 평균 |
| 3 | `build_height` | 슈퍼복셀 z-중심의 빌드 플레이트로부터 높이 = `layer_center × 0.05mm` |

```python
from scipy.ndimage import distance_transform_edt

# distance_from_edge: 각 레이어
part_mask = (part_ids[layer] > 0).astype(float)
dist_edge = distance_transform_edt(part_mask) * pixel_size_mm
# 슈퍼복셀 영역 내 평균

# distance_from_overhang: 
overhang_mask = (part_ids[layer] > 0) & (part_ids[layer-1] == 0)
if overhang_mask.any():
    dist_overhang = distance_transform_edt(~overhang_mask) * pixel_size_mm
else:
    dist_overhang = np.full_like(part_mask, np.inf)  # 오버행 없음
```

#### 그룹 B: DSCNN 세그멘테이션 기반 피처 (8개)

| # | 피처명 | HDF5 키 | 계산 방법 |
|---|--------|---------|---------|
| 4 | `seg_powder` | `segmentation_results/0` | 슈퍼복셀 내 CAD 영역 픽셀 중 해당 클래스 비율 |
| 5 | `seg_printed` | `segmentation_results/1` | 동일 |
| 6 | `seg_recoater_streaking` | `segmentation_results/3` | 동일 |
| 7 | `seg_edge_swelling` | `segmentation_results/5` | 동일 |
| 8 | `seg_debris` | `segmentation_results/6` | 동일 |
| 9 | `seg_super_elevation` | `segmentation_results/7` | 동일 |
| 10 | `seg_soot` | `segmentation_results/8` | 동일 |
| 11 | `seg_excessive_melting` | `segmentation_results/10` | 동일 |

**주의**: 논문의 DSCNN은 8개 클래스를 사용하지만, HDF5에는 12개 클래스가 저장됨. 논문에서 사용한 8개 클래스에 해당하는 ID를 매핑해야 함:

```
논문 클래스        → HDF5 segmentation_results ID
Powder            → 0
Printed           → 1
Recoater streaking → 3
Edge swelling     → 5 (Swelling)
Debris            → 6
Super-elevation   → 7
Soot              → 8 (Spatter)
Excessive melting → 10 (Over Melting)
```

```python
# 각 레이어, 각 슈퍼복셀 영역에서:
# 1. CAD 마스크 내 픽셀만 대상
# 2. 각 DSCNN 클래스의 확률/예측값의 평균 계산
# 3. 70개 레이어에 걸쳐 가중 평균 (CAD 교차 픽셀 수 비례)
```

#### 그룹 C: 프린터 로그 기반 피처 (7개)

| # | 피처명 | HDF5 키 | 비고 |
|---|--------|---------|------|
| 12 | `layer_print_time` | `temporal/layer_times` | 슈퍼복셀이 걸친 70개 레이어의 평균 |
| 13 | `top_gas_flow_rate` | `temporal/top_flow_rate` | 동일 |
| 14 | `bottom_gas_flow_rate` | `temporal/bottom_flow_rate` | 동일 |
| 15 | `module_oxygen` | `temporal/module_oxygen` | 동일 |
| 16 | `build_plate_temperature` | `temporal/build_plate_temperature` | 동일 |
| 17 | `bottom_flow_temperature` | `temporal/bottom_flow_temperature` | 동일 |
| 18 | `actual_ventilator_flow_rate` | `temporal/actual_ventilator_flow_rate` | 동일 |

```python
# 시간적 데이터는 레이어당 1개 스칼라값
# 슈퍼복셀의 z범위(70레이어)에 해당하는 값들의 평균
temporal_data = hdf5['temporal/layer_times'][layer_start:layer_end]
feature_value = np.mean(temporal_data)
```

#### 그룹 D: 레이저 스캔 경로 기반 피처 (3개)

| # | 피처명 | 계산 방법 |
|---|--------|---------|
| 19 | `laser_module` | 슈퍼복셀 영역 내 스캔 벡터의 레이저 모듈 (0 또는 1). 과반수 투표 |
| 20 | `laser_return_delay` | 인접 래스터 패스 간 시간 지연의 평균 |
| 21 | `laser_stripe_boundaries` | 슈퍼복셀이 스트라이프 경계에 위치하는지 인코딩 |

```python
# scans/{layer}: [x_start, x_end, y_start, y_end, relative_time]
# 슈퍼복셀 XY 영역과 교차하는 스캔 벡터만 필터링

# laser_module: HDF5에 레이저 모듈 정보가 별도 저장되어 있는지 확인 필요
#   parts/process_parameters/laser_module 활용 가능

# laser_return_delay: 연속된 스캔 벡터의 time 차이
#   같은 래스터 방향의 인접 벡터 간 시간차 계산

# laser_stripe_boundaries: 스캔 경로의 x 또는 y 불연속점
#   stripe_width (parts/process_parameters/stripe_width) 활용
```

### 4.2 슈퍼복셀 집계 방법

논문의 집계 규칙:
1. **XY 평면**: 슈퍼복셀 영역 내, CAD 형상과 교차하는 픽셀만 대상으로 평균
2. **Z 방향**: 70개 레이어에 걸쳐 슬라이딩 윈도우 방식으로 평균
3. **가중치**: 각 레이어에서 슈퍼복셀-CAD 교차 픽셀 수에 비례

```python
# 레이어 l에서 슈퍼복셀 (i,j)의 피처 계산:
sv_mask = get_supervoxel_mask(i, j)          # 슈퍼복셀 XY 영역
cad_mask = (part_ids[layer] > 0)             # CAD 형상
valid_mask = sv_mask & cad_mask              # 교차 영역
n_valid = valid_mask.sum()

if n_valid > 0:
    feature_at_layer = data[layer][valid_mask].mean()
    weight = n_valid
```

---

## 5. Phase 3: 데이터셋 구성 (`dataset.py`)

### 5.1 학습 데이터 구성

```
입력 (X): [N_supervoxels, 21] — 21개 정규화된 피처
출력 (Y): [N_supervoxels, 1]  — 정규화된 인장 특성 (모델별)
```

- **유효 슈퍼복셀**: SS-J3 게이지 섹션(sample_ids > 0)과 10% 이상 교차
- **라벨**: 해당 sample_id의 인장 시험 결과 (YS, UTS, UE, TE)
- 하나의 시편(sample)이 평균 4.7개의 슈퍼복셀과 대응
- **전체**: 6,299 시편 → 약 29,680 슈퍼복셀

### 5.2 정규화

```python
# Zero-center normalization to [-1, 1]
# 논문 Section 2.10: min-max 기반 zero-center

X_min = X_train.min(axis=0)  # 피처별 최소값
X_max = X_train.max(axis=0)  # 피처별 최대값
X_norm = 2.0 * (X - X_min) / (X_max - X_min + eps) - 1.0

# 라벨(인장 특성)도 동일하게 정규화
Y_min = Y_train.min(axis=0)
Y_max = Y_train.max(axis=0)
Y_norm = 2.0 * (Y - Y_min) / (Y_max - Y_min + eps) - 1.0
```

### 5.3 교차 검증 분할

```python
# 5-Fold Cross Validation
# **핵심**: sample-wise 분할 (같은 시편의 모든 슈퍼복셀은 같은 fold)
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
unique_sample_ids = np.unique(sample_ids_per_supervoxel)

for fold, (train_idx, val_idx) in enumerate(kf.split(unique_sample_ids)):
    train_samples = unique_sample_ids[train_idx]  # 80%
    val_samples = unique_sample_ids[val_idx]       # 20%
    
    # 슈퍼복셀을 sample 기준으로 분할
    train_mask = np.isin(sample_ids_per_supervoxel, train_samples)
    val_mask = np.isin(sample_ids_per_supervoxel, val_samples)
```

### 5.4 Out-of-Distribution 탐지

```python
# 학습 데이터의 피처별 min/max 범위를 벗어나는 슈퍼복셀 → NaN 처리
ood_mask = (X < X_train_min) | (X > X_train_max)
ood_supervoxels = ood_mask.any(axis=1)
# 논문: 전체의 약 0.05%만 해당
```

---

## 6. Phase 4: VPPM 모델 (`model.py`)

### 6.1 아키텍처

```python
import torch
import torch.nn as nn

class VPPM(nn.Module):
    """Voxelized Property Prediction Model (Perceptron)
    
    논문 Table 6:
    - FC(n_feats → 128)
    - Dropout(0.1)
    - FC(128 → 1)
    """
    def __init__(self, n_feats=21, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_feats, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # 초기화: 정규분포 (std=0.1)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.fc2.bias, mean=0.0, std=0.1)
    
    def forward(self, x):
        x = self.fc1(x)      # 활성화 함수 없음 (논문에 명시 없음 — 확인 필요)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 6.2 활성화 함수 관련 불확실성

논문(Table 6)에 활성화 함수가 명시되지 않음. 가능성:
1. **활성화 없음** (선형 모델) — "perceptron"이라는 표현과 부합
2. **ReLU** — 일반적인 MLP에서의 관례

→ **계획**: 둘 다 실험하여 논문 결과에 가까운 쪽 채택

### 6.3 4개 독립 모델

```python
# 각 인장 특성에 대해 별도 VPPM 학습
properties = ['yield_strength', 'ultimate_tensile_strength', 
              'uniform_elongation', 'total_elongation']

models = {prop: VPPM(n_feats=21) for prop in properties}
```

---

## 7. Phase 5: 학습 (`train.py`)

### 7.1 학습 설정

| 파라미터 | 값 | 근거 |
|---------|---|------|
| 옵티마이저 | Adam | 논문 Section 2.11 |
| Learning Rate | 1 × 10⁻⁸ | 논문 (매우 작음에 주의) |
| Beta1, Beta2 | 0.9, 0.999 | 논문 |
| Epsilon | 1 × 10⁻⁴ | 논문 |
| 손실 함수 | L1 (MAE) | 논문 Section 2.11 |
| 배치 크기 | 1000 | 논문 |
| 초기화 | N(0, 0.1) | 논문 |
| 조기 종료 | 검증 오차 plateau 시 | 논문 |

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-8,
    betas=(0.9, 0.999),
    eps=1e-4
)
criterion = nn.L1Loss()
```

### 7.2 학습 루프

```python
for epoch in range(max_epochs):
    model.train()
    for batch_X, batch_Y in train_loader:  # batch_size=1000
        pred = model(batch_X)
        loss = criterion(pred, batch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 검증
    model.eval()
    val_rmse = compute_rmse(model, val_loader)
    
    # 조기 종료: plateau 감지
    if early_stopper.check(val_rmse):
        break
```

### 7.3 조기 종료 전략

```python
class EarlyStopper:
    """검증 정확도 plateau 감지"""
    def __init__(self, patience=50, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('inf')
        self.counter = 0
    
    def check(self, val_score):
        if val_score < self.best_score - self.min_delta:
            self.best_score = val_score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
```

---

## 8. Phase 6: 평가 (`evaluate.py`)

### 8.1 평가 메트릭

```python
# 1. RMS 오차 (primary metric)
rmse = sqrt(mean((pred - true)^2))

# 2. Naive 예측 대비 감소율
naive_rmse = sqrt(mean((mean(true) - true)^2))
reduction = naive_rmse - vppm_rmse

# 3. 내재 측정 오차 분리 후 감소율 (논문 Eq. 4-5)
# RMSE_VPPM = RMSE_I + RMSE_M-VPPM
# RMSE_naive = RMSE_I + RMSE_M-naive
# 측정 오차: YS=16.6MPa, UTS=15.6MPa, UE=1.73%, TE=2.92%
measurement_error = {'YS': 16.6, 'UTS': 15.6, 'UE': 1.73, 'TE': 2.92}
```

### 8.2 재현 목표

| 메트릭 | YS (MPa) | UTS (MPa) | UE (%) | TE (%) |
|--------|----------|-----------|--------|--------|
| **논문 VPPM** | **24.7 ± 1.0** | **38.3 ± 0.9** | **9.0 ± 0.3** | **11.9 ± 0.1** |
| 논문 Naive | 35.4 ± 1.4 | 74.2 ± 1.8 | 16.1 ± 0.4 | 19.7 ± 0.2 |

### 8.3 시각화

1. **Correlation Plot**: 예측값 vs 실측값 2D 히스토그램 (논문 Fig. 17 재현)
2. **Scatter Plot**: 빌드별 색상 구분 (논문 Fig. 18 재현)
3. **ROC-like Curve**: 오차 임계값 vs 샘플 비율 (논문 Fig. 16 재현)
4. **Ablation 결과**: 피처 그룹별 성능 비교 바 차트 (논문 Table 8 재현)

### 8.4 테스트 시 추론 방식

```python
# 하나의 시편에 대응하는 복수의 슈퍼복셀 예측값 → 최소값 사용
# 논문: "Taking the minimum value provides a conservative estimate"
per_sample_predictions = {}
for sv_idx, sample_id in enumerate(supervoxel_sample_ids):
    pred_denorm = denormalize(model(X[sv_idx]))
    if sample_id not in per_sample_predictions:
        per_sample_predictions[sample_id] = []
    per_sample_predictions[sample_id].append(pred_denorm)

final_predictions = {
    sid: min(preds) for sid, preds in per_sample_predictions.items()
}
```

---

## 9. 구현 순서 및 의존성

```
Phase 1: supervoxel.py     ← HDF5 데이터 (part_ids, sample_ids)
    │
    ▼
Phase 2: features.py       ← Phase 1 + HDF5 (segmentation, temporal, scans)
    │
    ▼
Phase 3: dataset.py        ← Phase 2 + HDF5 (test_results)
    │
    ▼
Phase 4: model.py          ← (독립)
    │
    ▼
Phase 5: train.py          ← Phase 3 + Phase 4
    │
    ▼
Phase 6: evaluate.py       ← Phase 5
```

### 예상 실행 순서

```bash
# 1. 슈퍼복셀 그리드 구축 + 피처 계산 (가장 오래 걸림)
python -m Sources.vppm.run_pipeline --phase features --build all

# 2. 데이터셋 준비 (정규화 + CV 분할)
python -m Sources.vppm.run_pipeline --phase dataset

# 3. 학습 (4개 특성 × 5 fold = 20회 학습)
python -m Sources.vppm.run_pipeline --phase train

# 4. 평가
python -m Sources.vppm.run_pipeline --phase evaluate
```

---

## 10. 기술적 챌린지 및 대응 방안

### 10.1 메모리 관리 (가장 큰 도전)

| 문제 | 대응 |
|------|------|
| HDF5 파일당 40~54GB | `h5py` 슬라이싱으로 레이어 단위 접근 |
| 5개 빌드 × 수천 레이어 | z-block(70레이어) 단위 청크 처리 |
| 세그멘테이션 12채널 × full resolution | 필요한 8클래스만 로드, 슈퍼복셀 영역만 크롭 |
| 전체 슈퍼복셀 피처 저장 | 빌드별로 `.npz` 파일에 중간 결과 캐싱 |

```python
# 메모리 효율적 처리 패턴
with h5py.File(path, 'r') as f:
    for z_start in range(0, num_layers, 70):
        z_end = min(z_start + 70, num_layers)
        # 이 z-block의 데이터만 로드
        part_ids_block = f['slices/part_ids'][z_start:z_end]
        # 처리 후 메모리 해제
        del part_ids_block
```

### 10.2 좌표계 정합

```python
# 이미지 픽셀 → 물리 좌표 변환
# slices/origin: 이미지 (0,0)의 물리적 위치
# 이미지 크기: 1842 × 1842 pixels → 245 × 245 mm
pixel_size = 245.0 / 1842  # ≈ 0.133 mm/pixel

def pixel_to_mm(px, py, origin):
    x_mm = origin[0] + px * pixel_size
    y_mm = origin[1] + py * pixel_size
    return x_mm, y_mm

def mm_to_pixel(x_mm, y_mm, origin):
    px = int((x_mm - origin[0]) / pixel_size)
    py = int((y_mm - origin[1]) / pixel_size)
    return px, py
```

### 10.3 레이저 스캔 경로 피처 (가장 어려운 피처)

- `laser_module`: `parts/process_parameters/laser_module`에서 파트별 정보를 가져와, 슈퍼복셀 영역의 part_id로 매핑
- `laser_return_delay`: 스캔 경로의 시간 데이터에서 인접 벡터 간 시간차 계산 필요
- `laser_stripe_boundaries`: `parts/process_parameters/stripe_width`와 스캔 경로의 공간적 불연속을 탐지

→ **우선순위 낮음**: Ablation 결과에서 CAD+스캔 경로만으로는 naive와 유사한 성능. DSCNN 피처가 핵심이므로, 스캔 경로 피처는 후순위로 구현

### 10.4 처리 시간 예측

| 단계 | 예상 시간 | 병목 |
|------|----------|------|
| 피처 계산 (전체 5빌드) | 12~24시간 | HDF5 I/O, 거리 변환 |
| 데이터셋 준비 | 수 분 | 정규화, 분할 |
| VPPM 학습 (20회) | ~100분 | GPU 불필요 (얕은 모델) |
| 평가 | 수 분 | 추론 + 시각화 |

---

## 11. Ablation Study 계획

논문 Table 8을 재현하기 위한 피처 그룹별 실험:

| 실험 | 사용 피처 | 피처 수 | 논문 UTS RMSE |
|------|---------|---------|-------------|
| Full-featured VPPM | 모든 피처 | 21 | 38.3 ± 0.9 |
| CAD + Scan path | #1-3, #19-21 | 6 | 76.3 ± 2.1 |
| Printer log file | #12-18 | 7 | 73.5 ± 1.9 |
| DSCNN only | #4-11 | 8 | 40.6 ± 0.8 |
| 20% training data | 모든 피처 | 21 | 38.4 ± 0.9 |

---

## 12. 추가 패키지 요구사항

```
# requirements.txt에 추가 필요
torch>=2.0
scikit-learn>=1.0
scipy>=1.7
tqdm
```

---

## 13. 단계별 검증 포인트

각 Phase 완료 시 아래를 확인하여 중간 검증:

| Phase | 검증 포인트 |
|-------|-----------|
| 1 | B1.2에서 슈퍼복셀 수 ≈ 빌드 볼륨 내 유효 격자 수 확인 |
| 2 | 피처값 범위가 물리적으로 합리적인지 (예: build_height가 0~180mm 범위) |
| 2 | DSCNN 피처가 파라미터 세트별로 다른 분포를 보이는지 |
| 3 | 학습 슈퍼복셀 수 ≈ 29,680개 (논문 수치) |
| 3 | 시편당 평균 슈퍼복셀 수 ≈ 4.7개 |
| 5 | Naive RMSE가 논문 값(UTS: 74.2 MPa)과 유사한지 |
| 5 | 학습 손실이 수렴하는지 (fold당 5분 이내) |
| 6 | VPPM RMSE가 논문 값과 ±10% 이내인지 |

---

## 14. 간소화 구현 전략 (MVP)

전체 구현이 복잡하므로, 아래 순서로 **점진적으로 구현**:

### MVP 1: DSCNN 피처만으로 VPPM (최소 구현)
- 피처 #4-11 (8개 DSCNN 세그멘테이션 비율)만 사용
- 슈퍼복셀 대신 **시편(sample) 단위**로 세그멘테이션 통계 집계
- 이것만으로도 논문 UTS RMSE 40.6 MPa에 근접 가능

### MVP 2: + 시간적 데이터 피처 추가
- 피처 #12-18 (7개 프린터 로그) 추가
- 시편이 걸친 레이어 범위에서 평균

### MVP 3: + CAD 피처 추가
- 피처 #1-3 (거리 변환 기반)
- 완전한 슈퍼복셀 그리드 구현

### MVP 4: + 스캔 경로 피처 추가 (Full)
- 피처 #19-21
- 논문과 동일한 21개 피처 완성

각 MVP에서 성능을 측정하여 논문 결과와 비교하며 진행.
