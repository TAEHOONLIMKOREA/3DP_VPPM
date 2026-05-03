# ORNL L-PBF Tensile Property Dataset (Peregrine v2023-11)

## 개요

이 데이터셋은 Oak Ridge National Laboratory (ORNL)의 Manufacturing Demonstration Facility (MDF)에서 수행된 Laser Powder Bed Fusion (L-PBF) 적층제조 공정의 in-situ 센서 데이터와 인장 시험 결과를 포함합니다.

- **논문**: "A Data-Driven Framework for Direct Local Tensile Property Prediction of Laser Powder Bed Fusion Parts" (Materials 2023, 16, 7293)
- **데이터셋 크기**: 약 230 GB
- **파일 형식**: HDF5
- **DOI**: https://doi.ccs.ornl.gov/ui/doi/452
- **연락처**: Dr. Luke Scime (scimelr@ornl.gov)

---

## 데이터셋 다운로드 방법

데이터셋은 Globus를 통해 다운로드할 수 있습니다:

1. Globus 계정 생성
2. 컴퓨터에 Globus Endpoint 생성 (필요시 바이러스 백신 예외 설정)
3. **OLCF DOI-DOWNLOADS** Collection에서 사용자 Collection으로 전송
4. 필요시 Preferences > Access 탭에서 Globus 접근 디렉토리 수동 생성
5. 문의: Dr. Luke Scime (scimelr@ornl.gov)

---

## 실험 조건

### 프린터 및 재료
| 항목 | 상세 |
|------|------|
| 프린터 | Concept Laser M2 (General Electric Additive) |
| 레이저 | 2 x 400W 레이저 모듈 |
| 리코터 | Compliant recoater blade |
| 재료 | SS 316L 스테인리스 스틸 분말 (TruForm, Praxair) |
| 분말 입도 | D10: 20μm, D50: 31μm, D90: 43μm |
| 레이어 두께 | 50 μm |
| 쉴드 가스 | 아르곤 (Argon) |

### 분말 화학 조성 (wt%)
| C | Co | Cr | Cu | Fe | Mn | Mo | N | Ni | O | P | S | Si |
|---|----|----|----|----|----|----|---|----|---|---|---|-----|
| <0.005 | 0.08 | 17.01 | 0.00 | Bal | 1.29 | 2.48 | 0.01 | 12.67 | 0.03 | <0.005 | 0.005 | 0.59 |

---

## 파일 목록 및 빌드 정보

| 파일명 | 빌드 ID | 샘플 수 | 크기 | 주요 특성 |
|--------|---------|---------|------|----------|
| 2021-07-13 TCR Phase 1 Build 1.hdf5 | B1.1 | 503 | ~52 GB | 기준 공정 조건 |
| 2021-04-16 TCR Phase 1 Build 2.hdf5 | B1.2 | 2,705 | ~43 GB | 다양한 공정 파라미터 |
| 2021-04-28 TCR Phase 1 Build 3.hdf5 | B1.3 | 813 | ~46 GB | 오버행(Overhang) 형상 |
| 2021-08-03 TCR Phase 1 Build 4.hdf5 | B1.4 | 694 | ~54 GB | 스패터/아르곤 가스 유량 변화 |
| 2021-08-23 TCR Phase 1 Build 5.hdf5 | B1.5 | 1,584 | ~52 GB | 리코터 손상/분말 공급 부족 |

**총 인장 시험 샘플: 6,299개**

---

## HDF5 파일 구조

HDF5는 대용량 데이터 처리에 최적화된 계층적 자기 기술(self-describing) 데이터 형식입니다. 폴더 디렉토리와 유사하게 탐색할 수 있으며, 각 데이터에 메타데이터를 첨부할 수 있습니다.

### 3.1 Build Metadata (빌드 메타데이터)

빌드 전체에 적용되는 메타데이터:

| 키 | 설명 |
|-----|------|
| `core/` | 빌드 이름, 노트 등 고유 식별 정보 |
| `layer_notes/#` | 특정 프린트 레이어의 사용자 노트 (# = 레이어 번호) |
| `log_file/` | 프린터가 생성한 로그 파일 원본 텍스트 |
| `material/` | 재료 조성 및 레이어 두께 정보 |
| `people/` | 파트 요청자 및 제조자 정보 |
| `printer/` | 프린터 정보 및 물리적 치수 |
| `specimens/` | 프린트된 파트 및 추출 샘플 정보 |
| `user_defined/` | 프린터별 사용자 정의 메타데이터 필드 |

```python
import h5py
with h5py.File(path_to_hdf5_file, 'r') as build:
    print(build.attrs['core/build_name'])
```

### 3.2 Reference Images (참조 이미지)

빌드 전/중/후에 프린터 운영자가 촬영한 참조 이미지:

```python
import h5py
import matplotlib.pyplot as plt
with h5py.File(path_to_hdf5_file, 'r') as build:
    plt.imshow(build['reference_images/thumbnail'][...], interpolation='none')
```

### 3.3 Slice Data (슬라이스 데이터)

레이어별로 인덱싱된 2D 배열 형태의 데이터:

| 키 | 설명 |
|-----|------|
| `camera_data/visible/0` | 용융 직후 촬영된 5MP 가시광선 이미지 |
| `camera_data/visible/1` | 분말 도포 직후 촬영된 5MP 가시광선 이미지 |
| `slices/part_ids` | 각 픽셀의 파트 ID (0 = 파트 외부) |
| `slices/sample_ids` | 각 픽셀의 샘플 ID (0 = 샘플 외부) |
| `slices/segmentation_results/#` | DSCNN 이상 예측 결과 (# = 클래스 ID) |

좌표계 정보: `slices/origin`, `slices/x-axis` 등

```python
import h5py
import matplotlib.pyplot as plt
with h5py.File(path_to_hdf5_file, 'r') as build:
    plt.imshow(build['slices/camera_data/visible/0'][layer_number,...],
               cmap='jet', interpolation='none')
```

### 3.4 Temporal Data (시간적 데이터)

로그 파일에서 추출한 프린터 상태 센서 데이터 (레이어별 1D 배열):

| 키 | 설명 |
|-----|------|
| `temporal/absolute_image_capture_timestamp` | 각 레이어 UTC 타임스탬프 |
| `temporal/actual_ventilator_flow_rate` | 환기 장치 아르곤 유량 |
| `temporal/bottom_chamber_temperature` | 빌드 챔버 하부 온도 |
| `temporal/bottom_flow_rate` | 챔버 하부 아르곤 유량 |
| `temporal/bottom_flow_temperature` | 하부 아르곤 온도 |
| `temporal/build_chamber_position` | 빌드 플랫폼 높이 |
| `temporal/build_plate_temperature` | 빌드 플레이트 온도 |
| `temporal/build_time` | 빌드 시작 이후 누적 시간 |
| `temporal/gas_loop_oxygen` | 가스 루프 산소 농도 |
| `temporal/glass_scale_temperature` | 레이저 광학계 온도 |
| `temporal/laser_rail_temperature` | 레이저 광학계 온도 |
| `temporal/layer_times` | 각 레이어 프린트 소요 시간 |
| `temporal/module_oxygen` | 빌드 챔버 산소 농도 |
| `temporal/powder_chamber_position` | 분말 도징 플랫폼 높이 |
| `temporal/target_ventilator_flow_rate` | 환기 장치 유량 설정값 |
| `temporal/top_chamber_temperature` | 빌드 챔버 상부 온도 |
| `temporal/top_flow_rate` | 챔버 상부 아르곤 유량 |
| `temporal/top_flow_temperature` | 상부 아르곤 온도 |
| `temporal/ventilator_speed` | 환기 장치 속도 |

```python
import h5py
import matplotlib.pyplot as plt
with h5py.File(path_to_hdf5_file, 'r') as build:
    plt.scatter(np.arange(build['temporal/top_flow_rate'].shape[0]),
                build['temporal/top_flow_rate'])
```

### 3.5 Scan Path Data (스캔 경로 데이터)

QM Meltpool 시스템에서 복구된 레이저 스캔 경로:

- 각 레이어별 5열 배열 (행 수 = 스캔 벡터 수)
- **열 0-1**: 시작/끝 X 좌표
- **열 2-3**: 시작/끝 Y 좌표
- **열 4**: 상대 프린트 시간
- Raster 및 Contour 스캔 경로만 포함 (빔 터너라운드, 점프 라인 제외)

```python
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.collections as collections

with h5py.File(path_to_hdf5_file, 'r') as build:
    x = build['scans/%i' %(layer_number)][:,0:2]
    y = build['scans/%i' %(layer_number)][:,2:4]
    t = build['scans/%i' %(layer_number)][:,4]
    colorizer = cm.ScalarMappable(norm=mcolors.Normalize(np.min(t),np.max(t)),
                                  cmap='jet')
    line_collection = collections.LineCollection(np.stack([x,y],axis=-1),
                                                  colors=colorizer.to_rgba(t))
    fig = plt.figure('scan paths')
    ax = fig.add_subplot()
    plt.axis('scaled')
    ax.set_xlim(x.min(),x.max())
    ax.set_ylim(y.min(),y.max())
    ax.add_collection(line_collection)
```

### 3.6 Part and Sample Data (파트 및 샘플 데이터)

파트 ID 또는 샘플 ID로 인덱싱된 1D 배열:

**공정 파라미터:**
| 키 | 설명 |
|-----|------|
| `parts/process_parameters/hatch_spacing` | 레이저 래스터 트랙 간격 |
| `parts/process_parameters/laser_beam_power` | 래스터 레이저 빔 출력 |
| `parts/process_parameters/laser_beam_speed` | 래스터 레이저 빔 속도 |
| `parts/process_parameters/laser_module` | 사용된 레이저 모듈 (1 또는 2) |
| `parts/process_parameters/laser_spot_size` | 래스터 레이저 빔 스팟 크기 |
| `parts/process_parameters/parameter_set` | 파라미터 세트 이름 |
| `parts/process_parameters/scan_rotation` | 레이어별 스캔 전략 회전 |
| `parts/process_parameters/stripe_width` | 래스터 스트라이프 폭 |

**시험 결과:**
| 키 | 설명 |
|-----|------|
| `parts/test_results/yield_strength` | 항복강도 (YS) |
| `parts/test_results/ultimate_tensile_strength` | 인장강도 (UTS) |
| `parts/test_results/uniform_elongation` | 균일연신율 (UE) |
| `parts/test_results/total_elongation` | 총연신율 (TE) |
| `parts/test_results/burst_pressure` | 튜브 파열 압력 |
| `parts/test_results/burst_temperature` | 튜브 파열 온도 |

```python
import h5py
with h5py.File(path_to_hdf5_file, 'r') as build:
    key = 'samples/test_results/ultimate_tensile_strength'
    for i in range(1, min(build[key].shape[0], 10), 1):
        print(' %i ' %(i), build[key][i])
```

---

## 빌드별 상세 설명

### B1.1 - 기준 공정 조건
- **목적**: 기본 공정 조건에서의 인장 특성 측정
- **특징**:
  - 모든 파트에 nominal 공정 파라미터 적용
  - 다양한 파트 형상 (SSJ3-A, SSJ3-B, SSJ3-D)
  - 빌드 볼륨 내 위치별 변동성 평가
  - 레이저 모듈 변경에 따른 영향 평가
- **컨투어**: 활성화

### B1.2 - 다양한 공정 파라미터
- **목적**: 공정 파라미터 변화에 따른 인장 특성 변화 분석
- **특징**:
  - 4가지 공정 파라미터 세트 적용 (Nominal, Best, LOF, Keyhole)
  - SSJ3-D 파트 형상 사용
  - 가장 많은 샘플 수 (2,705개)
- **컨투어**: 비활성화

### B1.3 - 오버행 형상
- **목적**: 오버행 구조물의 인장 특성 평가
- **특징**:
  - 30° 오버행 각도 (z축 기준)
  - SSJ3-A, SSJ3-B, SSJ3-C 파트 형상 사용
  - 오버행 인접 표면의 특성 저하 분석
- **컨투어**: 활성화

### B1.4 - 스패터 및 가스 유량 변화
- **목적**: 스패터 입자와 아르곤 가스 유량 변화의 영향 평가
- **특징**:
  - Soot 파라미터 파트 포함 (스패터 생성용)
  - 아르곤 유량 25-40 m³/h 범위 변화
  - SSJ3-A, SSJ3-B, SSJ3-C 파트 형상
- **컨투어**: 활성화

### B1.5 - 리코터 손상 및 분말 공급 부족
- **목적**: 리코터 블레이드 손상과 분말 공급 부족의 영향 평가
- **특징**:
  - 의도적 손상된 리코터 블레이드 사용
  - 분말 도징 팩터 5-200% 변화
  - 인공적 분말 공급 부족 유도
- **컨투어**: 활성화

---

## 공정 파라미터 세트

| 파라미터 세트 | 레이저 출력 (W) | 스캔 속도 (mm/s) | 해치 간격 (μm) | 레이저 스팟 크기 (μm) | 스트라이프 폭 (mm) | 스캔 회전 (°/층) |
|--------------|---------------|-----------------|--------------|-------------------|----------------|---------------|
| **Nominal** | 370 | 1,350 | 90 | 130 | 10 | 67 |
| **Best** | 380 | 800 | 110 | 125 | 18 | 67 |
| **LOF** (Lack-of-Fusion) | 290 | 1,200 | 150 | 50 | 18 | 67 |
| **Keyhole** | 290 | 800 | 70 | 125 | 18 | 67 |
| **Soot** | 290 | 1,200 | 70 | 50 | 18 | 90 |

### 파라미터 특성
- **Best**: 기공(porosity) 최소화를 위해 최적화됨
- **LOF**: Lack-of-fusion 기공 유발 (저에너지 밀도)
- **Keyhole**: Keyhole 기공 유발 (고에너지 밀도)
- **Soot**: 주변 파트에 스패터 입자 증착 유도

---

## 인장 시편 형상

### SS-J3 시편 규격
SS-J3 subsize 인장 시편 사용 (ORNL 표준 금속 조사 시편 규격)

| 치수 | 값 |
|------|-----|
| 전체 길이 | 16.0 mm |
| 게이지 길이 | 5.0 mm |
| 게이지 폭 | 1.20 mm |
| 게이지 두께 | 0.75 mm |
| 그립 폭 | 4.0 mm |
| 필렛 반경 | 1.4 mm |

### 파트 형상 (인장 시편 추출용)
| 형상 | 샘플 수/파트 | 벽 두께 | 특징 |
|------|------------|---------|------|
| SSJ3-A | 24 | 0.75 mm | As-printed 표면만 |
| SSJ3-B | 24 | 1.5 mm | 기계가공 표면만 |
| SSJ3-C | 200 | 5.0 mm | As-printed + 기계가공 |
| SSJ3-D | 576 | 40 mm | As-printed + 기계가공 (대형 블록) |

---

## 열처리 조건

모든 빌드는 동일한 응력 완화 열처리를 거침:

| 단계 | 설명 |
|------|------|
| 1 | 10°C/min으로 650±10°C까지 승온 |
| 2 | 650±10°C에서 24±0.5시간 유지 |
| 3 | 100±20°C까지 노냉 |

---

## DSCNN 이상 분류 클래스

딥러닝 기반 이상 탐지 모델(DSCNN)이 분류하는 12가지 클래스:

| ID | 클래스 | 설명 |
|----|--------|------|
| 0 | Powder | 이상 또는 프린트된 파트가 없는 분말 베드 영역 |
| 1 | Printed | 이상이 감지되지 않은 프린트 영역 |
| 2 | Recoater Hopping | 리코터가 표면 아래 파트에 충돌할 때 발생하는 물결무늬 |
| 3 | Recoater Streaking | 리코터 손상 또는 큰 입자 끌림으로 인한 줄무늬 |
| 4 | Incomplete Spreading | 분말 베드에 불충분한 분말 도포 |
| 5 | Swelling | 분말 위로 돌출된 프린트 재료의 변형/뒤틀림 |
| 6 | Debris | 분말 베드의 소-중형 교란 (포괄적 클래스) |
| 7 | Super-Elevation | 프린트 영역 위의 분말 커버리지 부족 |
| 8 | Spatter | 용접 풀에서 튀어나와 분말 베드에 착지한 비산물 |
| 9 | Misprint | 의도된 파트 형상 외부에서 감지된 프린트 재료 |
| 10 | Over Melting | 고에너지 밀도 공정 파라미터로 용융된 영역 |
| 11 | Under Melting | 저에너지 밀도 공정 파라미터로 용융된 영역 |

클래스 이름 목록은 `slices/segmentation_results/class_names` 키로 조회 가능

---

## 인장 시험 결과 범위

| 특성 | 최소값 | 최대값 | 단위 |
|------|--------|--------|------|
| 항복강도 (YS) | 70 | 420 | MPa |
| 인장강도 (UTS) | 80 | 610 | MPa |
| 균일연신율 (UE) | 0 | 69 | % |
| 총연신율 (TE) | 4 | 94 | % |

### 참조값
| 소스 | YS (MPa) | UTS (MPa) | UE (%) | TE (%) |
|------|----------|-----------|--------|--------|
| ASTM A240 | 170 | 480 | 40 | N/A |
| Wrought SS 316L | 261 | 562 | 66.0 | 72.8 |

---

## 데이터 활용

이 데이터셋은 다음 연구에 활용 가능:
- L-PBF 공정의 인장 특성 예측 모델 개발
- 공정 파라미터-구조-특성 관계 분석
- In-situ 모니터링 기반 품질 관리 알고리즘 개발
- 디지털 트윈 구축
- 적층제조 부품 인증 프레임워크 연구

---

## 참고문헌

1. Scime, L.; Joslin, C.; Collins, D.A.; Sprayberry, M.; Singh, A.; Halsey, W.; Duncan, R.; Snow, Z.; Dehoff, R.; Paquit, V. *A Data-Driven Framework for Direct Local Tensile Property Prediction of Laser Powder Bed Fusion Parts*. Materials 2023, 16, 7293. https://doi.org/10.3390/ma16237293

2. Scime, L.; Siddel, D.; Baird, S.; Paquit, V. *Layer-wise anomaly detection and classification for powder bed additive manufacturing processes: A machine-agnostic algorithm for real-time pixel-wise semantic segmentation*. Addit. Manuf. 36 (2020) 101453. https://doi.org/10.1016/j.addma.2020.101453

3. Halsey, W.; Rose, D.; Scime, L.; Dehoff, R.; Paquit, V. *Localized Defect Detection from Spatially Mapped, In-Situ Process Data With Machine Learning*. Front. Mech. Eng. 7 (2021) 1–14. https://doi.org/10.3389/fmech.2021.767444

4. HDF5 for Python: https://docs.h5py.org/en/stable/

---

## 라이선스

이 데이터셋은 Creative Commons Attribution (CC BY) 4.0 라이선스 하에 공개되어 있습니다.
