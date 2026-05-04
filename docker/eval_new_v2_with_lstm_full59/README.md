# docker/eval_new_v2_with_lstm_full59

학습된 LSTM_FULL59 모델(`vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1`)을
신규 build `[new_v2] (Peregrine v2023-10) / 2023-03-15 AMMTO Spatial Variation Baseline.hdf5`에
적용해 part-level 인장 물성 예측 및 GT 비교를 수행한다.

- **학습은 하지 않는다.** 기존 5-fold 앙상블 모델로 inference 만.
- 예측 타겟: **YS / UTS / TE** (GT 존재). **UE** 는 GT 부재 — prediction 파일만 산출.
- 신규 build 의 GT 는 `parts/test_results` 에 part 단위로 저장되어 있어, SV→part 집계 후 비교.

자세한 실험 동기는 `Sources/vppm/eval_new_v2_with_lstm_full59/run.py` 의 docstring 및
`Sources/vppm/common/config.py` 의 `NEW_V2_EVAL_*` 섹션 참조.

## 사전조건

### 1. 학습된 fullstack 모델

`vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1` 실험이 완료되어 아래 파일이 존재해야 한다:

```bash
Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/
  models/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1_{YS,UTS,UE,TE}_fold{0..4}.pt  # 20개
  features/normalization.json
```

빠른 확인:
```bash
ls Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/models/*.pt | wc -l
ls Sources/pipeline_outputs/experiments/vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/features/normalization.json
```

### 2. 입력 HDF5

```
ORNL_Data/Co-Registered In-Situ and Ex-Situ Dataset/[new_v2] (Peregrine v2023-10)/
  2023-03-15 AMMTO Spatial Variation Baseline.hdf5
```

### 3. 호스트 사전 mkdir (1회)

docker daemon 이 bind mount 대상 디렉터리를 자동 생성하면 root 소유가 되어
컨테이너의 일반 사용자(UID 1001)가 쓸 수 없다. 미리 만들어 둔다:

```bash
mkdir -p /home/taehoon/3DP_VPPM/Sources/pipeline_outputs/experiments/eval_new_v2_with_lstm_full59/{cache,features,results}
```

이미 root 소유로 생성됐다면:
```bash
sudo chown -R taehoon:taehoon /home/taehoon/3DP_VPPM/Sources/pipeline_outputs/experiments/eval_new_v2_with_lstm_full59
```

## 환경변수

| 변수 | 기본값 | 설명 |
|:--|:--|:--|
| `EVAL_NEW_V2_PHASE` | `all` | `features` / `cache` / `evaluate` / `all` |
| `EVAL_NEW_V2_EXTRA` | (없음) | 추가 인자. `--quick` → evaluate 단계에서 첫 256 SV 만 |
| `NVIDIA_VISIBLE_DEVICES` | `0` | 사용할 GPU 번호 (단일 GPU) |
| `UID_GID` | `1001:1001` | 호스트 사용자 UID:GID (`.env` 에 명시) |

`.env` 파일에서 기본값을 변경하거나, `docker compose up` 앞에 인라인으로 지정한다.

## 풀런 (권장 패턴)

```bash
cd /home/taehoon/3DP_VPPM

# 0) 출력 디렉터리 사전 생성
mkdir -p Sources/pipeline_outputs/experiments/eval_new_v2_with_lstm_full59/{cache,features,results}

cd docker/eval_new_v2_with_lstm_full59

# 1) 전체 일괄 실행 (features → cache → evaluate)
docker compose up -d --build
docker compose logs -f          # 실시간 로그
docker compose down             # 완료 후 정리
```

### 단계별 실행

features/cache 단계는 HDF5 읽기 위주라 CPU 로도 충분하다.
evaluate 단계만 GPU 를 적극 사용한다.

```bash
# 1단계: features 추출 (CPU 도 충분, 수 분)
EVAL_NEW_V2_PHASE=features docker compose up -d --build
docker compose logs -f
docker compose down

# 2단계: 6 시퀀스 캐시 빌드 (1~2 시간, ~1.5 GB 디스크)
EVAL_NEW_V2_PHASE=cache docker compose up -d --build
docker compose logs -f
docker compose down

# 3단계: evaluate (5-fold ensemble inference → part-level 집계, 수십 분)
EVAL_NEW_V2_PHASE=evaluate docker compose up -d --build
docker compose logs -f
docker compose down
```

### 옵션 변경 예시

```bash
# GPU 1 번 사용
NVIDIA_VISIBLE_DEVICES=1 docker compose up -d --build

# smoke: evaluate 만 첫 256 SV 로 빠르게 확인 (수 분)
EVAL_NEW_V2_PHASE=evaluate EVAL_NEW_V2_EXTRA=--quick docker compose up -d --build
```

## 캐시 빌드 소요 시간 및 디스크 견적

신규 build `AMMTO_v2` 는 약 **217 k SVs, 1819 layers** 규모.

| 캐시 파일 | 예상 크기 |
|:--|:--|
| `crop_stacks_AMMTO_v2.h5` (visible/0) | ~400 MB |
| `crop_stacks_v1_AMMTO_v2.h5` (visible/1) | ~400 MB |
| `sensor_stacks_AMMTO_v2.h5` | ~80 MB |
| `dscnn_stacks_AMMTO_v2.h5` | ~200 MB |
| `cad_patch_stacks_AMMTO_v2.h5` | ~200 MB |
| `scan_patch_stacks_AMMTO_v2.h5` | ~200 MB |
| **합계** | **~1.5 GB** |

빌드 시간: 약 **1~2 시간** (HDF5 I/O 위주, GPU 불필요).
evaluate: 약 **수십 분** (5-fold × 4 prop × GPU inference).

## 산출물

```
Sources/pipeline_outputs/experiments/eval_new_v2_with_lstm_full59/
  features/
    features.npz            — 21-feat + part_ids + GT (YS/UTS/TE)
  cache/
    crop_stacks_AMMTO_v2.h5
    crop_stacks_v1_AMMTO_v2.h5
    sensor_stacks_AMMTO_v2.h5
    dscnn_stacks_AMMTO_v2.h5
    cad_patch_stacks_AMMTO_v2.h5
    scan_patch_stacks_AMMTO_v2.h5
  results/
    per_part_predictions.csv    — part-level 예측값 + GT (YS/UTS/TE/UE)
    per_sv_predictions.csv      — SV-level 예측값
    metrics_summary.json        — MAE/RMSE/R² per property
    scatter_{YS,UTS,TE}.png     — 예측 vs GT scatter plot
```

## 참고

- 진입점: `Sources/vppm/eval_new_v2_with_lstm_full59/run.py`
- 설정: `Sources/vppm/common/config.py` — `NEW_V2_EVAL_*` 섹션
- 학습 실험: `docker/lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1/`
