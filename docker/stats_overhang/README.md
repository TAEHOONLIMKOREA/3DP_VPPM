# docker/stats_overhang

`distance_from_overhang` 피처의 분포 통계를 5개 빌드 전체에 대해 산출하는 스크립트
(`Sources/vppm/tools/stats_overhang.py`) 를 컨테이너 안에서 실행한다.
GPU 불필요 — CPU/numpy/scipy/h5py 만 사용.

## 사전조건

- `ORNL_Data_Origin/` 에 5개 빌드 HDF5 파일 존재
- 호스트 `./venv` 빌드 완료 (h5py, numpy, scipy, tqdm 포함)
- 산출물 디렉토리 자동 생성됨 (없어도 무방)

## 풀런 (5개 빌드 전수)

```bash
cd docker/stats_overhang
docker compose up -d --build
docker compose logs -f
docker compose down
```

## 부분 실행

```bash
# 특정 빌드만
STATS_OH_EXTRA="--builds B1.3" docker compose up -d --build

# 여러 빌드
STATS_OH_EXTRA="--builds B1.3 B1.1" docker compose up -d --build

# PNG 히스토그램 생략
STATS_OH_EXTRA="--no-plot" docker compose up -d --build

# GPU 변경
NVIDIA_VISIBLE_DEVICES=2 docker compose up -d --build
```

## 산출물

```
Sources/pipeline_outputs/figures/stats_overhang/
    per_build/{B1.1,B1.2,B1.3,B1.4,B1.5}.npy   — 각 빌드 SV-level 배열
    all_builds.npy                                — 5개 빌드 concat
    stats.json                                    — mean/std/min/max/percentiles
    histogram.png                                 — 분포 히스토그램
```
