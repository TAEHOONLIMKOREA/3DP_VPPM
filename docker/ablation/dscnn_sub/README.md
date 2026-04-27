# Docker — DSCNN 서브 채널 Ablation (E5~E12 + E23, E24)

[PLAN_dscnn_subablation.md](../../../Sources/vppm/ablation/PLAN_dscnn_subablation.md) 의
DSCNN 8 클래스 서브 ablation 및 2 개 카테고리 묶음 실험을 **도커 컨테이너에서** 실행한다.

E1~E4, E13~E22, E31~E33 와 동일한 공용 이미지 `vppm-ablation:gpu` 를 공유
([docker/ablation/Dockerfile](../Dockerfile)).

## 실험 10종

| ID | drop_group | 제거 피처 idx | 카테고리 | 의미 |
|:--:|:----------:|:-------------:|:-------:|:----|
| E5  | `dscnn_powder`             | [3]  | Normal | 미용융 분말 (LOF inverse proxy) |
| E6  | `dscnn_printed`            | [4]  | Normal | 정상 프린트 영역 |
| E7  | `dscnn_recoater_streaking` | [5]  | Defect | **리코터 줄무늬 (B1.5 핵심 후보)** |
| E8  | `dscnn_edge_swelling`      | [6]  | Defect | 엣지 팽창 |
| E9  | `dscnn_debris`             | [7]  | Defect | 잔해물 (B1.4 스패터 관련) |
| E10 | `dscnn_super_elevation`    | [8]  | Defect | 과도 돌출 |
| E11 | `dscnn_soot`               | [9]  | Defect | 매연 |
| E12 | `dscnn_excessive_melting`  | [10] | Defect | **과다 용융 (B1.2 Keyhole 핵심 후보)** |
| E23 | `dscnn_defects_all`        | [5,6,7,8,9,10] | 묶음 | 결함 6채널 (Normal 2개만 남김) |
| E24 | `dscnn_normal`             | [3,4] | 묶음   | Normal 2채널 (Defect 6개만 남김) |

**주 판정 지표**: UE RMSE (E1 전체 제거 시 UE 가 naive 수준으로 붕괴했던 그룹이므로).
보조 지표는 UTS.

## 전제 조건

- 호스트 NVIDIA GPU 4장 + `nvidia-docker2` (compose `runtime: nvidia`)
- 호스트 `venv/` 에 torch + cuda 설치되어 있어야 함 (bind-mount 로 사용)
- `Sources/pipeline_outputs/features/all_features.npz` 존재 (baseline 피처 추출 완료)

## 실행 방법

### 단일 실험

```bash
cd docker/ablation/dscnn_sub
./run.sh E7                    # GPU 0 기본 (recoater_streaking)
./run.sh E12 --gpu 2           # GPU 2 지정 (excessive_melting)
./run.sh E23 --gpu 0 --quick   # smoke test (defects_all)
```

### 10개 실험 전체 (4-GPU 배치 병렬)

```bash
cd docker/ablation/dscnn_sub
./run_all.sh              # 전체 (~45–60분)
./run_all.sh --quick      # smoke (~5분)
```

배치 스케줄:

| Batch | 병렬 실행 | GPU 배정 |
|:-----:|:---------|:--------|
| 1/3 | E5·E6·E7·E8           | 0·1·2·3 |
| 2/3 | E9·E10·E11·E12        | 0·1·2·3 |
| 3/3 | E23·E24               | 0·1     |

각 배치 내부는 병렬, 배치 간은 순차. 로그는 `/tmp/dscnn_sub_logs_<timestamp>/E??.log`.

### 수동 compose (필요 시)

```bash
cd docker/ablation/dscnn_sub
EXPERIMENT_ID=E7 NVIDIA_VISIBLE_DEVICES=2 docker compose run --rm ablation-dscnn-sub
```

## 볼륨 매핑 (E1~E22 와 동일)

| 호스트 | 컨테이너 | 모드 | 용도 |
|---|---|:---:|---|
| `venv/` | `/workspace/venv` | ro | torch + cuda |
| `Sources/pipeline_outputs/features/` | 동일 | ro | `all_features.npz` |
| `Sources/pipeline_outputs/results/` | 동일 | ro | baseline 참조 |
| `Sources/pipeline_outputs/ablation/` | 동일 | rw | **산출물** |

## 산출물

```
Sources/pipeline_outputs/ablation/
├── E5_no_dscnn_powder/
├── E6_no_dscnn_printed/
├── E7_no_dscnn_recoater_streaking/
├── E8_no_dscnn_edge_swelling/
├── E9_no_dscnn_debris/
├── E10_no_dscnn_super_elevation/
├── E11_no_dscnn_soot/
├── E12_no_dscnn_excessive_melting/
├── E23_no_dscnn_defects_all/
├── E24_no_dscnn_normal/
└── summary.md         # run_all.sh 마지막에 재생성됨
```

각 실험 폴더 레이아웃은 E1~E22 와 동일
(`experiment_meta.json` + `models/` + `results/` + `features/normalization.json`).

## summary.md 흐름

- 각 컨테이너는 `--skip-summary` 로 실행되어 중간 덮어쓰기 방지.
- 모든 배치 완료 후 호스트 `venv` 로 `Sources.vppm.ablation.run --rebuild-summary` 를 호출해
  기존 E1~E4, E13~E22, E31~E33 까지 포함한 **통합 summary.md** 를 재생성.
- 수동 재생성:
  ```bash
  ./venv/bin/python -m Sources.vppm.ablation.run --rebuild-summary
  ```

## 트러블슈팅

- **GPU 부족**: 호스트에 GPU 4장이 아니면 `run_all.sh` 의 배치 편집 필요.
  또는 `./run.sh E5 --gpu 0 && ./run.sh E6 --gpu 0 && ...` 로 단일 GPU 순차.
- **권한 에러 (UID mismatch)**: `Sources/pipeline_outputs/ablation` 가 현재 사용자 소유인지 확인.
- **`EXPERIMENT_ID` 미지정**: compose 에서 `${EXPERIMENT_ID:?...}` 로 즉시 실패.
  항상 `run.sh` 또는 env 지정 후 호출.
- **이미지 재빌드**: `docker/ablation/Dockerfile` 수정 시 `run.sh` 가 자동으로 `compose build` 수행.
  필요하면 `docker image rm vppm-ablation:gpu` 후 재빌드.

## 후속 분석

실험 완료 후:

```bash
# 빌드별 잔차 분해 (예: E7 리코터 결함이 B1.5 에서만 지배하는지 확인)
./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment E7
# → Sources/pipeline_outputs/ablation/E7_no_dscnn_recoater_streaking/per_build_analysis.md
```

주요 확인 포인트 (PLAN §2.4 빌드별 특이성 가설):

| 빌드 | 핵심 예상 채널 | 확인 명령 |
|:----:|:-------------:|:---------|
| B1.2 Keyhole | `seg_excessive_melting` (E12) | `analyze_per_build --experiment E12` → B1.2 ΔRMSE 최대 확인 |
| B1.2 LOF     | `seg_powder` (E5)             | `analyze_per_build --experiment E5`  → B1.2 ΔRMSE 최대 확인 |
| B1.4 스패터  | `seg_debris` (E9)             | `analyze_per_build --experiment E9`  → B1.4 ΔRMSE 최대 확인 |
| B1.5 리코터  | `seg_recoater_streaking` (E7) | `analyze_per_build --experiment E7`  → B1.5 ΔRMSE 최대 확인 |

10 개 실험 결과 집계는 별도 리포트 작성 — `PLAN_dscnn_subablation.md §4.1` 참조.

## 연관 문서

- 계획서: [PLAN_dscnn_subablation.md](../../../Sources/vppm/ablation/PLAN_dscnn_subablation.md)
- 상위 그룹: [PLAN_E1_no_dscnn.md](../../../Sources/vppm/ablation/PLAN_E1_no_dscnn.md)
- 공통 설정: [PLAN.md](../../../Sources/vppm/ablation/PLAN.md)
- 종합 보고서: [FULL_REPORT.md](../../../Sources/pipeline_outputs/ablation/FULL_REPORT.md)
