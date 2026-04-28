# Docker — 스캔(G4) 서브 채널 Ablation (E31~E33)

스캔 피처(`laser_return_delay`, `laser_stripe_boundaries`) 의 단독 기여도를 ablation 으로 측정.

E1~E4, E13~E22 와 동일한 공용 이미지 `vppm-ablation:gpu` 를 공유
([docker/ablation/Dockerfile](../Dockerfile)).

---

## 전제 조건

- `Sources/vppm/origin/scan_features.py` 구현 완료 (build_melt_time_map / compute_return_delay_map /
  compute_stripe_boundaries_map)
- `Sources/pipeline_outputs/features/all_features.npz` 의 피처 #19, #20 이 실값으로 채워짐 (std > 0)
- Baseline 학습 완료 (`docker/baseline/` 또는 호스트에서 `run_pipeline --phase train,evaluate`)

---

## 실험 3종

| ID | drop_group | 제거 피처 idx | 의미 |
|:--:|:----------:|:-------------:|:----|
| E31 | `scan`                    | [18, 19, 20] | v2 No-Scan — G4 3개 전체 제거 (E4 v2 재실험) |
| E32 | `scan_return_delay`       | [19]         | #20 `laser_return_delay` 단독 제거 |
| E33 | `scan_stripe_boundaries`  | [20]         | #21 `laser_stripe_boundaries` 단독 제거 |

---

## 실행 방법

### 단일 실험

```bash
cd docker/ablation/scan_sub
./run.sh E32                   # GPU 0 기본
./run.sh E32 --gpu 1           # GPU 1 지정
./run.sh E32 --gpu 0 --quick   # smoke test
```

### 3개 실험 전체 (3-GPU 병렬, 단일 배치)

```bash
cd docker/ablation/scan_sub
./run_all.sh              # 전체 (~30분)
./run_all.sh --quick      # smoke (~2분)
```

GPU 배정:

| 실험 | GPU |
|:----:|:---:|
| E31 (v2 No-Scan) | 0 |
| E32 (No-ReturnDelay) | 1 |
| E33 (No-StripeBoundary) | 2 |

### 수동 compose

```bash
cd docker/ablation/scan_sub
EXPERIMENT_ID=E32 NVIDIA_VISIBLE_DEVICES=1 docker compose run --rm ablation-scan-sub
```

---

## 볼륨 매핑 (E1~E22 와 동일)

| 호스트 | 컨테이너 | 모드 | 용도 |
|---|---|:---:|---|
| `venv/` | `/workspace/venv` | ro | torch + cuda |
| `Sources/pipeline_outputs/features/` | 동일 | ro | **v2 `all_features.npz`** |
| `Sources/pipeline_outputs/results/` | 동일 | ro | **baseline v2 참조** |
| `Sources/pipeline_outputs/ablation/` | 동일 | rw | **산출물** |

---

## 산출물

```
Sources/pipeline_outputs/ablation/
├── E31_no_scan/                   # E4 와 동일한 그룹 제거 — exp_id 만 다른 별도 폴더
├── E32_no_scan_return_delay/
├── E33_no_scan_stripe_boundaries/
└── summary.md                     # run_all.sh 마지막에 재생성됨
```

---

## summary.md 흐름

- 각 컨테이너는 `--skip-summary` 로 실행되어 중간 덮어쓰기 방지.
- 모든 컨테이너 완료 후 호스트 `venv` 로 `Sources.vppm.ablation.run --rebuild-summary` 를 호출해
  통합 summary.md 를 재생성.
- 수동 재생성:
  ```bash
  ./venv/bin/python -m Sources.vppm.ablation.run --rebuild-summary
  ```

---

## 해석 포인트

| 실험 | 측정 대상 |
|:---:|:----------|
| E31 | 스캔 그룹 전체 가치 (E4 와 같은 그룹 제거 — 재현 / seed 영향 확인) |
| E32 | `laser_return_delay` 단독 기여 |
| E33 | `laser_stripe_boundaries` 단독 기여 |

각 ΔRMSE 가 fold std 보다 큰지 확인 → 통계적 유의성 판단.

---

## 후속 분석

```bash
# 빌드별 잔차 분해
./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment E31
./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment E32
./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment E33
```

각 실험 폴더에 `per_build_analysis.md` 생성됨.

---

## 트러블슈팅

- **EXPERIMENT_ID 미지정**: compose 에서 `${EXPERIMENT_ID:?...}` 로 즉시 실패. run.sh 사용 권장.
- **`all_features.npz 없음`**: `run_pipeline --phase features` 를 먼저 실행.
- **피처 #19/#20 이 0 으로 보이는 경우**: `np.load(features/all_features.npz)['features'][:, 19:21].std()` 로 검증. 0 이면 scan_features.py 적용 후 재추출 필요.
