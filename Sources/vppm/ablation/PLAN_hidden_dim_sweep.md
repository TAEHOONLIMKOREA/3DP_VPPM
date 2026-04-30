# Hidden-dim Sweep 실험 계획 (Model Capacity Ablation)

> **공통 설정** (피처 그룹 정의, CV, 데이터, 정규화) 은 [PLAN.md](./PLAN.md) 참조.
> 본 문서는 **모델 capacity** 를 변화시키는 ablation — 피처는 21개 baseline 동일.
>
> **목적**: VPPM 의 `Linear(21 → 128) → ReLU → Dropout → Linear(128 → 1)` 구조에서 hidden_dim=128 이 정말 필요한지, 더 작거나 더 큰 hidden 이 어떤 영향을 주는지 정량화한다.

---

## 1. 동기

baseline 모델의 hidden 차원은 [config.py:HIDDEN_DIM = 128](../common/config.py) 로 고정되어 있고, 논문 Section 2.11 의 "shallow perceptron" 정의를 그대로 따른 값이다. 그러나:

- **21 → 128 → 1** 은 21차원 입력 대비 약 6배 확장. **2,944 파라미터** vs 학습 ~24,000 SV.
- hidden=1 (극단적 1차원 bottleneck) 과 비교하면 **표현력 capacity 의 기여** 를 측정할 수 있음.
- hidden=64 가 충분하다면 1D CNN 모델의 MLP head 크기를 줄여 **공정한 비교 기준선**을 잡을 수 있음.
- hidden=256 까지 키워도 plateau 라면 "21 피처 자체의 정보 한계" 를 시사 — 새 피처 / 새 표현 (1D CNN, hierarchical) 도입 동기가 강해짐.

---

## 2. 실험 정의

총 **3 sweep 점** + 기존 E0 baseline (hidden=128) 을 비교 기준으로 사용.

| ID | hidden_dim | 비선형성 | 파라미터 수 (대략) | 비고 |
|:--:|:--:|:--:|:--:|:--|
| **H1** | **1** (극단 bottleneck) | ✅ ReLU+Dropout | **24** | 1차원 bottleneck — 표현력 하한선 (linear 와 거의 동급 예상) |
| **H2** | **64** | ✅ | **1,473** | 중간 capacity — plateau 진입 후보 |
| **(E0)** | 128 | ✅ | 2,945 | **기존 baseline 재사용**, 별도 학습 없음 |
| **H3** | **256** | ✅ | **5,889** | 상위 capacity — overfit / 추가 이득 검증 |

> **공통 고정**: 피처=21 (baseline), 5-fold sample-wise CV, L1 loss, Adam(lr=1e-3, β=(0.9, 0.999), eps=1e-4), batch=1000, dropout=0.1, weight init std=0.1, early-stop patience=50, max=5000 epoch — [config.py](../common/config.py).
>
> **유일 변수**: `HIDDEN_DIM` 만.

---

## 3. 가설

3점 sweep (1 / 64 / 256) + baseline (128) 으로 **단조성 / plateau / overfit** 셋 다 가능 여부를 한 번에 본다.

| 가설 | 예상 결과 | 의미 |
|:--|:--|:--|
| **G1: capacity 가 너무 작으면 명확히 나쁨** | H1 (hidden=1) UTS RMSE ≫ E0 (예: +5~10 MPa) | 1차원 bottleneck 으로는 21 피처의 비선형 결합 표현 불가 — 비선형 + 충분한 폭이 둘 다 필요 |
| **G2: 64 부터 plateau** | H2 (64) ≈ E0 (128) — 차이 < fold std (≈1.5 MPa) | 128 은 보수적 선택 — 64 만으로 충분 → 1D CNN MLP head 도 64 로 줄일 근거 |
| **G3: 256 은 추가 이득 없음** | H3 (256) ≈ E0 (128). 또는 fold std 가 약간 커짐 | 21 피처의 정보가 128 에서 포화 → 새 표현(1DCNN, hierarchical) 도입 동기 |

**판정 매트릭스**:

| H1 vs E0 | H2 vs E0 | H3 vs E0 | 결론 |
|:--:|:--:|:--:|:--|
| ≫ (나쁨) | ≈ | ≈ | 표준 시나리오 — 64 충분, 128 보수적 상한 |
| ≫ | ≪ (좋음) | ≪ | 64 가 오히려 더 좋음 — 128 이 over-parametrized |
| ≫ | ≈ | ≪ (좋음) | 21 피처에 아직 추출 안 된 정보 있음 — wider 필요 |
| ≫ | ≫ | ≫ | 128 이 정확한 sweet-spot |
| ≈ | ≈ | ≈ | hidden_dim 영향 자체가 미미 — 21 피처가 사실상 선형 결합 가능 (가능성 낮음) |

---

## 4. 구현

### 4.1 사전 요건

- `config.py:HIDDEN_DIM` 을 모듈 인자로 override 할 수 있게 변경 (CLI 인자 → train_all 에 전달).
- `Sources/vppm/baseline/model.py:VPPM` 의 `hidden_dim` 인자가 1/64/256 모두에서 정상 작동하는지 확인 (현재도 가능할 가능성 큼).
- `Sources/vppm/ablation/run_hidden_sweep.py` 신설 — `EXPERIMENTS` dict 와 동일 패턴, 단 drop_group 대신 `hidden_dim` 인자 사용.

### 4.2 산출물 디렉토리

```
Sources/pipeline_outputs/ablation/
├── H1_hidden_1/                      # 극단 bottleneck
├── H2_hidden_64/
└── H3_hidden_256/
    ├── experiment_meta.json          # {"hidden_dim": int, "n_params": int}
    ├── features/normalization.json   # 21-피처 동일 (기존 E0 재사용 가능)
    ├── models/
    │   ├── vppm_{YS,UTS,UE,TE}_fold{0-4}.pt
    │   └── training_log.json
    └── results/
        ├── metrics_raw.json
        ├── metrics_summary.json
        ├── predictions_{YS,UTS,UE,TE}.csv
        └── correlation_plots.png
```

> hidden=128 결과는 [experiments/vppm_baseline/results/metrics_raw.json](../../pipeline_outputs/experiments/vppm_baseline/results/metrics_raw.json) (기존 E0) 을 그대로 비교 기준으로 사용 — 별도 학습 X.

### 4.3 실행 명령

```bash
# 단일
./venv/bin/python -m Sources.vppm.ablation.run_hidden_sweep --hidden 1
./venv/bin/python -m Sources.vppm.ablation.run_hidden_sweep --hidden 64
./venv/bin/python -m Sources.vppm.ablation.run_hidden_sweep --hidden 256

# 전체 sweep (H1, H2, H3 순차)
./venv/bin/python -m Sources.vppm.ablation.run_hidden_sweep --all

# smoke test (1 fold, max_epoch=50)
./venv/bin/python -m Sources.vppm.ablation.run_hidden_sweep --hidden 64 --quick
```

---

## 5. 결과 (실행 후 채움)

### 5.1 5-fold RMSE (원본 스케일)

| ID | hidden | YS | UTS | UE | TE | 평균 epoch | val_loss (정규화) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| H1 | 1   | _ | _ | _ | _ | _ | _ |
| H2 | 64  | _ | _ | _ | _ | _ | _ |
| **(E0)** | **128** | **24.28 ± 0.75** | **42.88 ± 2.00** | **9.34 ± 0.28** | **11.27 ± 0.50** | _ | _ |
| H3 | 256 | _ | _ | _ | _ | _ | _ |

### 5.2 ΔRMSE vs E0 (= 128 baseline)

| ID | ΔYS | ΔUTS | ΔUE | ΔTE |
|:--:|:--:|:--:|:--:|:--:|
| H1 | _ | _ | _ | _ |
| H2 | _ | _ | _ | _ |
| H3 | _ | _ | _ | _ |

### 5.3 학습 안정성 (fold std)

| ID | std(YS) | std(UTS) | std(UE) | std(TE) |
|:--:|:--:|:--:|:--:|:--:|
| H1 | _ | _ | _ | _ |
| H2 | _ | _ | _ | _ |
| H3 | _ | _ | _ | _ |

→ H3 (256) 에서 fold std 가 커지면 overfit 신호.

---

## 6. 해석 가이드

### 6.1 가설별 판정 기준

| 판정 | 기준 | 의미 |
|:--|:--|:--|
| G1 ✅ | ΔUTS(H1 vs E0) > 5 MPa | 1차원 bottleneck 으로 표현 불가 — 충분한 hidden 폭이 필요 |
| G1 ❌ | ΔUTS(H1 vs E0) < 1 MPa | hidden_dim 영향이 사실상 없음 (가능성 낮음) |
| G2 ✅ | \|ΔUTS(H2 vs E0)\| < fold std (≈1.5 MPa) | 64 가 충분, 128 은 보수적 상한 |
| G3 ✅ | \|ΔUTS(H3 vs E0)\| < std AND fold std(H3) ≥ std(E0) | overfit 진입, 추가 capacity 무용 |
| G3 ❌ | UTS(H3) < UTS(E0) − std | 21 피처에 아직 추출 안 된 정보 있음 → wider 필요 |

### 6.2 1D CNN PLAN 으로의 피드백

- **H2 (64) ≈ E0 → 64 충분**: 1D CNN MLP head 도 64 로 줄여 학습 속도 ↑ 비교 공정성 유지.
- **H2 ≪ E0 (=64 가 더 좋음)**: 128 은 over-parametrized → baseline 도 단순화 후 재학습 필요.
- **H3 (256) > E0 (=256 이 더 좋음)**: baseline 자체가 underfit 상태 → 기존 ablation 결과 (G4 scan +18 등) 도 underfit 영향이 섞인 값일 수 있어 재해석 필요.

---

## 7. 위험 / 주의사항

| 위험 | 대응 |
|:--|:--|
| H1 (hidden=1) 학습 불안정 — 매우 좁은 bottleneck 으로 gradient 흐름 이슈 가능 | 다중 seed 로 robust 확인 (선택). 첫 run 만 보고 NaN/divergence 발생 시 lr 1/2 축소 |
| 작은 모델 (H1) 이 batch=1000 / epoch=5000 에서 빠르게 수렴 후 멈춤 (early-stop patience=50 가 너무 짧을 수 있음) | patience 변경하지 말고 동일 — capacity 영향 분리 측정이 목적 |
| H3 (256) 메모리 / 시간 부담 | 작아서 무시 가능 (~6k 파라미터). CPU 로도 충분 |
| E0 비교 시 환경 차이 (라이브러리 버전, seed) | E0 의 [metrics_raw.json](../../pipeline_outputs/experiments/vppm_baseline/results/metrics_raw.json) 을 그대로 사용 — 재학습 안 함 |

---

## 8. 일정

총 **3 sweep × 4 prop × 5 fold = 60 학습 run**.
1 run ≈ 3~5 분 (CPU) → **총 ~3~5 시간 (직렬)** 또는 **~1 시간 (4-process 병렬)**.

| 단계 | 작업 | 예상 |
|:--:|:--|:--:|
| 1 | run_hidden_sweep.py 작성 + model.py sanity 확인 | 20 분 |
| 2 | smoke test (H2 --quick) | 5 분 |
| 3 | 전체 sweep 실행 (H1 / H2 / H3) | 1~5 시간 |
| 4 | 결과 표 채우기 + 해석 | 20 분 |

---

## 9. 연관 문서

- 공통 설정 / 인덱스: [PLAN.md](./PLAN.md)
- 피처 ablation 결과: [FULL_REPORT.md](../../pipeline_outputs/ablation/FULL_REPORT.md)
- 1D CNN 후속 활용: [Sources/vppm/1dcnn/PLAN.md](../1dcnn/PLAN.md) §6 학습 설정
- Baseline 모델 정의: [Sources/vppm/baseline/model.py](../baseline/model.py)
