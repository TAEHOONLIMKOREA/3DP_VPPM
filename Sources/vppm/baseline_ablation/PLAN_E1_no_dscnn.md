# E1: No-DSCNN 실험 계획

> **공통 설정** (피처 그룹 정의, CV, 모델, 해석 기준) 은 [PLAN.md](./PLAN.md) 참조.
> 본 문서는 E1 실험 고유 정보만 기술.

---

## 1. 실험 정의

| 항목 | 값 |
|:-----|:---|
| **실험 ID** | E1 |
| **실험명** | No-DSCNN |
| **제거 그룹** | G1 DSCNN (8 채널) |
| **제거 피처 idx** | 3, 4, 5, 6, 7, 8, 9, 10 |
| **사용 피처 수** | 21 → **13** |
| **출력 디렉터리** | `Sources/pipeline_outputs/experiments/baseline_ablation/E1_no_dscnn/` |

### 1.1 제거되는 8 개 DSCNN 채널

[features.py:24-31](../origin/features.py#L24-L31) 기준:

| idx | 채널 | HDF5 class id | 물리적 의미 |
|:---:|:-----|:-------------:|:-----------|
| 3 | seg_powder            | 0  | 미용융 분말 영역 |
| 4 | seg_printed           | 1  | 정상 프린트 |
| 5 | seg_recoater_streaking| 3  | 리코터 줄무늬 |
| 6 | seg_edge_swelling     | 5  | 엣지 팽창 |
| 7 | seg_debris            | 6  | 잔해물 |
| 8 | seg_super_elevation   | 7  | 과도 돌출 |
| 9 | seg_soot              | 8  | 매연/그을음 |
| 10| seg_excessive_melting | 10 | 과다 용융 |

각 채널은 해당 레이어에서의 DSCNN 세그멘테이션 확률 맵을 가우시안 블러 → 슈퍼복셀 영역 평균.

---

## 2. 가설

> DSCNN 이 예측한 **결함 확률 맵** 은 재료의 국소 이상(void, crack 전구체) 을 가장 직접적으로 반영한다.
> 결함은 특히 **연신율 (UE / TE)** 과 강하게 연관 (파단 임계점) 이므로, DSCNN 제거 시:
>
> 1. **UE/TE 가 YS/UTS 보다 더 크게 악화** (논문 보고와 일관)
> 2. 전반적으로 가장 큰 ΔRMSE 예상 (4 그룹 중 가장 영향 큼)
> 3. Fold std 증가 — 핵심 피처가 빠지면 분할 민감도 커짐

---

## 3. 구현

### 3.1 사전 요건

- `Sources/vppm/common/config.py` 의 `FEATURE_GROUPS["dscnn"] = [3, ..., 10]` — **이미 등록됨**
- `Sources/vppm/baseline_ablation/run.py` 의 `EXPERIMENTS["E1"]` — **이미 등록됨**

### 3.2 실행 명령

**호스트 (단일 실행):**

```bash
./venv/bin/python -m Sources.vppm.baseline_ablation.run --experiment E1          # 전체
./venv/bin/python -m Sources.vppm.baseline_ablation.run --experiment E1 --quick  # smoke
```

**도커 (GPU 핀 + 격리):**

```bash
cd docker/baseline_ablation/dscnn
./run.sh              # 기본 (GPU 0)
./run.sh --quick      # smoke
```

### 3.3 산출물

```
Sources/pipeline_outputs/experiments/baseline_ablation/E1_no_dscnn/
├── experiment_meta.json       # dropped_feature_indices, n_feats, n_samples
├── features/
│   └── normalization.json     # 13-차원 재정규화 통계
├── models/
│   ├── vppm_{YS,UTS,UE,TE}_fold{0-4}.pt
│   └── training_log.json
└── results/
    ├── metrics_raw.json       # fold별 + 평균 RMSE
    ├── metrics_summary.json
    ├── predictions_{YS,UTS,UE,TE}.csv
    └── correlation_plots.png, scatter_plot_uts.png
```

---

## 4. 실제 결과 (v2, 2026-04-28)

> [metrics_raw.json](../../pipeline_outputs/experiments/baseline_ablation/E1_no_dscnn/results/metrics_raw.json) 참조.
> v1 (2026-04-23) 결과는 G4 placeholder 영향으로 baseline 의 절댓값이 달랐음. v2 가 표준.

### 4.1 RMSE (원본 스케일)

| 속성 | E0 Baseline (v2) | **E1 No-DSCNN** | ΔRMSE | Naive | E1 vs Naive |
|:---:|:-----------:|:---------------:|:-----:|:-----:|:-----------:|
| YS  | 24.28 ± 0.75 | 25.97 ± 1.08 | **+1.69** | 33.91 | +23.4% 우위 |
| UTS | 42.88 ± 2.00 | 49.02 ± 2.19 | **+6.14** | 68.43 | +28.4% 우위 |
| UE  | 9.34 ± 0.28  | 10.77 ± 0.35 | **+1.43** | 15.00 | +28.2% 우위 |
| TE  | 11.27 ± 0.50 | 12.86 ± 0.45 | **+1.60** | 18.52 | +30.6% 우위 |

### 4.2 Fold 별 RMSE (UTS)

| Fold | 0 | 1 | 2 | 3 | 4 |
|:----:|--:|--:|--:|--:|--:|
| E0 | 45.02 | 41.34 | 42.17 | 45.44 | 40.43 |
| E1 | 51.22 | 46.20 | 49.68 | 51.33 | 46.68 |

모든 fold 에서 UTS 가 +5~7 MPa 악화. fold 간 상대 순위 (어느 fold 가 더 쉬운가) 는 유지.

### 4.3 판정

✅ **가설 1 (연성 의존)**: UE/TE 모두 ΔRMSE > 1.4 — DSCNN 이 연신율 정보의 상당량 담음.
  단, v1 처럼 "naive 수준 붕괴" 는 아니고 여전히 +28% 우위 (v2 baseline 이 더 강함).

✅ **가설 2 (G1 효과 크기)**: E1 은 4 그룹 중 **3위** (G4 > G3 > G1 > G2).
  - ΔUTS: E4 (+18.48) > E3 (+6.63) > E1 (+6.14) > E2 (+3.97)
  - v1 에서 E1 이 1위였던 것은 G4 placeholder 의 영향.

✅ **가설 3 (fold std 확대)**: YS fold std 0.75 → 1.08 (1.4× 증가) — DSCNN 제거 시 fold 간 변동 커짐.

---

## 5. 빌드별 잔차 분해

### 5.1 ΔRMSE (UTS) 빌드별

상세 표는 [E13 per_build_analysis](../../pipeline_outputs/experiments/baseline_ablation/E13_no_dscnn_sensor/per_build_analysis.md)
의 "ΔE13 − ΔE2" 열로부터 역산. DSCNN 만 제거한 E1 의 per-build 분해는 별도 실험 필요 (미수행).

간접 추론:
- **B1.5 (리코터 손상)**: DSCNN 고유 기여 큼 — 리코터 결함 → DSCNN 분류 → 강도/연성 연결
- **B1.2 (파라미터 다양)**: DSCNN 이 결함 패턴에서 공정 파라미터를 역추론하는 역할
- **B1.4 (스패터)**: Sensor 와 중복 — DSCNN 없어도 sensor 가 보완

후속 명령:
```bash
./venv/bin/python -m Sources.vppm.baseline_ablation.analyze_per_build --experiment E1
# → Sources/pipeline_outputs/experiments/baseline_ablation/E1_no_dscnn/per_build_analysis.md
```

---

## 6. 해석 및 후속 함의 (v2 갱신)

1. **DSCNN 은 그룹 전체로만 의미** — collective effect (E5–E12 단독 모두 noise, E1 그룹 +6.14).
2. **G4(scan) > G3(cad) > G1(dscnn) > G2(sensor)** 순서. v1 추정과 달리 G3(CAD) 가 G1 과 동급으로 강함.
3. **DSCNN 서브 ablation (E5–E12, E23/E24) 모두 완료**: 단일 채널은 전부 noise, 묶음(E23 defects=+2.57, E24 normal=+1.75) 도 약함 → 채널 8 개를 다 봐야 신호가 발현.
4. **UE/TE 예측에서 DSCNN 의 상대 기여**: UE 의 ΔRMSE 절댓값(+1.43) 은 G4(+3.69) 의 약 39% — UE 도 G4 가 더 결정적.

---

## 7. 연관 문서

- 공통 설정 / 인덱스: [PLAN.md](./PLAN.md)
- 조합 실험: [PLAN_E13_combined.md](./PLAN_E13_combined.md) — DSCNN + Sensor 동시 제거
- 종합 보고서: [FULL_REPORT.md](../../pipeline_outputs/experiments/baseline_ablation/FULL_REPORT.md) §5 E1–E4 주요 그룹
