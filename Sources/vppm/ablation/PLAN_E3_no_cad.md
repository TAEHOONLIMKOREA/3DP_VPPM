# E3: No-CAD 실험 계획

> **공통 설정** (피처 그룹 정의, CV, 모델, 해석 기준) 은 [PLAN.md](./PLAN.md) 참조.
> 본 문서는 E3 실험 고유 정보만 기술.

---

## 1. 실험 정의

| 항목 | 값 |
|:-----|:---|
| **실험 ID** | E3 |
| **실험명** | No-CAD |
| **제거 그룹** | G3 CAD / 좌표 (3 채널) |
| **제거 피처 idx** | 0, 1, 2 |
| **사용 피처 수** | 21 → **18** |
| **출력 디렉터리** | `Sources/pipeline_outputs/ablation/E3_no_cad/` |

### 1.1 제거되는 3 개 CAD 채널

[features.py:21-23](../origin/features.py#L21-L23) 기준:

| idx | 채널 | 계산 방법 | 물리적 의미 |
|:---:|:-----|:----------|:-----------|
| 0 | distance_from_edge     | `slices/part_ids` 기반 거리변환, saturation=3mm | 파트 외곽까지의 최단 거리 (mm) |
| 1 | distance_from_overhang | 레이어간 part 겹침 비교, saturation=71 layers  | 수직 하방 오버행까지 거리 (layers) |
| 2 | build_height           | 슈퍼복셀 z 중심 (mm)                           | 절대 높이 |

---

## 2. 가설

> CAD 좌표는 **형상 민감 특성** (엣지·오버행 근처의 열 응축 / 냉각 비등방성) 에 영향을 줄 수 있다.
> 특히:
>
> 1. **YS / UTS 에 영향** — 엣지/오버행 근처는 냉각 속도 차이 → 입자 미세구조 차이
> 2. **B1.3 (오버행 형상 빌드) 에서 가장 크게 손상** 예상
> 3. 다만 DSCNN 과 sensor 가 공간적으로 상관된 정보를 일부 담고 있어 **전체 ΔRMSE 는 크지 않을 것**

---

## 3. 구현

### 3.1 사전 요건

- `config.FEATURE_GROUPS["cad"] = [0, 1, 2]` — **이미 등록됨**
- `EXPERIMENTS["E3"]` — **이미 등록됨**

### 3.2 실행 명령

**호스트:**

```bash
./venv/bin/python -m Sources.vppm.ablation.run --experiment E3
./venv/bin/python -m Sources.vppm.ablation.run --experiment E3 --quick
```

**도커:**

```bash
cd docker/ablation/cad
./run.sh              # 기본 (GPU 2)
./run.sh --quick
```

### 3.3 산출물

```
Sources/pipeline_outputs/ablation/E3_no_cad/
├── experiment_meta.json
├── features/normalization.json
├── models/{vppm_*.pt, training_log.json}
└── results/{metrics_raw.json, metrics_summary.json,
           predictions_*.csv, correlation_plots.png, scatter_plot_uts.png}
```

---

## 4. 실제 결과 (2026-04-23 기준)

### 4.1 RMSE (원본 스케일)

| 속성 | E0 | **E3** | ΔRMSE | Naive | E3 vs Naive |
|:---:|:--:|:-----:|:-----:|:-----:|:-----------:|
| YS  | 28.66 ± 0.62 | 29.32 ± 0.65 | **+0.66** | 33.91 | +13.5% |
| UTS | 60.72 ± 2.59 | 60.95 ± 0.93 | **+0.23** | 68.43 | +10.9% |
| UE  | 12.79 ± 0.27 | 13.10 ± 0.09 | **+0.31** | 15.00 | +12.7% |
| TE  | 15.46 ± 0.20 | 15.96 ± 0.17 | **+0.50** | 18.52 | +13.8% |

### 4.2 Fold 별 RMSE (UTS)

| Fold | 0 | 1 | 2 | 3 | 4 |
|:----:|--:|--:|--:|--:|--:|
| E0 | 61.26 | 60.75 | 63.22 | 62.53 | 55.86 |
| E3 | 60.83 | 61.24 | 59.80 | 62.55 | 60.34 |

**E3 UTS fold std 0.93 은 baseline 2.59 보다 훨씬 작음** — CAD 제거가 오히려 fold 간 안정성 향상.
이는 CAD 피처가 fold 분할에 따라 변동성을 유발하는 요인이었음을 시사.

### 4.3 판정

✅ **가설 3 검증**: 전체 ΔRMSE 가 모든 속성에서 **0.7 MPa 이내**. 4 그룹 중 **가장 작은 영향**
 (E4 의 −1.04 다음으로 작음, 단 E4 는 placeholder 특수 케이스).

⚠️ **가설 1 부분 검증**: YS (+0.66), TE (+0.50) 는 약간의 증가. UTS (+0.23) 는 거의 무변화.
 강도에 특히 크다는 가설은 **약하게만 지지**.

⚠️ **가설 2 검증 부족**: B1.3 (오버행) 에서의 per-build ΔRMSE 분해는 별도 실행 필요.

---

## 5. 빌드별 잔차 분해

별도 실행 필요:

```bash
./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment E3
# → Sources/pipeline_outputs/ablation/E3_no_cad/per_build_analysis.md
```

### 5.1 예상 패턴 (미실행)

- **B1.3 (오버행)**: `distance_from_overhang` 의 직접 수혜 빌드. 제거 시 가장 큰 타격 예상.
- **B1.1, B1.2**: 기하가 단순 — CAD 제거 영향 작을 것.
- **B1.4, B1.5**: 공정/리코터 변동이 주요 — CAD 는 부차적.

→ per_build 분해가 가설 2 검증의 핵심.

---

## 6. 해석 및 후속 함의

1. **CAD 정보는 DSCNN·sensor 에 상당 부분 흡수됨** — 3 채널 제거로도 전체 성능은 거의 동일.
   이는 피처 간 **정보 중복** 을 시사. 특히 엣지 근처의 결함 패턴은 DSCNN 이, 오버행에 의한
   냉각 비등방성은 build_plate_temperature (센서) 가 간접적으로 반영할 가능성.
2. **경량화 후보로 CAD 그룹 제거는 안전**: ΔUTS +0.23 은 내재 측정오차(15.6 MPa) 대비 무시 가능.
   단 B1.3 편향 주의 — per-build 분해 후 결정 권장.
3. **Fold std 감소** 는 흥미로운 부작용 — CAD 가 fold 변동성을 추가했을 가능성 (훈련/검증 사이
   part 경계 분포 차이). 후속 seed 반복으로 검증.

---

## 7. 연관 문서

- 공통 설정: [PLAN.md](./PLAN.md)
- 종합 보고서: [FULL_REPORT.md](../../pipeline_outputs/ablation/FULL_REPORT.md) §5
- 후속 추천: per-build 분해 실행 후 B1.3 결과 확인
