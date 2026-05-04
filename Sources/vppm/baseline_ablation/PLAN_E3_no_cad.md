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
| **출력 디렉터리** | `Sources/pipeline_outputs/experiments/baseline_ablation/E3_no_cad/` |

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
./venv/bin/python -m Sources.vppm.baseline_ablation.run --experiment E3
./venv/bin/python -m Sources.vppm.baseline_ablation.run --experiment E3 --quick
```

**도커:**

```bash
cd docker/baseline_ablation/cad
./run.sh              # 기본 (GPU 2)
./run.sh --quick
```

### 3.3 산출물

```
Sources/pipeline_outputs/experiments/baseline_ablation/E3_no_cad/
├── experiment_meta.json
├── features/normalization.json
├── models/{vppm_*.pt, training_log.json}
└── results/{metrics_raw.json, metrics_summary.json,
           predictions_*.csv, correlation_plots.png, scatter_plot_uts.png}
```

---

## 4. 실제 결과 (v2, 2026-04-28)

### 4.1 RMSE (원본 스케일)

| 속성 | E0 (v2) | **E3** | ΔRMSE | Naive | E3 vs Naive |
|:---:|:--:|:-----:|:-----:|:-----:|:-----------:|
| YS  | 24.28 ± 0.75 | 26.13 ± 1.00 | **+1.85** | 33.91 | +23.0% 우위 |
| UTS | 42.88 ± 2.00 | 49.51 ± 2.46 | **+6.63** | 68.43 | +27.7% 우위 |
| UE  | 9.34 ± 0.28  | 10.20 ± 0.33 | **+0.86** | 15.00 | +32.0% 우위 |
| TE  | 11.27 ± 0.50 | 12.33 ± 0.40 | **+1.07** | 18.52 | +33.4% 우위 |

### 4.2 Fold 별 RMSE (UTS)

| Fold | 0 | 1 | 2 | 3 | 4 |
|:----:|--:|--:|--:|--:|--:|
| E0 | 45.02 | 41.34 | 42.17 | 45.44 | 40.43 |
| E3 | 49.45 | 47.23 | 50.99 | 53.29 | 46.57 |

각 fold 에서 +5~8 MPa 일관 악화. fold 분포 모양 유지.

### 4.3 판정 (v2 — 가설 대폭 수정)

⚠️ **v1 의 "CAD 는 약한 영향" 결론은 G4 placeholder 의 영향이었음**.
v2 baseline 이 강해진 만큼 CAD 의 절대 기여도 더 명확:

✅ **CAD 는 G1(DSCNN) 동급으로 강함** — ΔUTS +6.63 (E3) ≈ +6.14 (E1).
4 그룹 중 G4 다음 **2위 동급**. v1 에서 "DSCNN 이 CAD 를 흡수한다" 는 추론은 부정.

✅ **YS·UTS 에서 지배적**: E3 가 ΔYS +1.85 (E1 의 +1.69 보다 큼).
오버행 거리 / 엣지 거리 / 빌드 높이 3 채널이 강도 예측의 핵심 spatial prior.

⚠️ **per-build 분해 (가설 2)**: v2 미실행. B1.3 (오버행) 에서의 의존성 재검증 필요.

---

## 5. 빌드별 잔차 분해

별도 실행 필요:

```bash
./venv/bin/python -m Sources.vppm.baseline_ablation.analyze_per_build --experiment E3
# → Sources/pipeline_outputs/experiments/baseline_ablation/E3_no_cad/per_build_analysis.md
```

### 5.1 예상 패턴 (미실행)

- **B1.3 (오버행)**: `distance_from_overhang` 의 직접 수혜 빌드. 제거 시 가장 큰 타격 예상.
- **B1.1, B1.2**: 기하가 단순 — CAD 제거 영향 작을 것.
- **B1.4, B1.5**: 공정/리코터 변동이 주요 — CAD 는 부차적.

→ per_build 분해가 가설 2 검증의 핵심.

---

## 6. 해석 및 후속 함의 (v2)

1. **CAD 는 핵심 그룹** — v1 에서 약하다고 추정한 것은 G4 placeholder 효과. 3 채널만으로 G1(8ch DSCNN) 과 동급의 정보를 운반.
2. **경량화 시 CAD 제거는 위험** — ΔUTS +6.63 은 내재오차(15.6) 의 약 42% 수준으로 무시 불가.
3. **G1 과의 redundancy 미규명**: E3 단독 +6.63, E1 단독 +6.14, E1+E3 동시 제거 실험은 미수행.
   동시 제거 시 ΔUTS 가 합 (+12.8) 에 가까우면 직교, 그보다 작으면 redundancy 존재.

---

## 7. 연관 문서

- 공통 설정: [PLAN.md](./PLAN.md)
- 종합 보고서: [FULL_REPORT.md](../../pipeline_outputs/experiments/baseline_ablation/FULL_REPORT.md) §5
- 후속 추천: per-build 분해 실행 후 B1.3 결과 확인
