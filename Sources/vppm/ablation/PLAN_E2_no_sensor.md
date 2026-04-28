# E2: No-Sensor 실험 계획

> **공통 설정** (피처 그룹 정의, CV, 모델, 해석 기준) 은 [PLAN.md](./PLAN.md) 참조.
> 본 문서는 E2 실험 고유 정보만 기술.

---

## 1. 실험 정의

| 항목 | 값 |
|:-----|:---|
| **실험 ID** | E2 |
| **실험명** | No-Sensor |
| **제거 그룹** | G2 Temporal Sensor (7 채널) |
| **제거 피처 idx** | 11, 12, 13, 14, 15, 16, 17 |
| **사용 피처 수** | 21 → **14** |
| **출력 디렉터리** | `Sources/pipeline_outputs/ablation/E2_no_sensor/` |

### 1.1 제거되는 7 개 Sensor 채널

[features.py:32-38](../origin/features.py#L32-L38) 및 [config.py:68-76](../common/config.py#L68-L76):

| idx | 채널 | HDF5 `temporal/` | 물리적 의미 |
|:---:|:-----|:-----------------|:-----------|
| 11 | layer_print_time              | `layer_times`                  | 레이어별 프린트 시간 (s) |
| 12 | top_gas_flow_rate             | `top_flow_rate`                | 상단 가스 유량 (m/s) |
| 13 | bottom_gas_flow_rate          | `bottom_flow_rate`             | 하단 가스 유량 (m/s) |
| 14 | module_oxygen                 | `module_oxygen`                | 레이저 모듈 내 산소 농도 (ppm) |
| 15 | build_plate_temperature       | `build_plate_temperature`      | 빌드 플레이트 온도 (℃) |
| 16 | bottom_flow_temperature       | `bottom_flow_temperature`      | 하단 가스 온도 (℃) |
| 17 | actual_ventilator_flow_rate   | `actual_ventilator_flow_rate`  | 환풍기 실제 유량 (m/s) |

각 채널은 해당 슈퍼복셀의 70 레이어 범위 평균값 스칼라.

---

## 2. 가설

> Temporal Sensor 는 **빌드 단위 공정 상태**(산소 분위기, 가스 유량, 온도)를 반영한다.
> 공정 조건은 **용융 품질 → 강도** 를 지배하는 일차 인자이므로, 센서 제거 시:
>
> 1. **YS / UTS** 에 특히 큰 영향 (L-PBF 문헌과 일치)
> 2. 빌드 간 차이 설명력 감소 — 공정 파라미터가 다른 B1.2·B1.4 에서 특히 타격
> 3. 전반적 악화 (모든 속성 ΔRMSE > 0) 하지만 DSCNN 보다는 작음

---

## 3. 구현

### 3.1 사전 요건

- `config.FEATURE_GROUPS["sensor"] = [11, ..., 17]` — **이미 등록됨**
- `EXPERIMENTS["E2"]` — **이미 등록됨**

### 3.2 실행 명령

**호스트:**

```bash
./venv/bin/python -m Sources.vppm.ablation.run --experiment E2
./venv/bin/python -m Sources.vppm.ablation.run --experiment E2 --quick
```

**도커:**

```bash
cd docker/ablation/sensor
./run.sh              # 기본 (GPU 1)
./run.sh --quick
```

### 3.3 산출물

```
Sources/pipeline_outputs/ablation/E2_no_sensor/
├── experiment_meta.json
├── features/normalization.json
├── models/{vppm_*.pt, training_log.json}
└── results/{metrics_raw.json, metrics_summary.json,
           predictions_*.csv, correlation_plots.png, scatter_plot_uts.png}
```

---

## 4. 실제 결과 (2026-04-23 기준)

### 4.1 RMSE (원본 스케일)

| 속성 | E0 | **E2** | ΔRMSE | Naive | E2 vs Naive |
|:---:|:--:|:-----:|:-----:|:-----:|:-----------:|
| YS  | 24.28 ± 0.75 | 25.29 ± 0.97 | **+1.01** | 33.91 | +25.4% 우위 |
| UTS | 42.88 ± 2.00 | 46.85 ± 2.07 | **+3.97** | 68.43 | +31.5% 우위 |
| UE  | 9.34 ± 0.28  | 10.17 ± 0.49 | **+0.83** | 15.00 | +32.2% 우위 |
| TE  | 11.27 ± 0.50 | 12.21 ± 0.31 | **+0.94** | 18.52 | +34.0% 우위 |

### 4.2 Fold 별 RMSE (UTS)

| Fold | 0 | 1 | 2 | 3 | 4 |
|:----:|--:|--:|--:|--:|--:|
| E0 | 45.02 | 41.34 | 42.17 | 45.44 | 40.43 |
| E2 | 48.76 | 45.28 | 46.50 | 49.61 | 44.12 |

각 fold 에서 UTS +3~4 MPa 악화. fold std 변화 작음 (E0 2.00 → E2 2.07).

### 4.3 판정 (v2)

⚠️ **가설 1 일부 수정**: v2 에서 ΔYS=+1.01 로 4 그룹 중 **4위 (가장 약함)**. v1 에서 1위였던 것은 G4 placeholder 영향. 강도 지배 정보원은 **G4 scan** (ΔYS +4.6, ΔUTS +18).

✅ **가설 2 (B1.4 의존)**: 빌드별 분해는 v2 에서 미실행 — `analyze_per_build --experiment E2` 후 재검증 필요.

✅ **가설 3 (그룹 효과)**: 모든 속성 ΔRMSE > 0. 다만 단일 채널 ablation (E14–E22) 모두 noise 수준 (|ΔUTS|<0.55) → **센서는 redundancy 강한 collective effect**.

---

## 5. 빌드별 잔차 분해 (v2 미실행)

> v1 분해는 G4 placeholder baseline 기준이라 무효. v2 에서는 다음 명령으로 재생성 필요:
> ```bash
> ./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment E2
> # → Sources/pipeline_outputs/ablation/E2_no_sensor/per_build_analysis.md
> ```

가설:
- B1.4 (스패터/가스 유량 변화) 에서 가장 큰 타격 — 센서 변동성이 곧 공정 신호
- B1.2 (파라미터 다양성) 에서 fold std 확대 가능
- B1.3 (오버행), B1.5 (리코터 손상) 에서는 영향 작을 것 (CAD/DSCNN 가 지배)

---

## 6. 해석 및 후속 함의 (v2)

1. **센서는 4 그룹 중 가장 약함** — ΔUTS +3.97 (G4 의 22% 수준).
2. **단일 채널은 모두 noise** — E14–E22 ablation 결과 |ΔUTS|<0.55 (fold std ~2 이내). 7 개 센서가 강한 redundancy 를 가지며 "같은 공정 상태의 다른 관측" 으로 작동.
3. **경량화 후보**: E21 (가스 유량 3ch) + E22 (온도 2ch) 도 단독 marginal → 두 묶음을 동시에 제거 (5ch 제거, n=16) 해도 큰 손실 없을 것으로 예측. 후속 실험 권장.
4. **DSCNN 과의 직교성**: [E13](./PLAN_E13_combined.md) 에서 ΔUTS +10.06 ≈ ΔE1+ΔE2 (+10.11) → 두 그룹은 거의 독립.

---

## 7. 연관 문서

- 공통 설정: [PLAN.md](./PLAN.md)
- 센서 세부: [PLAN_sensor_subablation.md](./PLAN_sensor_subablation.md) (E14–E22)
- 조합: [PLAN_E13_combined.md](./PLAN_E13_combined.md) (E1 × E2)
- 종합 보고서: [FULL_REPORT.md](../../pipeline_outputs/ablation/FULL_REPORT.md) §5
