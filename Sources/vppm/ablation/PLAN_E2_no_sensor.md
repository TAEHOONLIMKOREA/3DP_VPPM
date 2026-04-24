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
| YS  | 28.66 ± 0.62 | 31.17 ± 1.37 | **+2.51** | 33.91 | +8.1% |
| UTS | 60.72 ± 2.59 | 66.47 ± 1.73 | **+5.75** | 68.43 | **+2.9%** (거의 naive) |
| UE  | 12.79 ± 0.27 | 13.92 ± 0.30 | **+1.13** | 15.00 | +7.2% |
| TE  | 15.46 ± 0.20 | 16.61 ± 0.16 | **+1.16** | 18.52 | +10.3% |

### 4.2 Fold 별 RMSE (UTS)

| Fold | 0 | 1 | 2 | 3 | 4 |
|:----:|--:|--:|--:|--:|--:|
| E0 | 61.26 | 60.75 | 63.22 | 62.53 | 55.86 |
| E2 | 65.53 | 66.69 | 69.24 | 66.91 | 63.99 |

E2 UTS fold std (1.73) 는 **오히려 E0 (2.59) 보다 작음** — 성능이 naive 에 가까워지면서 fold 간 편차 압축.

### 4.3 판정

✅ **가설 1 검증**: **ΔYS = +2.51 로 4 그룹 중 최대**. ΔUTS 는 E1 (+7.46) 다음 2위. 강도 지배 정보원 확인.

✅ **가설 2 검증**: 빌드별 분해에서 **B1.4 가 가장 큰 타격** (per-build ΔUTS +7.92, B1.1·B1.2 다음).
  B1.4 는 스패터/가스 유량 변화 빌드 → 센서 없으면 핵심 변동성 설명 불가.

✅ **가설 3 검증**: 모든 속성 ΔRMSE > 0. DSCNN 대비 UE/TE 의 악화는 작음 (E1 대비 절반 수준).

---

## 5. 빌드별 잔차 분해

전체 상세 → [per_build_analysis.md](../../pipeline_outputs/ablation/per_build_analysis.md)

### 5.1 UTS ΔRMSE 빌드별 하이라이트

| Build | ΔUTS | 해석 |
|:-----:|-----:|:----|
| B1.1 | +4.72 | 기준 조건, +46% 악화 |
| B1.2 | **+8.15** | 파라미터 다양성 — 센서 없이는 fold 간 편동 2× |
| B1.3 | +1.07 | 오버행 — CAD 가 지배적, 센서 영향 작음 |
| **B1.4** | **+7.92** | 스패터/가스 변화 — **센서 의존도 최대** (ΔUE +2.89, ΔTE +3.45) |
| B1.5 | +1.13 | 리코터 손상 — DSCNN 이 지배, 센서 거의 불필요 |

### 5.2 Fold std 증가 원인

E2 YS fold std 가 baseline 0.62 → 1.37 (2× 증가). 분해 결과:
- B1.2 per-build std: 1.10 → 2.37 (전체 증가분의 90% 이상이 B1.2 기여)
- 다른 빌드의 std 는 거의 변화 없음

> 재생성:
> ```bash
> ./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment E2
> ```

---

## 6. 해석 및 후속 함의

1. **센서 전체 제거는 강도 예측에 치명적** (특히 B1.4).
2. **단일 채널 단위에서는 모두 marginal** — [E14~E22 서브 실험](./PLAN_sensor_subablation.md) 에서 확인됨.
   각 채널이 "같은 물리량의 다른 noisy 관측" 이라 **집단 효과** 로만 기능.
3. 경량화 시도: E21 (가스 유량 3채널) + E22 (온도 2채널) 를 모두 빼고 `layer_print_time` +
   `module_oxygen` 만 남기는 실험은 미수행 → 후속 권장.
4. **DSCNN 과의 보완 관계**: [E13](./PLAN_E13_combined.md) 에서 두 그룹이 대체로 독립 (additive)
   으로 판정. 즉 sensor 제거를 DSCNN 이 대체해주지 않음.

---

## 7. 연관 문서

- 공통 설정: [PLAN.md](./PLAN.md)
- 센서 세부: [PLAN_sensor_subablation.md](./PLAN_sensor_subablation.md) (E14–E22)
- 조합: [PLAN_E13_combined.md](./PLAN_E13_combined.md) (E1 × E2)
- 종합 보고서: [FULL_REPORT.md](../../pipeline_outputs/ablation/FULL_REPORT.md) §5
