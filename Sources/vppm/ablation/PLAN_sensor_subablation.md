# E14–E20: Temporal Sensor 서브 채널 Ablation 실험 계획

> **목적**: E2 (no-sensor, ΔUTS +5.75 MPa) 의 센서 7채널 중 **어느 한 채널이 실제 기여자**
> 인지 규명한다.
>
> **가설**: 산소·가스 유량 계열(대기 환경)이 용융 품질을 지배하고, 시간·온도는 redundant.
> per-build 분석에서 B1.4 (스패터/가스 유량 변화 빌드) 에 센서 의존도가 집중된 것이
> 이 가설과 일치.

---

## 1. 센서 채널 정리

[features.py:32-38](../origin/features.py#L32-L38) 기준:

| Feature idx | 이름 | HDF5 경로 (`temporal/`) | 물리적 의미 | 예상 기여 |
|:-----------:|-----|:-----------------------|:-----------|:---------:|
| 11 | `layer_print_time`         | `layer_times`                 | 레이어별 프린트 시간 (s) — 파트 면적/복잡도 proxy | **중** |
| 12 | `top_gas_flow_rate`        | `top_flow_rate`               | 챔버 상단 가스 유량 (m/s) | **상** |
| 13 | `bottom_gas_flow_rate`     | `bottom_flow_rate`            | 챔버 하단 가스 유량 (m/s) | **상** |
| 14 | `module_oxygen`            | `module_oxygen`               | 레이저 모듈 내 산소 농도 (ppm) | **최상** |
| 15 | `build_plate_temperature`  | `build_plate_temperature`     | 빌드 플레이트 온도 (℃) | 하 |
| 16 | `bottom_flow_temperature`  | `bottom_flow_temperature`     | 하단 가스 온도 (℃) | 하 |
| 17 | `actual_ventilator_flow_rate` | `actual_ventilator_flow_rate` | 환풍기 실제 유량 (m/s) | 중 |

---

## 2. 실험 설계

### 2.1 실험 목록 (7개 서브 + 2개 그룹)

| ID | 제거 채널 | 사용 피처 수 | 의미 |
|:--:|----------|:-----------:|------|
| E14 | `layer_print_time` (11) | 20 | 레이어 소요 시간의 독자 기여 |
| E15 | `top_gas_flow_rate` (12) | 20 | 상단 가스 유량 단독 |
| E16 | `bottom_gas_flow_rate` (13) | 20 | 하단 가스 유량 단독 |
| E17 | `module_oxygen` (14) | 20 | **산소 농도 — 최고 관심** |
| E18 | `build_plate_temperature` (15) | 20 | 플레이트 온도 단독 |
| E19 | `bottom_flow_temperature` (16) | 20 | 가스 온도 단독 |
| E20 | `actual_ventilator_flow_rate` (17) | 20 | 환풍기 유량 단독 |
| E21 | `gas_flow` (12,13,17) | 18 | **가스 유량 3채널 일괄 제거** |
| E22 | `thermal` (15,16) | 19 | **열(온도) 2채널 일괄 제거** |

> 2단계 설계: 1단계 (E14~E20) 는 "채널 단독 기여도" 측정, 2단계 (E21·E22) 는
> "중복 가능성이 높은 채널을 묶어 함께 제거" 하여 개별 ablation 의 한계(한 채널만 빠져도
> 다른 채널이 정보를 보완) 를 보완한다.

### 2.2 해석 기준

| 판정 | 기준 |
|:----:|:-----|
| **Critical**    | ΔRMSE ≥ 50% × ΔE2 (단독 제거로 E2 효과의 절반 이상) |
| **Contributing**| 20% ≤ ΔRMSE < 50% × ΔE2 |
| **Marginal**    | ΔRMSE < 20% × ΔE2 (실질적 기여 없음) |
| **Redundant (via 2단계)** | E14~E20 단독은 Marginal 인데 E21 이나 E22 묶음 제거 시 ΔRMSE 가 E2 에 근접 → 중복 정보 |

구체적 기준선 (UTS 기준, ΔE2 = +5.75 MPa):
- Critical:  ΔUTS ≥ 2.88
- Contributing: 1.15 ≤ ΔUTS < 2.88
- Marginal: ΔUTS < 1.15

### 2.3 추가 관찰

- **B1.4 특이성**: 단독 ablation 중 하나라도 B1.4 에서 ΔRMSE 가 E2 수준 (UTS +7.92, UE +2.89, TE +3.45) 에 근접하면 그 채널이 **B1.4 특화** 정보원.
- **Fold std**: E2 에서 YS std 가 0.62→1.37 로 2배 증가한 것이 어느 채널에서 주로 발생하는지 분해.

---

## 3. 구현

### 3.1 config.py 에 서브 그룹 추가

```python
# Sources/vppm/common/config.py — FEATURE_GROUPS 아래
FEATURE_GROUPS_SENSOR_SUB = {
    "sensor_print_time":     [11],
    "sensor_top_flow":       [12],
    "sensor_bottom_flow":    [13],
    "sensor_oxygen":         [14],
    "sensor_plate_temp":     [15],
    "sensor_flow_temp":      [16],
    "sensor_ventilator":     [17],
    "sensor_gas_flow_all":   [12, 13, 17],   # E21: 유량 3채널
    "sensor_thermal_all":    [15, 16],        # E22: 온도 2채널
}
FEATURE_GROUPS.update(FEATURE_GROUPS_SENSOR_SUB)
```

### 3.2 run.py 확장

```python
# Sources/vppm/ablation/run.py — EXPERIMENTS 에 추가
EXPERIMENTS.update({
    "E14": ("sensor_print_time",  "No-PrintTime — 레이어 프린트 시간 제거"),
    "E15": ("sensor_top_flow",    "No-TopFlow — 상단 가스 유량 제거"),
    "E16": ("sensor_bottom_flow", "No-BottomFlow — 하단 가스 유량 제거"),
    "E17": ("sensor_oxygen",      "No-Oxygen — 산소 농도 제거"),
    "E18": ("sensor_plate_temp",  "No-PlateTemp — 플레이트 온도 제거"),
    "E19": ("sensor_flow_temp",   "No-FlowTemp — 가스 온도 제거"),
    "E20": ("sensor_ventilator",  "No-Ventilator — 환풍기 유량 제거"),
    "E21": ("sensor_gas_flow_all","No-GasFlowAll — 유량 3채널 제거"),
    "E22": ("sensor_thermal_all", "No-ThermalAll — 온도 2채널 제거"),
})
```

### 3.3 일괄 실행 CLI

```bash
# 순차 실행 (GPU 1대 가정)
for E in E14 E15 E16 E17 E18 E19 E20 E21 E22; do
  ./venv/bin/python -m Sources.vppm.ablation.run --experiment $E
done
./venv/bin/python -m Sources.vppm.ablation.run --rebuild-summary
```

혹은 `--all-sensor-sub` 같은 편의 플래그 추가 (optional).

### 3.4 별도 summary 생성

기존 `summary.md` 는 E1–E4 중심이므로, 센서 서브 결과는 별도 `sensor_sub_summary.md` 에 정리:

```markdown
# Sensor Sub-Channel Ablation Summary

| Exp | Channel | ΔYS | ΔUTS | ΔUE | ΔTE | 판정 |
|:---:|---------|:---:|:----:|:---:|:---:|:----:|
| E14 | print_time   | ? | ? | ? | ? | M/C/Cr |
| E15 | top_flow     | ? | ? | ? | ? | M/C/Cr |
| ...
| E2  | **전체 (ref)** | +2.51 | +5.75 | +1.13 | +1.16 | — |
```

`run.py` 에 `--summary-style sensor_sub` 같은 필터 옵션을 추가하거나, 별도 스크립트
`Sources/vppm/ablation/build_sensor_sub_summary.py` 신규 작성.

---

## 4. 결과 산출물

```
Sources/pipeline_outputs/ablation/
├── E14_no_sensor_print_time/
├── E15_no_sensor_top_flow/
├── E16_no_sensor_bottom_flow/
├── E17_no_sensor_oxygen/
├── E18_no_sensor_plate_temp/
├── E19_no_sensor_flow_temp/
├── E20_no_sensor_ventilator/
├── E21_no_sensor_gas_flow_all/
├── E22_no_sensor_thermal_all/
└── sensor_sub_summary.md
```

각 폴더 레이아웃은 기존 E1~E4 와 동일 (models / results / features / experiment_meta.json).

---

## 5. 리소스 및 일정

- **학습 시간**: 9 실험 × 4 속성 × 5 fold × ~1분 = **~3시간** (GPU)
- **디스크**: ~180 MB
- **연속 실행 가능**: for-loop 로 순차 실행, 1개씩 끝난 후 `--rebuild-summary` 반복 실행해도 무방

---

## 6. 성공 기준

- [x] 9개 실험 모두 완주
- [x] 서브 채널별 ΔRMSE 표 (4 속성) 완성
- [x] 각 채널 Marginal / Contributing / Critical 판정
- [x] 2단계(E21/E22) 결과로 개별-vs-묶음 중복도 평가
- [x] B1.4 에 특화된 채널 식별 (있다면)

---

## 7. 리스크

- **단일 피처 제거 효과가 noise-level (±0.3 MPa 이하)**: 모든 단독 실험이 Marginal 로 나올 가능성. 이 경우 E21·E22 결과가 핵심 → 묶음 단위로만 해석 가능하다고 결론.
- **Random seed 의존성**: 단독 실험 ΔRMSE 가 작을 때 5-fold 평균의 분산이 상대적으로 크게 보이므로, 필요시 seed 2~3개 반복 실행 (E14–E20 중 Marginal 판정 케이스만 한정).
- **Data leakage 없음 확인**: sensor 컬럼은 레이어 단위 aggregate 이므로 CV split (샘플 단위) 과 독립 — 기존 로직 그대로 안전.
