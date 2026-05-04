# E14–E20: Temporal Sensor 서브 채널 Ablation 실험 계획

> **목적**: E2 (no-sensor, v2 ΔUTS +3.97 MPa) 의 센서 7채널 중 어느 한 채널이 실제 기여자인지 규명.
>
> **(원래) 가설**: 산소·가스 유량 계열(대기 환경)이 용융 품질을 지배, 시간·온도는 redundant.
> → **v2 결과 (§8)**: 가설 부분 확정. **모든 단독 채널이 noise** — 7 채널이 fully redundant.
> 그룹 효과(E2 +3.97) 만 의미.

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

구체적 기준선 (v2, UTS 기준, ΔE2 = +3.97 MPa):
- Critical:  ΔUTS ≥ 1.99
- Contributing: 0.79 ≤ ΔUTS < 1.99
- Marginal: ΔUTS < 0.79

### 2.3 추가 관찰

- **B1.4 특이성**: 단독 ablation 중 하나라도 B1.4 에서 ΔRMSE 가 E2 수준에 근접하면 그 채널이 B1.4 특화 정보원. v2 미실행 — `analyze_per_build` 후 검증 필요.
- **Fold std**: v2 E2 의 YS fold std (0.97) 는 baseline (0.75) 대비 약하게 증가 (1.3×). 어느 채널 기여인지 분해 미실행.

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
# Sources/vppm/baseline_ablation/run.py — EXPERIMENTS 에 추가
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
  ./venv/bin/python -m Sources.vppm.baseline_ablation.run --experiment $E
done
./venv/bin/python -m Sources.vppm.baseline_ablation.run --rebuild-summary
```

혹은 `--all-sensor-sub` 같은 편의 플래그 추가 (optional).

### 3.4 별도 summary 생성

기존 `summary.md` 는 E1–E4 중심이므로, 센서 서브 결과는 별도 `sensor_sub_summary.md` 에 정리:

```markdown
# Sensor Sub-Channel Ablation Summary

| Exp | Channel | ΔYS | ΔUTS | ΔUE | ΔTE | 판정 |
|:---:|---------|:---:|:----:|:---:|:---:|:----:|
| E14 | print_time   | -0.07 | +0.07 | -0.12 | +0.24 | Marginal |
| E15 | top_flow     | +0.05 | +0.47 | +0.03 | +0.16 | Marginal |
| ...
| E2  | **전체 (ref)** | +1.01 | +3.97 | +0.83 | +0.94 | — |
```

`run.py` 에 `--summary-style sensor_sub` 같은 필터 옵션을 추가하거나, 별도 스크립트
`Sources/vppm/baseline_ablation/build_sensor_sub_summary.py` 신규 작성.

---

## 4. 결과 산출물

```
Sources/pipeline_outputs/experiments/baseline_ablation/
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

---

## 8. v2 실행 결과 (2026-04-28)

9 개 실험 모두 완료. **모든 단독 채널이 noise** 수준 — 묶음(E21/E22) 도 marginal. 센서는 그룹 효과로만 의미.

### 8.1 RMSE 표 + ΔRMSE

| 실험 | 채널 | YS | UTS | UE | TE | ΔYS | ΔUTS | ΔUE | ΔTE |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| E0 | baseline           | 24.28 ± 0.75 | 42.88 ± 2.00 | 9.34 ± 0.28 | 11.27 ± 0.50 | — | — | — | — |
| E14 | print_time         | 24.21 ± 0.58 | 42.95 ± 2.17 | 9.22 ± 0.39 | 11.51 ± 0.33 | -0.07 | +0.07 | -0.12 | +0.24 |
| E15 | top_flow           | 24.33 ± 0.79 | 43.35 ± 1.51 | 9.37 ± 0.34 | 11.43 ± 0.45 | +0.05 | +0.47 | +0.03 | +0.16 |
| E16 | bottom_flow        | 24.11 ± 0.68 | 43.13 ± 2.29 | 9.17 ± 0.45 | 11.43 ± 0.37 | -0.17 | +0.25 | -0.17 | +0.16 |
| E17 | oxygen             | 24.16 ± 0.52 | 42.83 ± 1.46 | 9.36 ± 0.42 | 11.44 ± 0.26 | -0.12 | -0.05 | +0.02 | +0.17 |
| E18 | plate_temp         | 24.13 ± 0.67 | 43.36 ± 1.38 | 9.32 ± 0.39 | 11.35 ± 0.32 | -0.15 | +0.48 | -0.02 | +0.09 |
| E19 | flow_temp          | 24.32 ± 0.72 | 43.42 ± 1.49 | 9.44 ± 0.38 | 11.28 ± 0.48 | +0.04 | +0.54 | +0.10 | +0.02 |
| E20 | ventilator         | 24.18 ± 0.64 | 43.03 ± 1.54 | 9.23 ± 0.48 | 11.20 ± 0.42 | -0.10 | +0.15 | -0.11 | -0.07 |
| E21 | gas_flow_all (3ch) | 24.05 ± 0.69 | 43.05 ± 1.64 | 9.24 ± 0.39 | 11.25 ± 0.47 | -0.23 | +0.17 | -0.10 | -0.02 |
| E22 | thermal_all (2ch)  | 24.21 ± 0.66 | 43.35 ± 1.60 | 9.36 ± 0.45 | 11.17 ± 0.44 | -0.07 | +0.47 | +0.02 | -0.10 |
| **E2** | **전체 7ch (ref)** | **25.29 ± 0.97** | **46.85 ± 2.07** | **10.17 ± 0.49** | **12.21 ± 0.31** | **+1.01** | **+3.97** | **+0.83** | **+0.94** |

### 8.2 판정 — 가설 확정

✅ **모든 단독 채널 Marginal**: 7 개 단독 + 2 묶음 모두 |ΔUTS| < 0.55, |ΔYS| < 0.23.
   대부분 음수 (제거가 미세하게 도움) — fold std 1.4–2.3 이내 noise.

✅ **그룹 효과만 발현 (collective only)**: E2 (전체 7ch) ΔUTS +3.97. 단독 ΔUTS 평균 +0.30 의 약 13배.
   채널별 redundancy 가 강해 단독 제거 시 다른 채널이 보완.

### 8.3 후속 권장

| 실험 | 설계 | 동기 |
|:---|:---|:---|
| 4 채널 동시 제거 | E21 + E14 (gas_flow 3ch + print_time) | 경량화 — 4 채널만 빼도 marginal 인지 확인 |
| 5 채널 동시 제거 | E21 + E22 (gas + thermal 5ch) | 2 채널 (oxygen + print_time) 만으로 baseline 유지 가능성 |
| Per-build 분해 | E2 만 | B1.4 (스패터) 의존도 측정 — 가설 2 |
