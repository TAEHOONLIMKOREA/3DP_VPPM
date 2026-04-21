# VPPM Feature Ablation 실험 계획

> **목적**: VPPM 모델의 21개 입력 피처 중 실제로 예측 성능에 기여하는 피처 그룹을 식별한다.
> 슈퍼복셀 단위 입력 피처를 소스(데이터 종류) 기준으로 4개 그룹으로 나누고,
> 각 그룹을 제거했을 때 4개 타겟(YS / UTS / UE / TE) RMSE 변화를 측정한다.
>
> **베이스라인**: [MODEL_SUMMARY.md](../../pipeline_outputs/MODEL_SUMMARY.md) — 21-feat 기준 RMSE (5-Fold CV)

---

## 1. 피처 그룹 정의

논문 Table A4 및 [Sources/vppm/config.py](../config.py) 정의를 기준으로 21개 피처를 4개 소스 그룹으로 묶는다.

| 그룹 | 피처 수 | 피처 인덱스 | 피처명 | 제거 시 남는 차원 |
|:----:|:------:|:----------:|--------|:---------------:|
| **G1. DSCNN** | 8 | 4–11 | seg_powder / seg_printed / seg_recoater_streaking / seg_edge_swelling / seg_debris / seg_super_elevation / seg_soot / seg_excessive_melting | 13 |
| **G2. Temporal Sensor (CSV)** | 7 | 12–18 | layer_print_time / top_gas_flow_rate / bottom_gas_flow_rate / module_oxygen / build_plate_temperature / bottom_flow_temperature / actual_ventilator_flow_rate | 14 |
| **G3. CAD / 좌표** | 3 | 1–3 | distance_from_edge / distance_from_overhang / build_height | 18 |
| **G4. 스캔 (Laser)** | 3 | 19–21 | laser_module / laser_return_delay / laser_stripe_boundaries | 18 |

> 참고: G4 의 `laser_return_delay`, `laser_stripe_boundaries` 는 현재 placeholder(0) — 실제 기여도는 사실상 `laser_module` 단독의 효과에 가깝다.

---

## 2. 실험 설계

### 2.1 실험 목록

| ID | 실험명 | 제거 그룹 | 사용 피처 수 | 가설 |
|:--:|-------|:--------:|:-----------:|------|
| E0 | **Baseline** | — | 21 | 기준 성능 (기존 결과 재사용) |
| E1 | No-DSCNN | G1 | 13 | DSCNN 결함 예측치가 주요 결함성 정보이므로 **UE/TE 악화 큼** |
| E2 | No-Sensor | G2 | 14 | 빌드 단위 공정 상태 반영 — 빌드 간 차이 설명력이 줄어 **전반적 악화** |
| E3 | No-CAD | G3 | 18 | 엣지/오버행 거리, 빌드 높이 — 형상 민감도 큰 **YS/UTS 에 영향** |
| E4 | No-Scan | G4 | 18 | placeholder 2개 포함 — **악화 미미** 예상 (유효 1개만 제거) |

### 2.2 제어 변수 (모든 실험 공통)

- 데이터: `Sources/pipeline_outputs/features/all_features.npz` (19,313 유효 슈퍼복셀)
- CV: 동일 5-Fold (seed=42, **샘플 단위 분할**)
- 모델: VPPM 2-layer MLP, hidden=128, dropout=0.1
- 학습 하이퍼파라미터: [config.py](../config.py) 의 기본값 (LR=1e-3, batch=1000, max_epoch=5000, patience=50)
- 손실: L1 (MAE), 정규화 공간 [-1, 1]
- 평가: 원본 스케일 RMSE, 샘플별 예측 집계는 "최소값" (기존과 동일)

> **주의**: 피처 제거 시 재정규화가 필요하다. 전체 21차원 통계로 정규화된 기존 `normalization.json` 을 그대로 사용하지 말고, **각 실험에서 남은 차원만으로 f_min/f_max 를 다시 계산**해 [-1, 1] 로 스케일한다.

### 2.3 실험당 산출물

각 실험은 4 속성 × 5 folds = **20 모델** 을 생성한다. 저장 위치:

```
Sources/pipeline_outputs/ablation/
├── E1_no_dscnn/
│   ├── models/        # vppm_{YS,UTS,UE,TE}_fold{0-4}.pt
│   ├── normalization.json
│   └── metrics.json
├── E2_no_sensor/
├── E3_no_cad/
├── E4_no_scan/
└── summary.md         # 4개 실험 통합 비교표
```

---

## 3. 구현 방식

### 3.1 피처 인덱스 상수 추가

`Sources/vppm/config.py` 에 그룹별 인덱스를 상수로 추가한다 (0-based).

```python
FEATURE_GROUPS = {
    "cad":     [0, 1, 2],                      # G3
    "dscnn":   [3, 4, 5, 6, 7, 8, 9, 10],      # G1
    "sensor":  [11, 12, 13, 14, 15, 16, 17],   # G2
    "scan":    [18, 19, 20],                   # G4
}
```

### 3.2 실행 스크립트

`Sources/vppm/ablation/run.py` 를 신규 작성한다. 골격:

```python
# 1) all_features.npz 로드
# 2) drop_group 에 해당하는 컬럼을 features 에서 제거
# 3) build_dataset() 을 새 features 로 호출 → 재정규화
# 4) 출력 디렉터리를 ablation/E*_*/ 로 바꿔 train_all() 호출
# 5) evaluate_all() 실행 → metrics.json 저장
```

CLI 사용 예:

```bash
./venv/bin/python -m Sources.vppm.ablation.run --experiment E1  # no DSCNN
./venv/bin/python -m Sources.vppm.ablation.run --experiment E2  # no sensor
./venv/bin/python -m Sources.vppm.ablation.run --experiment E3  # no CAD
./venv/bin/python -m Sources.vppm.ablation.run --experiment E4  # no scan
./venv/bin/python -m Sources.vppm.ablation.run --all            # 4개 연속
```

### 3.3 모델 입력 차원 대응

`VPPM(n_feats=...)` 는 이미 생성자에서 입력 차원을 받으므로 수정 불필요.
[run_pipeline.py:143](../run_pipeline.py#L143) 의 `run_train(n_feats=...)` 경로가 이미 `min(n_feats, features.shape[1])` 로 동작하니 ablation 용 경로도 동일 방식으로 구현한다.

---

## 4. 비교 및 해석

### 4.1 결과 요약 테이블 템플릿

| 실험 | 사용 피처 | YS RMSE (MPa) | UTS RMSE (MPa) | UE RMSE (%) | TE RMSE (%) |
|:----:|:--------:|:-------------:|:--------------:|:-----------:|:-----------:|
| E0 Baseline | 21 | 28.7 ± 0.6 | 60.7 ± 2.6 | 12.8 ± 0.3 | 15.5 ± 0.2 |
| E1 No-DSCNN | 13 |  ?  |  ?  |  ?  |  ?  |
| E2 No-Sensor | 14 |  ?  |  ?  |  ?  |  ?  |
| E3 No-CAD | 18 |  ?  |  ?  |  ?  |  ?  |
| E4 No-Scan | 18 |  ?  |  ?  |  ?  |  ?  |

### 4.2 해석 기준

- **ΔRMSE = E*i* − E0** 로 정의. 양수일수록 해당 그룹이 중요.
- **내재 측정오차** 선(YS 16.6 / UTS 15.6 / UE 1.73 / TE 2.92) 을 Reference 로 함께 표기한다.
- **Fold 별 분산**도 기록한다 — 그룹 제거 후 분산이 크게 늘면 "부분적으로만 기여하는 피처" 로 해석.
- 모든 타겟에서 ΔRMSE 가 0.5 MPa(또는 0.1 %) 이내이면 **해당 그룹은 현행 형태로는 유의미하지 않음** → 향후 다음 중 하나를 고려:
  1. 피처 공학을 다시 설계 (예: G4 placeholder 를 실제 값으로 구현)
  2. 해당 그룹 드롭 → 경량화된 모델 채택

### 4.3 추가 관찰 포인트

- **강도(YS/UTS) vs 연성(UE/TE)** 민감도 차이 — 논문에선 연성이 결함에 더 민감하다고 보고. G1 제거 시 UE/TE 가 YS/UTS 보다 더 악화되는지 확인.
- **수렴 에포크 변화** — 피처가 줄어 under-determined 이면 early-stop 이 더 일찍 걸릴 수 있다. `training_log.json` 에서 평균 에포크 확인.
- **빌드별 잔차** — 일부 그룹은 특정 빌드(B1.4 스패터, B1.5 리코터 손상)에서만 유의미할 수 있으니 빌드별 RMSE 분해를 권장.

---

## 5. 실행 순서

1. `Sources/vppm/config.py` 에 `FEATURE_GROUPS` 상수 추가.
2. `Sources/vppm/run_ablation.py` 작성 (피처 드롭 + 재정규화 + 학습 + 평가 래퍼).
3. E1 → E2 → E3 → E4 순으로 실행 (각 실험 약 20~30분 소요 예상, GPU 기준).
4. `Sources/pipeline_outputs/ablation/summary.md` 에 4.1 표를 채워 업데이트.
5. 필요 시 E5 (모든 placeholder 포함 피처 드롭), E6 (DSCNN + sensor 동시 제거 등 조합) 실험 확장.

---

## 6. 리스크와 한계

- **데이터 누출**: CV 분할은 반드시 `sample_ids` 기준이어야 한다 (기존 로직 유지).
- **재정규화 일관성**: 학습·검증·테스트 공통의 f_min/f_max 를 학습 데이터 전체 기준으로 계산하는 것은 논문과 동일한 관행. 폴드별로 다시 계산하는 변형은 본 실험에선 하지 않는다.
- **G4 의 한계**: placeholder 2개 포함이므로, G4 제거가 "스캔 데이터 전체 효과"를 의미하지는 않는다. 결과 해석 시 이 한계를 명시한다.
- **하이퍼파라미터 고정**: 차원이 줄면 최적 hidden 이 달라질 수 있지만, 비교 공정성을 위해 baseline 과 동일 구조(128)를 유지한다. 후속 실험에서 hidden tuning 을 분리 수행할 수 있다.
