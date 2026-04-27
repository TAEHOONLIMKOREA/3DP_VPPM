# VPPM Feature Ablation — 개요 · 인덱스 · 공통 설정

> **목적**: VPPM 모델의 21 개 입력 피처 중 실제로 예측 성능에 기여하는 피처 그룹/채널을 식별한다.
> 슈퍼복셀 단위 입력 피처를 소스(데이터 종류) 기준으로 분해하고, 한 번에 한 그룹(또는 채널)을
> 제거했을 때 4 개 타겟(YS / UTS / UE / TE) RMSE 변화를 측정한다.
>
> **기준 모델 (E0 Baseline)**: 21-feat, 5-Fold CV — [results/vppm_origin/metrics_raw.json](../../pipeline_outputs/results/vppm_origin/metrics_raw.json)
>
> **종합 결과 보고서**: [FULL_REPORT.md](../../pipeline_outputs/ablation/FULL_REPORT.md) — 배경부터 해석·후속까지 14 섹션.

---

## 1. 실험 인덱스

실험 계획은 **실험 단위 (혹은 실험 군 단위) 로 개별 md 파일** 로 관리한다. 본 파일은 공통 설정과
인덱스 역할만 담당한다.

### 1.1 주요 그룹 ablation (E1–E4)

| ID | 계획 문서 | 제거 | n_feats | 핵심 결과 요약 |
|:--:|:----------|:-----|:-------:|:--------------|
| E1 | [PLAN_E1_no_dscnn.md](./PLAN_E1_no_dscnn.md)  | G1 DSCNN 8ch      | 13 | ΔUTS **+7.46**, UE 가 naive 수준 붕괴 |
| E2 | [PLAN_E2_no_sensor.md](./PLAN_E2_no_sensor.md) | G2 Sensor 7ch    | 14 | ΔYS **+2.51** 최대, B1.4 에서 특히 의존 |
| E3 | [PLAN_E3_no_cad.md](./PLAN_E3_no_cad.md)      | G3 CAD 3ch        | 18 | ΔRMSE 모두 < 0.7 (CAD 는 DSCNN 에 흡수됨) |
| E4 | [PLAN_E4_no_scan.md](./PLAN_E4_no_scan.md)    | G4 Scan 3ch*      | 18 | ΔUTS **−1.04** (placeholder 때문, 재구현 필요) |

\* E4 의 scan 그룹은 3 채널 중 2 개가 placeholder=0 상태로 학습됨.

### 1.2 조합 ablation (E13)

| ID | 계획 문서 | 제거 | n_feats | 핵심 결과 |
|:--:|:----------|:-----|:-------:|:---------|
| E13 | [PLAN_E13_combined.md](./PLAN_E13_combined.md) | G1 ∪ G2 (15ch) | 6 | **독립 (additive)** — ΔE13 ≈ ΔE1 + ΔE2, 모두 naive 이하 |

### 1.3 DSCNN 서브 ablation (E5–E12 + E23/E24)

단일 파일로 10 개 실험 계획:

- [PLAN_dscnn_subablation.md](./PLAN_dscnn_subablation.md) — **부분 실행 (E5~E8 완료, E9~E12·E23·E24 대기)**

| 범위 | 성격 | 상태 | 핵심 발견 / 질문 |
|:----|:-----|:----:|:--------------|
| E5–E8   | Normal 2 + Defect 2 단독 제거 | ✅ 완료 | **가설 반전**: Normal 2 채널 Critical (ΔUE +1.40~1.47), Defect 2 채널 Marginal |
| E9–E12  | Defect 4 채널 단독 제거 | ⏳ 대기 | E12 (excessive_melting) 가 B1.2 Keyhole 의 핵심인가? |
| E23     | 결함 6 채널 묶음 (normal 2개만 남김) | ⏳ 대기 | **결정적 실험**: 결함 정보는 Defect 채널에 집중되어 있는가? |
| E24     | Normal 2 채널 묶음 (defect 6개만 남김) | ⏳ 대기 | **결정적 실험**: Normal 채널이 결함 통합 신호로 기능하는가? |

> 잠정 결론: DSCNN 의 핵심 기여는 "어떤 결함이냐"보다 "결함이 얼마나 있느냐" — Normal 채널이 통합 결함률 신호로 작용 중.

### 1.4 센서 서브 ablation (E14–E22)

단일 파일로 9 개 실험 계획:

- [PLAN_sensor_subablation.md](./PLAN_sensor_subablation.md)

| 범위 | 성격 | 핵심 결과 |
|:----|:-----|:---------|
| E14–E20 | 7 개 채널 **단독** 제거 | 모든 채널이 **Marginal** — 단일 기여 없음 |
| E21     | 가스 유량 3 채널 묶음     | ΔUTS −0.94 (noise-level) |
| E22     | 온도 2 채널 묶음         | ΔUTS ±0 (무의미) |

> 결론: 센서는 **집단 효과** — 단독 채널은 marginal 하지만 E2 (전체 7 개 제거) 에서 ΔUTS +5.75.

### 1.5 스캔 서브 ablation (E30–E33, 정식 재실험 계획)

- [PLAN_G4_scan_reengineering.md](./PLAN_G4_scan_reengineering.md) — **실행 전** (PLAN §4 전제조건 선행 필요)

| ID | 제거 | 상태 |
|:--:|:----|:----:|
| E30 | — (Baseline v2, 21 피처 재학습) | 계획 완료 |
| E31 | scan 전체 3ch (v2)               | placeholder 상태 수치만 존재 |
| E32 | scan_return_delay (#20)         | 〃 |
| E33 | scan_stripe_boundaries (#21)    | 〃 |

### 1.6 실행 도커 인프라

실험군별 도커 디렉터리:

| 실험군 | 도커 경로 |
|:------|:---------|
| E1 (dscnn) | [docker/ablation/dscnn/](../../../docker/ablation/dscnn/) |
| E2 (sensor) | [docker/ablation/sensor/](../../../docker/ablation/sensor/) |
| E3 (cad)   | [docker/ablation/cad/](../../../docker/ablation/cad/) |
| E4 (scan)  | [docker/ablation/scan/](../../../docker/ablation/scan/) |
| E5–E12 + E23/E24 | [docker/ablation/dscnn_sub/](../../../docker/ablation/dscnn_sub/) |
| E14–E22    | [docker/ablation/sensor_sub/](../../../docker/ablation/sensor_sub/) |
| E31–E33    | [docker/ablation/scan_sub/](../../../docker/ablation/scan_sub/) |
| E1–E4 병렬 | `docker/ablation/run_all.sh` |

---

## 2. 피처 그룹 정의 (공통)

논문 Table A4 및 [config.py:126-155](../common/config.py#L126-L155) 기준, 21 개 피처를 4 개 소스 그룹으로 묶는다.

| 그룹 | 피처 수 | idx (0-based) | 내용 | 제거 후 n_feats |
|:----:|:------:|:-------------:|:----|:--------------:|
| **G1. DSCNN**              | 8 | 3–10  | 8 개 결함 세그멘테이션 클래스 | 13 |
| **G2. Temporal Sensor**    | 7 | 11–17 | 프린트 시간 / 유량 / 산소 / 온도 | 14 |
| **G3. CAD / 좌표**         | 3 | 0–2   | distance_edge / distance_overhang / build_height | 18 |
| **G4. 스캔 (Laser)**       | 3 | 18–20 | laser_module / return_delay* / stripe_boundaries* | 18 |

\* G4 의 return_delay 와 stripe_boundaries 는 현재 placeholder(0). 상세: [PLAN_G4_scan_reengineering.md](./PLAN_G4_scan_reengineering.md)

### 2.1 피처별 상세 정보

21 개 피처 전체 표는 [FULL_REPORT.md §2.2](../../pipeline_outputs/ablation/FULL_REPORT.md) 참조.

---

## 3. 제어 변수 (모든 실험 공통)

- **데이터**: [Sources/pipeline_outputs/features/all_features.npz](../../pipeline_outputs/features/) (36,047 슈퍼복셀, 6,373 유효 샘플)
- **CV**: 동일 5-Fold, seed=42, **샘플 단위 분할** — 모든 실험이 같은 분할 재사용
- **모델**: VPPM 2-layer MLP, hidden=128, dropout=0.1
- **학습 하이퍼파라미터**: [config.py](../common/config.py)
  - Adam (lr=1e-3, β=(0.9, 0.999), ε=1e-4)
  - batch=1000, MAX_EPOCHS=5000, early-stop patience=50
- **손실**: L1 (MAE), 정규화 공간 [-1, 1]
- **평가**: 원본 스케일 RMSE, 샘플별 예측 집계는 "최소값" (보수적 추정, 논문 Section 3.1)

### 3.1 재정규화 원칙

피처 제거 시 **반드시 재정규화** 한다:
- 전체 21 차원 통계로 정규화된 기존 `normalization.json` 을 **그대로 사용하지 말 것**
- 각 실험에서 **남은 차원만으로 f_min/f_max 를 재계산** 해 [-1, 1] 로 스케일
- 이는 [run.py](./run.py) 의 `build_dataset()` 재호출로 자동 처리

### 3.2 재현성

- `torch.manual_seed` 는 명시적으로 고정하지 않음 — fold 간 weight init 가 자연스럽게 다름 (표본 변동 반영)
- 같은 실험을 재실행하면 RMSE 가 ±1 MPa 수준에서 재현 — fold std 와 같은 크기
- 확실한 재현이 필요한 경우 seed 고정 + 3 회 반복 권장

---

## 4. 실험당 공통 산출물 구조

각 실험은 4 속성 × 5 folds = **20 모델**을 생성한다. 저장 레이아웃:

```
Sources/pipeline_outputs/ablation/
├── E{id}_no_{group}/
│   ├── experiment_meta.json       # exp_id, drop_group, dropped_idx, n_feats, n_samples
│   ├── features/
│   │   └── normalization.json     # 재정규화 통계
│   ├── models/
│   │   ├── vppm_{YS,UTS,UE,TE}_fold{0-4}.pt
│   │   └── training_log.json      # fold 별 수렴 epoch
│   └── results/
│       ├── metrics_raw.json       # fold별 + 평균 RMSE (원본 스케일)
│       ├── metrics_summary.json
│       ├── predictions_{YS,UTS,UE,TE}.csv
│       ├── correlation_plots.png
│       └── scatter_plot_uts.png
├── summary.md                      # 전 실험 자동 생성 요약 (--rebuild-summary)
└── FULL_REPORT.md                  # 수동 작성 종합 보고서
```

---

## 5. 해석 기준 (공통)

### 5.1 ΔRMSE 정의

ΔRMSE = E*i* − E0. **양수일수록 해당 그룹이 중요**. 부호 해석:

| 부호 | 의미 |
|:---:|:----|
| Δ > 0, 큼 | 해당 그룹이 **핵심** 기여 (E1, E2 사례) |
| 0 < Δ < 0.5 MPa | **Marginal** — 기여 있으나 작음 (E3 사례) |
| Δ ≈ 0 | 기여 없음 또는 noise (E14–E22 단독 사례) |
| Δ < 0 | **역효과** — 해당 그룹이 학습을 방해했음 (E4 placeholder 사례) |

### 5.2 통계적 유의성

- fold std (σ) 대비 |ΔRMSE| 비교:
  - |Δ| > 2σ : 유의
  - σ < |Δ| < 2σ : 경향성
  - |Δ| < σ : noise
- 정식 통계 검정 (paired fold bootstrap 등) 은 현재 미수행 — 후속 과제

### 5.3 내재 측정오차 참고선

| 속성 | 측정오차 | 의미 |
|:---:|--------:|:----|
| YS  | 16.6 MPa | 재측정 표준편차 — 물리적 RMSE 하한 |
| UTS | 15.6 MPa | 〃 |
| UE  | 1.73 %   | 〃 |
| TE  | 2.92 %   | 〃 |

Baseline RMSE 가 내재오차의 수 배 — UE 는 7.4× 로 가장 큰 격차 (개선 여지 큼).

### 5.4 빌드별 잔차 분해

일부 그룹은 특정 빌드에서만 유의미할 수 있다 (B1.4 스패터, B1.5 리코터 손상 등). 빌드별 분해:

```bash
./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment E{id}
# → Sources/pipeline_outputs/ablation/E{id}_no_{group}/per_build_analysis.md
```

---

## 6. 공통 리스크와 한계

- **데이터 누출**: CV 분할은 반드시 `sample_ids` 기준이어야 한다 (기존 로직 유지).
- **샘플 단위 CV 의 한계**: 같은 빌드 샘플이 train/val 에 섞임 — **빌드 간 일반화** 는 측정 불가.
  실제 배포 시엔 Leave-One-Build-Out (LOBO) CV 로 재검증 필요.
- **하이퍼파라미터 고정**: 피처 수가 줄면 최적 hidden 이 달라질 수 있으나, 비교 공정성을 위해
  baseline 과 동일 구조(128)를 유지. 후속 실험에서 hidden tuning 분리 수행 가능.
- **G4 의 placeholder 한계**: 피처 19, 20 이 상수 0 이므로 E4 는 사실상 `laser_module` 단독 효과만 측정.
  정식 해석은 [PLAN_G4_scan_reengineering.md](./PLAN_G4_scan_reengineering.md) 실행 후 가능.
- **5 빌드 밖 일반화 불가**: 결과는 본 데이터셋(SS 316L, Concept Laser M2, 특정 공정 윈도우) 에 한정.

---

## 7. 후속 실험 로드맵

| 우선순위 | 실험 | 계획 문서 | 상태 |
|:-------:|:----|:---------|:----:|
| 1 | DSCNN 서브 잔여 6 실험 (E9–E12, E23, E24) | [PLAN_dscnn_subablation.md](./PLAN_dscnn_subablation.md) | E5–E8 완료, 잔여 대기 — **즉시 실행 가능** |
| 2 | PLAN_G4 정식 실험 (E30–E33) | [PLAN_G4_scan_reengineering.md](./PLAN_G4_scan_reengineering.md) | 계획 완료, 구현 대기 |
| 3 | LOBO CV                         | 미계획 | — |
| 4 | E14–E22 + E5–E8 seed 반복      | 미계획 | — |
| 5 | Hidden size sweep              | 미계획 | — |

상세는 [FULL_REPORT.md §13](../../pipeline_outputs/ablation/FULL_REPORT.md) 참조.

---

## 8. 파일 구조 참고

```
Sources/vppm/ablation/
├── PLAN.md                              # (이 파일) — 개요·인덱스·공통 설정
├── PLAN_E1_no_dscnn.md                  # E1 계획 + 결과
├── PLAN_E2_no_sensor.md                 # E2 계획 + 결과
├── PLAN_E3_no_cad.md                    # E3 계획 + 결과
├── PLAN_E4_no_scan.md                   # E4 계획 + 결과
├── PLAN_E13_combined.md                 # E13 조합 계획
├── PLAN_dscnn_subablation.md            # E5–E12 + E23/E24 DSCNN 서브 계획 (실행 전)
├── PLAN_sensor_subablation.md           # E14–E22 센서 서브 계획 + 결과 요약
├── PLAN_G4_scan_reengineering.md        # E30–E33 스캔 재구현 계획 (실행 전)
├── run.py                               # 실행 러너 (--experiment / --all / --rebuild-summary)
├── analyze_per_build.py                 # 빌드별 잔차 분해 스크립트
└── __init__.py
```
