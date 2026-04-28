# E13: DSCNN + Sensor 조합 Ablation 실험 계획

> **목적**: DSCNN (G1) 과 Temporal Sensor (G2) 두 주요 피처 그룹이 상호 보완 관계인지,
> 아니면 독립적으로 작동하는지 규명한다.
>
> **배경 (v2)**: 두 그룹 ablation 결과:
> - E1 (no-DSCNN):  ΔUTS +6.14 / ΔUE +1.43 / ΔTE +1.60
> - E2 (no-sensor): ΔUTS +3.97 / ΔUE +0.83 / ΔTE +0.94
>
> 두 그룹 모두 제거했을 때 ΔRMSE 가 단순 **합산값에 가까우면 독립**, **훨씬 크면 상호보완** 관계.
>
> **v2 결과**: ΔE13 UTS = +10.06 ≈ ΔE1+ΔE2 = +10.11 → **시나리오 A (독립/orthogonal)**.

---

## 1. 실험 정의

### 1.1 실험 목록

| ID | 실험명 | 제거 그룹 | 사용 피처 수 | 가설 |
|:--:|-------|:--------:|:-----------:|------|
| E0  | Baseline             | —              | 21 | 기준 |
| E1  | No-DSCNN             | G1             | 13 | 기존 결과 재사용 |
| E2  | No-Sensor            | G2             | 14 | 기존 결과 재사용 |
| **E13** | **No-DSCNN+Sensor** | **G1 ∪ G2** | **6** | **단독 합보다 큰 악화 예상** |

제거 인덱스: `[3..10] ∪ [11..17]` = 총 15개. 남는 피처: CAD(3) + Scan(3) = 6개.

### 1.2 세 가지 해석 시나리오

| 시나리오 | ΔRMSE (E13) 패턴 | 해석 |
|:-------:|:----------------|:-----|
| A. 독립 | ΔE13 ≈ ΔE1 + ΔE2 | 두 그룹이 서로 다른 정보를 담고 있음 (orthogonal) |
| B. 중복 | ΔE13 < ΔE1 + ΔE2 | 두 그룹이 일부 같은 신호를 설명함 (redundant) |
| C. 보완 | ΔE13 > ΔE1 + ΔE2 | 한 쪽이 다른 쪽의 노이즈를 정규화함 (complementary) |

예시 — UTS 기준으로 **ΔE1+ΔE2 = +13.2 MPa**.
- ΔE13 ≈ +13 → 시나리오 A
- ΔE13 ≈ +9 → 시나리오 B
- ΔE13 ≈ +18 → 시나리오 C

### 1.3 추가 관찰 포인트

- **CAD+Scan 단독 학습 가능성**: E13 이 naive baseline 수준(UTS 68.4, UE 15.0, TE 18.5)으로 회귀하는지. 회귀하면 "결함/센서 데이터 없이는 예측 불가" 로 결론.
- **Fold std**: E1·E2 에서 YS std 가 2배로 증가했음. E13 에서 더 증가하면 "핵심 피처 부재 시 모델 불안정성 악화" 확인.
- **빌드별 잔차**: `analyze_per_build.py` 를 E13 에도 확장해 B1.4 의 UE/TE 가 어디까지 무너지는지 확인 (기존 E2 에서 +2.89, +3.45 → E13 에선?).

---

## 2. 구현

### 2.1 config.py 확장

`FEATURE_GROUPS` 에 조합 키 추가:

```python
# Sources/vppm/common/config.py (FEATURE_GROUPS 아래)
FEATURE_GROUPS_COMBO = {
    "dscnn_sensor": FEATURE_GROUPS["dscnn"] + FEATURE_GROUPS["sensor"],  # 8+7=15
}
```

### 2.2 run.py 확장

`EXPERIMENTS` 딕셔너리에 E13 엔트리 추가:

```python
# Sources/vppm/ablation/run.py
EXPERIMENTS = {
    "E1":  ("dscnn",        "No-DSCNN — DSCNN 8피처 제거"),
    "E2":  ("sensor",       "No-Sensor — Temporal 센서 7피처 제거"),
    "E3":  ("cad",          "No-CAD — CAD/좌표 3피처 제거"),
    "E4":  ("scan",         "No-Scan — 스캔 3피처 제거"),
    "E13": ("dscnn_sensor", "No-DSCNN+Sensor — G1∪G2 15피처 제거"),
}
```

`drop_feature_group()` 은 `FEATURE_GROUPS` 만 참조하므로, 조합 그룹도 동일 dict 에 병합:

```python
# config.py 끝에
FEATURE_GROUPS.update(FEATURE_GROUPS_COMBO)
```

나머지 학습·평가 파이프라인은 기존 코드 재사용 (피처 15개 제거 → 6-feat 모델로 자동 동작).

### 2.3 실행

```bash
./venv/bin/python -m Sources.vppm.ablation.run --experiment E13
./venv/bin/python -m Sources.vppm.ablation.run --rebuild-summary  # summary.md 갱신
```

### 2.4 빌드별 분석 확장

`analyze_per_build.py` 에 `--experiment` 플래그를 추가하거나, E13 용 사본을 만들어 per-build ΔRMSE 를 출력.

---

## 3. 결과 산출물

```
Sources/pipeline_outputs/ablation/
├── E13_no_dscnn_sensor/
│   ├── experiment_meta.json
│   ├── models/          # vppm_{YS,UTS,UE,TE}_fold{0-4}.pt (n_feats=6)
│   ├── results/
│   │   ├── metrics_raw.json, metrics_summary.json
│   │   ├── predictions_*.csv
│   │   └── correlation_plots.png, scatter_plot_uts.png
│   └── features/normalization.json
└── summary.md  (E13 행 추가됨)
```

### 3.1 v2 결과

| 속성 | E0 | E1 | E2 | **E13** | ΔE1+ΔE2 | ΔE13 | 시나리오 |
|:---:|:--:|:--:|:--:|:-------:|:-------:|:----:|:--------:|
| YS  | 24.28 | 25.97 | 25.29 | **27.21** | +2.70 | **+2.93** | A (≈) |
| UTS | 42.88 | 49.02 | 46.85 | **52.94** | +10.11 | **+10.06** | **A** (정확 일치) |
| UE  | 9.34  | 10.77 | 10.17 | **11.38** | +2.26 | **+2.04** | A/B (약간 redundant) |
| TE  | 11.27 | 12.86 | 12.21 | **13.74** | +2.54 | **+2.47** | A (≈) |

**판정 — 시나리오 A (독립/orthogonal)**: 모든 속성에서 ΔE13 ≈ ΔE1+ΔE2.
DSCNN 과 Sensor 는 거의 직교적인 정보를 운반하며, 어느 한 쪽이 다른 쪽의 손실을 보완하지 않는다.

**Naive 대비**: E13 (n_feats=6, CAD+Scan 만 사용) 도 naive RMSE 보다 모두 우수.
- YS 27.21 < 33.91 (naive), UTS 52.94 < 68.43, UE 11.38 < 15.00, TE 13.74 < 18.52
- → CAD 3 + Scan 3 = 6 피처만으로도 의미 있는 학습 가능 (특히 G4 scan 의 강한 신호 덕분).

---

## 4. 리소스 및 일정

- **학습 시간**: 4 속성 × 5 fold × ~1분 (GPU) = **~25분**
- **평가 + 리포트**: ~5분
- **총 소요**: 30분 이내
- **디스크**: ~20 MB (모델 체크포인트 + 결과)

---

## 5. 성공 기준

- [x] E13 학습 완료 (20 모델 모두 수렴 또는 early-stop)
- [x] `metrics_raw.json` 4속성 모두 기록
- [x] `summary.md` 에 E13 행 + ΔE13 열 자동 갱신
- [x] 시나리오 A/B/C 판정 (ΔE13 vs ΔE1+ΔE2 비율) 명시
- [x] Naive baseline 대비 초과 여부 확인

---

## 6. 리스크

- **수렴 실패**: 6-feat 밖에 안 남으므로 hidden=128 은 과도할 수 있음 — early-stop 에 맡기고, 만일 20 모델 중 다수가 1 epoch 후 plateau 면 hidden=32 로 재시도 고려.
- **Naive 초과**: E13 RMSE 가 naive 보다 크면 결론은 "이 6 피처는 학습 대비 노이즈"로 해석하고 끝낼 것. 추가 튜닝 유혹 금지 (비교 공정성 훼손).
- **하이퍼파라미터 고정**: baseline 과 동일 구조 유지 — hidden/lr 튜닝은 별도 실험 (Arch ablation) 으로 분리.
