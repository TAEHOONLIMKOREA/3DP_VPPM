# E4: No-Scan 실험 계획

> **공통 설정** (피처 그룹 정의, CV, 모델, 해석 기준) 은 [PLAN.md](./PLAN.md) 참조.
> 본 문서는 E4 실험 고유 정보만 기술.

---

## 1. 실험 정의

| 항목 | 값 |
|:-----|:---|
| **실험 ID** | E4 |
| **실험명** | No-Scan |
| **제거 그룹** | G4 Scan / Laser (3 채널) |
| **제거 피처 idx** | 18, 19, 20 |
| **사용 피처 수** | 21 → **18** |
| **출력 디렉터리** | `Sources/pipeline_outputs/ablation/E4_no_scan/` |

### 1.1 제거되는 3 개 Scan 채널

[features.py:39-41](../origin/features.py#L39-L41) 및 [features.py:105-116](../origin/features.py#L105-L116):

| idx | 채널 | 현재 구현 상태 | 물리적 의미 |
|:---:|:-----|:--------------|:-----------|
| 18 | laser_module            | **정상** (0/1 바이너리)     | 사용된 레이저 모듈 (1 or 2) |
| 19 | laser_return_delay      | **⚠️ placeholder = 0.0**   | 스트라이프 경계 통과 후 재스캔까지 시간 (냉각 proxy) |
| 20 | laser_stripe_boundaries | **⚠️ placeholder = 0.0**   | 스트라이프 경계 밀도 (melt-time Sobel RMS) |

> 3 채널 중 **2 채널이 placeholder 상수 0** — 사실상 `laser_module` 단독 제거 실험.
> 정식 구현은 [PLAN_G4_scan_reengineering.md](./PLAN_G4_scan_reengineering.md) 참조.

---

## 2. 가설

> Scan 그룹은 스트라이프 패턴·레이저 모듈 차이를 통해 **국소 냉각 속도 비등방성** 을 반영한다.
> 그러나 현재:
>
> 1. **ΔRMSE 미미 예상** — placeholder 2 개 포함이므로 유효 정보는 laser_module 하나뿐
> 2. laser_module 은 파트 단위 상수 (대부분 1 또는 2) — 풍부한 정보량 아님
> 3. 4 그룹 중 **가장 작은 영향** 예상

---

## 3. 구현

### 3.1 사전 요건

- `config.FEATURE_GROUPS["scan"] = [18, 19, 20]` — **이미 등록됨**
- `EXPERIMENTS["E4"]` — **이미 등록됨**

### 3.2 실행 명령

**호스트:**

```bash
./venv/bin/python -m Sources.vppm.ablation.run --experiment E4
./venv/bin/python -m Sources.vppm.ablation.run --experiment E4 --quick
```

**도커:**

```bash
cd docker/ablation/scan
./run.sh              # 기본 (GPU 3)
./run.sh --quick
```

### 3.3 산출물

```
Sources/pipeline_outputs/ablation/E4_no_scan/
├── experiment_meta.json
├── features/normalization.json
├── models/{vppm_*.pt, training_log.json}
└── results/{metrics_raw.json, metrics_summary.json,
           predictions_*.csv, correlation_plots.png, scatter_plot_uts.png}
```

---

## 4. 실제 결과 (2026-04-23 기준)

### 4.1 RMSE (원본 스케일)

| 속성 | E0 | **E4** | ΔRMSE | Naive | E4 vs Naive |
|:---:|:--:|:-----:|:-----:|:-----:|:-----------:|
| YS  | 28.66 ± 0.62 | 28.79 ± 0.61 | +0.13 | 33.91 | +15.1% |
| UTS | 60.72 ± 2.59 | 59.68 ± 2.96 | **−1.04** | 68.43 | **+12.8%** (baseline 대비 개선!) |
| UE  | 12.79 ± 0.27 | 12.94 ± 0.25 | +0.16 | 15.00 | +13.7% |
| TE  | 15.46 ± 0.20 | 15.60 ± 0.24 | +0.14 | 18.52 | +15.8% |

### 4.2 Fold 별 RMSE (UTS)

| Fold | 0 | 1 | 2 | 3 | 4 |
|:----:|--:|--:|--:|--:|--:|
| E0 | 61.26 | 60.75 | 63.22 | 62.53 | 55.86 |
| E4 | 54.39 | 60.87 | 60.07 | 63.47 | 59.60 |

fold 0 이 54.39 로 특히 낮음. fold std 2.96 > |ΔUTS| 1.04 이므로 개선은 **통계적으로 비유의**.

### 4.3 판정

✅ **가설 1 검증**: 모든 속성 ΔRMSE 가 0.2 MPa (또는 %) 이내. 4 그룹 중 **가장 작은 영향**.

❗ **예상 외 결과 — UTS 가 오히려 개선** (ΔUTS = −1.04):
  - placeholder 상수 0 이 정규화 후 [-1, 1] 경계값으로 들어가 **학습 gradient 를 왜곡** 했을 가능성
  - 즉 "무의미한 상수 열" 이 학습을 방해하는 **노이즈 피처** 로 작용
  - 단, fold std 2.96 대비 1.04 는 noise level — 재현성은 seed 반복 필요

---

## 5. 빌드별 잔차 분해

별도 실행 필요:

```bash
./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment E4
# → Sources/pipeline_outputs/ablation/E4_no_scan/per_build_analysis.md
```

### 5.1 예상 패턴

- 모든 빌드에서 거의 무변화 예상 (laser_module 이 파트 단위 상수이고 placeholder 들은 0).
- 만약 특정 빌드에서만 유의미한 ΔRMSE 가 나온다면 laser_module 선택의 빌드별 patter 차이.

---

## 6. 해석 및 후속 함의

1. **현 상태 (placeholder 포함) 의 Scan 그룹은 경량화 즉시 제거 대상**:
   - ΔUTS −1.04 = naïve 대비 reduction 이 더 큼 (12.8% vs baseline 11.3%)
   - ΔYS, ΔUE, ΔTE 모두 < 0.2 — 내재오차 대비 무시 가능
   - **18 피처 VPPM** 을 경량 기본 모델로 채택 검토 권장
2. **정식 scan 피처의 가치는 별개 실험** — [PLAN_G4_scan_reengineering.md](./PLAN_G4_scan_reengineering.md)
   의 E30~E33 완료 후 재판정.
3. **가능 시나리오 3가지**:
   - (a) 재구현 후 baseline v2 성능이 E0 보다 좋고 E31 ΔRMSE > 0 → **G4 유의, 재구현 가치 입증**
   - (b) 재구현 후에도 ΔRMSE ≈ 0 → **G4 본질 불필요, 영구 폐기 + 18-feat 채택**
   - (c) 재구현 후 baseline 이 E0 보다 악화 → **재구현 코드 버그 또는 Scan 데이터 자체가 해로움**

---

## 7. 연관 문서

- 공통 설정: [PLAN.md](./PLAN.md)
- **정식 재실험 계획**: [PLAN_G4_scan_reengineering.md](./PLAN_G4_scan_reengineering.md) — E30, E31, E32, E33
- 종합 보고서: [FULL_REPORT.md](../../pipeline_outputs/ablation/FULL_REPORT.md) §5, §8
