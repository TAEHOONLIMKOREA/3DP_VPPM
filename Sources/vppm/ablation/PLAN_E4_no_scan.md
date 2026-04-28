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

[features.py:39-41](../origin/features.py#L39-L41) 및 [scan_features.py](../origin/scan_features.py):

| idx | 채널 | 물리적 의미 |
|:---:|:-----|:-----------|
| 18 | laser_module            | 사용된 레이저 모듈 (1 or 2) |
| 19 | laser_return_delay      | 1mm 커널 내 melt-time max−min (재방문 시간 = 냉각 시간 proxy) |
| 20 | laser_stripe_boundaries | melt-time 맵의 Sobel RMS (스트라이프 경계 밀도) |

---

## 2. 가설

> Scan 그룹은 스트라이프 패턴·레이저 경로를 통해 **국소 냉각 속도 비등방성** 을 반영한다.
> 따라서 제거 시:
>
> 1. **냉각 의존 물성 (UTS, 연성) 에 영향** 예상
> 2. 스트라이프 경계가 결함 응집 부위와 상관되므로 **연신율 (UE/TE) 도 영향** 가능
> 3. baseline 대비 ΔRMSE > 0 (성능 악화) 이면 G4 가 유의미한 정보 운반자임을 확인

---

## 3. 구현

### 3.1 사전 요건

- `config.FEATURE_GROUPS["scan"] = [18, 19, 20]` — **이미 등록됨**
- `EXPERIMENTS["E4"]` — **이미 등록됨**
- `Sources/pipeline_outputs/features/all_features.npz` 의 #19, #20 std > 0 (실 구현)

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

## 4. 실제 결과

> Phase 5 (전체 ablation 재실행) 완료 후 [FULL_REPORT.md](../../pipeline_outputs/ablation/FULL_REPORT.md) 와
> [summary.md](../../pipeline_outputs/ablation/summary.md) 에서 v2 결과 확인.

### 4.1 빌드별 잔차 분해

```bash
./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment E4
# → Sources/pipeline_outputs/ablation/E4_no_scan/per_build_analysis.md
```

빌드별 패턴 가설:
- **B1.4 (스패터, 가스 유량 변화)**: 스캔 경로 변동성이 클 가능성 → 가장 큰 영향 예상
- **B1.5 (리코터 손상)**: 정상 스캔 패턴 가정 가능 → 영향 작을 것
- **B1.3 (오버행)**: 스트라이프 경계 효과가 형상에 의존 → 중간

---

## 5. 연관 문서

- 공통 설정: [PLAN.md](./PLAN.md)
- 종합 보고서: [FULL_REPORT.md](../../pipeline_outputs/ablation/FULL_REPORT.md)
- 스캔 서브 채널 분해: [docker/ablation/scan_sub/README.md](../../../docker/ablation/scan_sub/README.md)
  (E31~E33 — `laser_module` / `return_delay` / `stripe_boundaries` 단독 영향)
