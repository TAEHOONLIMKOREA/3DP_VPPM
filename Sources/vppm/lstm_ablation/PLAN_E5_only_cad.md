# E5: Only-CAD (CAD patch 분기 단독) 실험 계획

> **공통 설정** (가설, CV, 모델, 해석 기준) 은 [PLAN.md](./PLAN.md) 참조.
> 코드 변경 (6-flag 토글 모델) 은 [E3 §3.1](./PLAN_E3_only_v0_img.md#31-코드-변경) 참조.
> 본 문서는 E5 실험 고유 정보만 기술.

---

## 1. 실험 정의

| 항목 | 값 |
|:--|:--|
| **실험 ID** | E5 |
| **실험명** | Only-CAD (CAD geometry patch 분기 단독, `feat_static` 도 제거) |
| **유지 분기** | `branch_cad` |
| **제거 분기** | `feat_static`, `branch_v0`, `branch_v1`, `branch_sensor`, `branch_dscnn`, `branch_scan` |
| **MLP 입력 차원** | 86 → **8** (= 0 + 0 + 0 + 0 + 0 + **8** + 0) |
| **출력 디렉터리** | `Sources/pipeline_outputs/experiments/lstm_ablation/E5_only_cad/` |
| **flag** | `use_cad=True`, 그 외 모두 False (`use_static=False` 포함) |

### 1.1 유지되는 분기의 물리적 의미

CAD patch = SV 8×8 영역의 **2 채널 기하 정보** (HDF5 무관, 빌드 시 계산):

| 채널 | 의미 | 처리 |
|:-:|:--|:--|
| 0 | edge_proximity | `3.0 - dist_to_edge_mm`, `cad_mask` 픽셀곱, edge_saturation=3.0 mm |
| 1 | overhang_proximity | `71 - dist_to_overhang_layer`, `cad_mask` 픽셀곱, overhang_saturation=71 layers (수직-column) |

- 입력 캐시: `cad_patch_cache_{B}.h5` (`cache_cad_patch.py` 산출물)
- 형상: (B, T≤70, 2, 8, 8) — 2 채널 8×8 패치 시퀀스
- 인코더: spatial CNN (in=2 → d=32) → LSTM(d_hidden=16) → proj(8)

> CAD 분기는 **공정 데이터 0**, **결함 데이터 0** 인 순수 기하 정보. baseline 21-feat 의 정적 처리와 비교했을 때 본 실험은 **layer-축 시간성 처리** 만 추가됨.

---

## 2. 가설

> **CAD 단독** 시:
> - **5 개 단일 분기 중 가장 약한 standalone 성능 후보** → 결함/공정 시그널 부재로 인장 특성 변동을 직접 설명할 수 없음
> - 단, **vppm_baseline (21-feat)** 보다는 좋을 가능성 → baseline 도 dist_to_edge / dist_to_overhang 을 사용하지만 SV-수준 정적 평균이며, 본 실험은 layer-축 시퀀스 + 8×8 spatial 패턴 보존
> - **B1.3 (오버행 형상) 빌드에서 회복도 가장 좋음** → overhang_proximity 채널이 직접 신호. 다른 빌드에서는 거의 변화 없음
> - **YS / UTS 보다 UE / TE 회복도 더 낮음** → 결함 종류 정보 (UE/TE 결정) 가 CAD 에 거의 없음

### 2.1 정량 기대치

| 속성 | E0 (풀-스택) | vppm_baseline (21-feat) | E5 예상 RMSE | 회복도 |
|:--:|:--:|:--:|:--:|:--|
| YS  | 20.1 | 24.3 | 23.0 ~ 24.5 | E0 대비 60 % 손실, baseline 동등 |
| UTS | 28.5 | 42.9 | 36.0 ~ 41.0 | E0 대비 50 % 손실, baseline 보다 약간 좋음 |
| UE  |  6.5 |  9.3 |  8.5 ~  9.3 | baseline 동등 |
| TE  |  8.1 | 11.3 | 10.5 ~ 11.5 | baseline 동등 |

> CAD 단독은 baseline 21-feat (정적 평균) 의 시간성 확장 정도. 큰 개선 기대치 없음.

---

## 3. 구현

### 3.1 코드 변경

[E3 §3.1](./PLAN_E3_only_v0_img.md#31-코드-변경) 의 7-flag 토글 모델 그대로 사용. `run.py`:

```python
EXPERIMENTS["E5"] = dict(
    use_static=False,
    use_v0=False, use_v1=False,
    use_sensor=False, use_dscnn=False,
    use_cad=True,     use_scan=False,
    out_subdir="E5_only_cad",
    n_total_feats=8,
    kept=["branch_cad"],
)
```

### 3.2 dataset / dataloader

`load_septet_dataset` 그대로. CAD patch 캐시는 base 풀-스택 빌드 시 생성된 것 (`cache_cad_patch.py`) 재사용.

### 3.3 학습 hp (E0 / E1 / E2 / E3 / E4 동일)

E3 §3.3 와 동일. 입력 차원 8-d.

> 파라미터 카운트: CAD 분기 spatial CNN (~3k) + LSTM (~3k) + proj (~140) + MLP (~50k) ≈ 56k.
> **주의**: `feat_static` (build_height, laser_module) 도 제거되므로 CAD 분기가 build/laser 정보를 implicit 하게 학습할 수 있는지도 함께 측정.

### 3.4 실행 명령

```bash
./venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E5 --quick    # smoke
./venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E5            # full run (사용자)

# 도커
cd docker/lstm_ablation
docker compose run --rm e5
```

### 3.5 산출물

```
Sources/pipeline_outputs/experiments/lstm_ablation/E5_only_cad/
├── experiment_meta.json       # use_cad=True only (use_static=False), n_total_feats=8
├── features/
│   └── normalization.json     # 8-차원 재정규화 (static 통계 제외)
├── models/
│   ├── vppm_lstm_ablation_E5_{YS,UTS,UE,TE}_fold{0-4}.pt
│   └── training_log.json
└── results/
    ├── metrics_raw.json
    ├── metrics_summary.json
    ├── predictions_{YS,UTS,UE,TE}.csv
    ├── correlation_plots.png
    └── scatter_plot_uts.png
```

---

## 4. 판정 기준

### 4.1 1차 판정 — vs E0 (풀-스택)

| ΔRMSE 범위 | 판정 |
|:--|:--|
| Δ < 2σ | CAD 단독으로 풀-스택의 70 % 이상 회복 — **이상**. 풀-스택의 다른 분기들이 결국 CAD 에 추가 가치 없음을 의미 |
| 2σ ≤ Δ < 4σ | 정상 — CAD 단독으론 큰 한계. 결함/공정 정보 부재로 자연스러운 결과 |
| 4σ ≤ Δ | 매우 큰 정보 부족 — CAD 는 풀-스택 기여도에서 최하위. baseline 보다도 약간만 개선 |

### 4.2 2차 판정 — vs vppm_baseline

baseline 도 cad 정보 일부 (정적 dist_to_edge / dist_to_overhang 평균) 사용:

| RMSE_E5 vs RMSE_baseline | 판정 |
|:--|:--|
| E5 ≪ baseline | CAD layer-축 시퀀스 처리가 정적 평균보다 정보 추가 — 시간성 가치 입증 |
| E5 ≈ baseline | CAD 시간성 처리는 정적 평균 대비 추가 가치 없음 — geometric 정보의 시간성 변동이 인장 특성과 약한 상관 |
| E5 > baseline | CAD 단독은 baseline 보다도 못함. 본 분기는 다른 분기와의 결합으로만 가치 발현 |

### 4.3 빌드별 분해

predictions_*.csv 에서 B1.1~B1.5 별 RMSE.

- **B1.3 (오버행) RMSE 회복도 ≫ 평균** → overhang_proximity 채널이 핵심 신호 → CAD 분기는 형상 의존 빌드에서만 강함
- **B1.4, B1.5 회복도 평균 이하** → 형상 변화 없는 빌드에서는 CAD 가치 없음 → 본 분기의 작동 영역 제한적임을 입증
- B1.3 RMSE_E5 vs RMSE_E0: 1σ 이내라면 **CAD 단독으로 형상 의존 빌드 거의 풀-스택 회복**

### 4.4 spatial CNN 효과 검증

CAD 분기는 spatial CNN (8×8 패치 → 32-d) + LSTM 의 두 단계. 본 실험에서 모델 단순화 (예: spatial CNN 제거 → 패치 평균만 사용) 와의 비교는 별도 후속 실험 (E5b 가능):

| E5 (full CAD branch) | E5b (CAD without spatial CNN, patch mean) |
|:--|:--|
| spatial 패턴 보존 | 패치 평균 → baseline 21-feat 의 dist_to_edge / dist_to_overhang 과 거의 동등 |

> E5 가 E5b 대비 큰 개선 없으면 **spatial CNN 무용** 결론. 본 실험은 E5 만 실행하고 E5b 는 결과에 따라 후속.

---

## 5. 예상 결과 시나리오

### 5.1 시나리오 L (가장 가능성 높음 — CAD 빈약)

```
E0 풀-스택  : YS 20.1, UTS 28.5, UE 6.5, TE 8.1
baseline   : YS 24.3, UTS 42.9, UE 9.3, TE 11.3
E5 only-CAD: YS 24.0, UTS 40.0, UE 9.0, TE 11.0
ΔE5 (vs E0): +3.9, +11.5, +2.5, +2.9   (≫ 2σ, 가장 큰 ΔRMSE 후보)
```

**해석**: CAD 단독은 baseline 와 거의 동등. layer-축 시간성이 정적 평균 대비 미미한 개선만 추가. CAD 분기는 다른 분기 (특히 결함/공정) 와의 결합으로만 가치 발현.

### 5.2 시나리오 M (CAD 시간성 가치)

```
E5 only-CAD: YS 22.5, UTS 35.0, UE 8.0, TE 10.0
ΔE5        : +2.4, +6.5, +1.5, +1.9   (2-3σ)
```

**해석**: CAD 시간성 처리가 baseline 정적 처리보다 명확히 개선. 다른 분기와 결합 시 추가 정보 제공 가능.

### 5.3 시나리오 N (B1.3 강세)

```
B1.3 RMSE_E5 / RMSE_E0 ≈ 1.05 (오버행 빌드만 풀-스택과 거의 동등)
B1.1, B1.4 RMSE_E5 / RMSE_E0 ≈ 1.5 (형상 변화 없으면 CAD 빈약)
```

**해석**: CAD 분기는 **형상 의존 빌드 (B1.3) 에서만** 강한 신호. 풀-스택에서 CAD 의 가치는 B1.3 sample 비중에 한정. → 빌드별 모델 분기 검토 가능.

---

## 6. 후속 실험 분기

| 본 결과 | 다음 실험 |
|:--|:--|
| 시나리오 L (CAD 빈약) | E5 단독 결과 ≈ baseline 이면 CAD spatial CNN 무용 입증. 다음: cad d_embed sweep (4/8/16) 으로 표현력 한계 |
| 시나리오 M (CAD 시간성 가치) | CAD + 1 개 페어 (CAD+sensor, CAD+dscnn) 으로 결합 가치 측정 |
| 시나리오 N (B1.3 강세) | CAD 가 B1.3 핵심 → 빌드별 분기 모델 (B1.3 전용 CAD 가중치 ↑) 검토 |

---

## 7. 연관 문서

- 공통 설정: [PLAN.md](./PLAN.md)
- 시리즈 동료 실험: [E3 only_v0_img](./PLAN_E3_only_v0_img.md), [E4 only_dscnn](./PLAN_E4_only_dscnn.md), [E6 only_scan](./PLAN_E6_only_scan.md), [E7 only_sensor](./PLAN_E7_only_sensor.md)
- Base 풀-스택: [Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md)
- CAD patch 캐시 빌더: [Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache_cad_patch.py](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache_cad_patch.py)
- baseline 21-feat 결과 (참조선): [Sources/pipeline_outputs/experiments/vppm_baseline/results/metrics_raw.json](../../pipeline_outputs/experiments/vppm_baseline/results/metrics_raw.json)
