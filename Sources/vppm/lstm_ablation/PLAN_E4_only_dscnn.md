# E4: Only-DSCNN (DSCNN 8-class 분기 단독) 실험 계획

> **공통 설정** (가설, CV, 모델, 해석 기준) 은 [PLAN.md](./PLAN.md) 참조.
> 코드 변경 (6-flag 토글 모델) 은 [E3 §3.1](./PLAN_E3_only_v0_img.md#31-코드-변경) 참조.
> 본 문서는 E4 실험 고유 정보만 기술.

---

## 1. 실험 정의

| 항목 | 값 |
|:--|:--|
| **실험 ID** | E4 |
| **실험명** | Only-DSCNN (DSCNN 8-class 이상 분류 분기 단독, `feat_static` 도 제거) |
| **유지 분기** | `branch_dscnn` |
| **제거 분기** | `feat_static`, `branch_v0`, `branch_v1`, `branch_sensor`, `branch_cad`, `branch_scan` |
| **MLP 입력 차원** | 86 → **8** (= 0 + 0 + 0 + 0 + **8** + 0 + 0) |
| **출력 디렉터리** | `Sources/pipeline_outputs/experiments/lstm_ablation/E4_only_dscnn/` |
| **flag** | `use_dscnn=True`, 그 외 모두 False (`use_static=False` 포함) |

### 1.1 유지되는 분기의 물리적 의미

DSCNN = **분말 도포 직후** (`visible/1`) 카메라 이미지에 대한 사전 학습된 deep semantic CNN 의 8-class 분류 결과 (HDF5 경로: `slices/segmentation_results/{0-11}` 중 8 채널 선별).

| Idx | HDF5 class | Paper name |
|:-:|:-:|:--|
| 0 | 0 | Powder |
| 1 | 1 | Printed |
| 2 | 3 | Recoater Streaking |
| 3 | 5 | Edge Swelling |
| 4 | 6 | Debris |
| 5 | 7 | Super-Elevation |
| 6 | 8 | Soot |
| 7 | 10 | Excessive Melting |

- 입력 캐시: `crop_stacks_dscnn_{B}.h5`
- 형상: (B, T≤70, 8) — SV 별 layer-축 8-채널 anomaly 확률 시퀀스
- 인코더: per-channel `_GroupLSTMBranch` (8 채널 각각 LSTM(d_hidden=16) → proj(1) 평균) → 최종 d_embed=8

> DSCNN 채널은 **이미 v1 image 로부터 결함 정보가 추출된 압축 표현**. 본 실험은 사실상 "v1 image 의 결함 정보 단독" 이 인장 특성을 어디까지 예측하는지를 측정한다.

---

## 2. 가설

> **DSCNN 단독** 시:
> - **5 개 단일 분기 중 가장 강한 standalone 성능 후보** → DSCNN 은 이미 v1 raw image 의 결함 시그널을 supervised 로 압축한 표현이므로 정보 밀도 최고
> - **B1.5 (리코터 손상) 빌드에서 회복도 가장 높음** → "Recoater Streaking" 채널이 직접 신호로 들어감
> - **UE/TE 보다 UTS/YS 회복도 높음** → 결함 종류 (UE/TE) 보다 결함 밀도/면적 (YS/UTS) 정보가 픽셀 레벨 분류 결과에 더 직접 반영
> - **풀-스택 (E0) 의 90 % 이상 회복 가능성**: DSCNN 채널이 다른 모든 분기와 redundant 일 수 있음 (특히 v1, scan 정보 일부 포함)

### 2.1 정량 기대치

| 속성 | E0 (풀-스택) | vppm_baseline (21-feat) | E4 예상 RMSE | 회복도 |
|:--:|:--:|:--:|:--:|:--|
| YS  | 20.1 | 24.3 | 21.0 ~ 22.5 | E0 대비 80-95 % |
| UTS | 28.5 | 42.9 | 30.0 ~ 33.0 | E0 대비 75-90 % |
| UE  |  6.5 |  9.3 |  6.8 ~  7.5 | E0 대비 80-95 % |
| TE  |  8.1 | 11.3 |  8.5 ~  9.5 | E0 대비 75-95 % |

> 단일 분기 isolation 시리즈 중 **가장 좋은 standalone 성능** 을 보일 후보. 풀-스택 (86-d) 까지 17.4 % 비중에 불과하지만 사전 학습된 압축이라 정보 밀도가 높음.

---

## 3. 구현

### 3.1 코드 변경

[E3 §3.1](./PLAN_E3_only_v0_img.md#31-코드-변경) 의 7-flag 토글 모델 그대로 사용. `run.py`:

```python
EXPERIMENTS["E4"] = dict(
    use_static=False,
    use_v0=False, use_v1=False,
    use_sensor=False, use_dscnn=True,
    use_cad=False,    use_scan=False,
    out_subdir="E4_only_dscnn",
    n_total_feats=8,
    kept=["branch_dscnn"],
)
```

### 3.2 dataset / dataloader

`load_septet_dataset` 그대로. 모든 캐시 로드되지만 모델 forward 가 미사용 입력 무시.

### 3.3 학습 hp (E0 / E1 / E2 / E3 동일)

E3 §3.3 와 동일. 입력 차원 8-d 로 매우 작아 MLP fc1 (256-h) 이 표현력 충분.

> 파라미터 카운트: ~ DSCNN 분기 ~9k + MLP ~50k = ~59k. 6,373 SV 대비 ~108 SV/param. 과적합 위험 작음.
> **주의**: `feat_static` (build_height, laser_module) 도 제거되므로 DSCNN 분기가 build/laser 정보를 implicit 하게 학습할 수 있는지도 함께 측정.

### 3.4 실행 명령

```bash
./venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E4 --quick    # smoke
./venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E4            # full run (사용자)

# 도커 (사용자 풀런)
cd docker/lstm_ablation
docker compose run --rm e4
```

### 3.5 산출물

```
Sources/pipeline_outputs/experiments/lstm_ablation/E4_only_dscnn/
├── experiment_meta.json       # use_dscnn=True only (use_static=False), n_total_feats=8
├── features/
│   └── normalization.json     # 8-차원 재정규화 (static 통계 제외)
├── models/
│   ├── vppm_lstm_ablation_E4_{YS,UTS,UE,TE}_fold{0-4}.pt    (20개)
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
| Δ < 1σ | DSCNN 단독으로 풀-스택 회복 → **DSCNN = 풀-스택의 핵심 정보원** (강력한 시나리오 G) |
| 1σ ≤ Δ < 2σ | DSCNN 단독으로 풀-스택의 80-90 % 회복 — DSCNN 압축 표현이 매우 강력 |
| 2σ ≤ Δ < 3σ | 정상 — DSCNN 단독으론 정보 부족하지만 5 개 중 최상위권 가능 |
| 3σ ≤ Δ | DSCNN 단독으론 부족 — v1 raw image 가 더 풍부 (DSCNN 의 supervised 압축 한계) |

### 4.2 2차 판정 — E4 vs E2 (no-cameras) 정합성

E2 의 RMSE 와 E4 의 RMSE 비교:

| 관계 | 해석 |
|:--|:--|
| RMSE_E4 ≈ RMSE_E2 | DSCNN 만으로 E2 (sensor+dscnn+cad+scan, 54-d) 와 동일 → sensor/cad/scan 모두 redundant 와 v1 결함 정보 |
| RMSE_E4 > RMSE_E2 | DSCNN 만으론 부족, sensor/cad/scan 의 추가 정보가 **유의** (정상) |
| RMSE_E4 < RMSE_E2 | DSCNN 만으로 더 잘함 → **이상**. E2 의 cad/scan/sensor 가 학습 잡음 추가했을 가능성. 모델 수렴 동학 재검토 필요 |

### 4.3 빌드별 분해

predictions_*.csv 에서 B1.1~B1.5 별 RMSE.

- **B1.5 (리코터 손상) 회복도 ≫ 평균** → "Recoater Streaking" 채널이 핵심 신호로 입증
- **B1.2 (Keyhole/LOF) 회복도 ≫ 평균** → "Excessive Melting" + "Super-Elevation" 채널이 melt mode 결함 잘 포착
- **B1.4 (스패터/가스) 회복도 평균 이하** → "Soot"/"Debris" 채널이 본 빌드의 결함을 충분히 잡지 못함 → DSCNN 클래스 셋의 한계

---

## 5. 예상 결과 시나리오

### 5.1 시나리오 I (가장 강한 가설 — DSCNN 우위)

```
E0 풀-스택    : YS 20.1, UTS 28.5, UE 6.5, TE 8.1
E4 only-DSCNN: YS 21.5, UTS 31.0, UE 7.0, TE 8.7
ΔE4          : +1.4, +2.5, +0.5, +0.6   (1-2σ)
```

**해석**: DSCNN 단독으로 풀-스택의 80-90 % 회복. 사전 학습된 supervised 압축 표현의 정보 밀도가 raw 멀티-분기 합산을 능가함. **DSCNN 만으로 충분히 좋은 모델 가능** → 학습/추론 비용 ↓ (다른 캐시 ~1 GB 불필요).

### 5.2 시나리오 J (정상 — 유의 ΔRMSE)

```
E4 only-DSCNN: YS 22.5, UTS 33.0, UE 7.4, TE 9.3
ΔE4          : +2.4, +4.5, +0.9, +1.2   (≥ 2σ)
```

**해석**: DSCNN 정보는 풀-스택 단독으로 60-75 % 회복. 5 개 단일 분기 중 가장 강하지만 다른 분기 (특히 sensor) 보완 필요.

### 5.3 시나리오 K (DSCNN 빈약 — supervised 압축 한계)

```
E4 only-DSCNN: YS 24.5, UTS 38.0, UE 8.5, TE 10.5
ΔE4          : +4.4, +9.5, +2.0, +2.4   (≫ 2σ)
```

**해석**: DSCNN 8-class 압축은 v1 raw image 의 정보 손실 큼. 사전 학습 시 본 task 와 다른 클래스 셋 사용으로 인한 mismatch. → DSCNN 채널 보강 (12-class 전체 사용) 또는 v1 raw 재도입 (E5+v1 페어) 검토.

---

## 6. 후속 실험 분기

| 본 결과 | 다음 실험 |
|:--|:--|
| 시나리오 I (DSCNN 우위) | DSCNN + 1 개 분기 페어 실험으로 보완 분기 식별 (예: E4+sensor 으로 어디까지 회복하는지) |
| 시나리오 J (정상) | E3-E7 모두 풀런 후 standalone 랭킹 표 작성 → top-2 분기 페어 ablation 후속 |
| 시나리오 K (DSCNN 빈약) | 12-class 전체 DSCNN 사용 또는 v1 raw image 재도입. 또는 DSCNN d_embed sweep (4/8/16) |

---

## 7. 연관 문서

- 공통 설정: [PLAN.md](./PLAN.md)
- 시리즈 동료 실험: [E3 only_v0_img](./PLAN_E3_only_v0_img.md), [E5 only_cad](./PLAN_E5_only_cad.md), [E6 only_scan](./PLAN_E6_only_scan.md), [E7 only_sensor](./PLAN_E7_only_sensor.md)
- 보완 실험: [E2 no_cameras](./PLAN_E2_no_cameras.md) (양쪽 카메라 제거 시 DSCNN 단독 가치)
- Base 풀-스택: [Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md)
- DSCNN 클래스 매핑: `config.DSCNN_FEATURE_MAP`
