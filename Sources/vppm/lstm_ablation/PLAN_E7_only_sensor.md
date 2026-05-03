# E7: Only-Sensor (Sensor 7-field 1D-CNN 분기 단독) 실험 계획

> **공통 설정** (가설, CV, 모델, 해석 기준) 은 [PLAN.md](./PLAN.md) 참조.
> 코드 변경 (6-flag 토글 모델) 은 [E3 §3.1](./PLAN_E3_only_v0_img.md#31-코드-변경) 참조.
> 본 문서는 E7 실험 고유 정보만 기술.

---

## 1. 실험 정의

| 항목 | 값 |
|:--|:--|
| **실험 ID** | E7 |
| **실험명** | Only-Sensor (Sensor 7-field per-field 1D-CNN 분기 단독, `feat_static` 도 제거) |
| **유지 분기** | `branch_sensor` |
| **제거 분기** | `feat_static`, `branch_v0`, `branch_v1`, `branch_dscnn`, `branch_cad`, `branch_scan` |
| **MLP 입력 차원** | 86 → **28** (= 0 + 0 + 0 + **28** + 0 + 0 + 0) |
| **출력 디렉터리** | `Sources/pipeline_outputs/experiments/lstm_ablation/E7_only_sensor/` |
| **flag** | `use_sensor=True`, 그 외 모두 False (`use_static=False` 포함) |

### 1.1 유지되는 분기의 물리적 의미

Sensor = 7 개 layer-축 시계열 채널 (HDF5 `temporal/*` 기반):

| Idx | 필드명 | 물리량 |
|:-:|:--|:--|
| 0 | `layer_times` | 레이어별 가공 시간 (s) |
| 1 | `top_flow_rate` | 챔버 상단 유량 |
| 2 | `bottom_flow_rate` | 챔버 하단 유량 |
| 3 | `module_oxygen` | 모듈 산소 농도 |
| 4 | `build_plate_temperature` | 빌드 플레이트 온도 |
| 5 | `bottom_flow_temperature` | 하단 유량 온도 |
| 6 | `actual_ventilator_flow_rate` | 실제 환기 유량 |

- 입력 캐시: `sensor_cache_{B}.h5`
- 형상: (B, T≤70, 7) — layer-축 7-채널 시계열
- 인코더: `_PerFieldConv1DBranch` — 채널별 독립 1D-CNN (in=1, hidden=16, kernel=5) → proj(4) → concat → 7×4 = 28-d

> Sensor 분기는 5 개 단일 분기 중 **가장 큰 임베딩 (28-d)** + **가장 풍부한 시간성** 정보. 단, **per-SV 공간 정보 없음** — sensor 데이터는 layer-단위라서 같은 layer 의 모든 SV 가 동일 sensor 값 공유.

---

## 2. 가설

> **Sensor 단독** 시:
> - **단일 분기 isolation 시리즈 중 회복도 1-2 위 후보** (DSCNN 과 경합) → 28-d 임베딩 + 모든 layer 시간성 정보 + 글로벌 공정 상태 시그널
> - **빌드 간 분리 강함** → 빌드별 sensor 분포 차이 명확 (B1.4 가스 유량 변화, B1.2 melt 모드 vs nominal). 빌드 분류는 거의 100 % 가능. 다만 **빌드 내 sample 변동** 은 sensor 만으로 잡기 어려움 (sample 별 위치 정보 없음)
> - **B1.4 (가스 유량 변화) 빌드에서 회복도 가장 높음** → 가스 유량 자체가 sensor 채널
> - **YS / UTS 회복도 ≈ UE / TE** → sensor 는 멀티-속성 (모든 인장 특성) 에 글로벌하게 영향
> - 풀-스택 (E0) 의 75-90 % 회복 가능성

### 2.1 정량 기대치

| 속성 | E0 (풀-스택) | vppm_baseline (21-feat) | E7 예상 RMSE | 회복도 |
|:--:|:--:|:--:|:--:|:--|
| YS  | 20.1 | 24.3 | 21.0 ~ 23.0 | E0 대비 80-95 % |
| UTS | 28.5 | 42.9 | 30.0 ~ 33.5 | E0 대비 75-90 % |
| UE  |  6.5 |  9.3 |  6.8 ~  7.8 | E0 대비 75-95 % |
| TE  |  8.1 | 11.3 |  8.5 ~  9.5 | E0 대비 75-95 % |

> baseline 21-feat 도 sensor 일부 (예: layer_times, oxygen) 사용했으나 정적 평균 처리. 본 실험은 **layer-축 시간성 1D-CNN** 으로 동일 정보를 풍부하게 인코딩.

---

## 3. 구현

### 3.1 코드 변경

[E3 §3.1](./PLAN_E3_only_v0_img.md#31-코드-변경) 의 7-flag 토글 모델 그대로 사용. `run.py`:

```python
EXPERIMENTS["E7"] = dict(
    use_static=False,
    use_v0=False, use_v1=False,
    use_sensor=True, use_dscnn=False,
    use_cad=False,    use_scan=False,
    out_subdir="E7_only_sensor",
    n_total_feats=28,
    kept=["branch_sensor"],
)
```

### 3.2 dataset / dataloader

`load_septet_dataset` 그대로. Sensor 캐시는 base `lstm_dual_img_4_sensor_7` 또는 풀-스택에서 생성된 것 재사용 (`config.LSTM_FULL86_CACHE_SENSOR_DIR`).

### 3.3 학습 hp

E3 §3.3 와 동일.

> 파라미터 카운트: Sensor 분기 (7 채널 × per-field 1D-CNN ~1.2k + 7 proj ~140 + ~9k) ≈ 11k + MLP ~58k = ~69k. 6,373 SV 대비 ~92 SV/param.
> **주의**: `feat_static` (build_height, laser_module) 도 제거되므로 Sensor 분기가 build/laser 정보를 implicit 하게 학습할 수 있는지도 함께 측정 (sensor 채널 자체에 빌드 분류 시그널이 강해 implicit 학습 용이 예상).

### 3.4 실행 명령

```bash
./venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E7 --quick    # smoke
./venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E7            # full run (사용자)

# 도커
cd docker/lstm_ablation
docker compose run --rm e7
```

### 3.5 산출물

```
Sources/pipeline_outputs/experiments/lstm_ablation/E7_only_sensor/
├── experiment_meta.json       # use_sensor=True only (use_static=False), n_total_feats=28
├── features/
│   └── normalization.json     # 28-차원 재정규화 (static 통계 제외)
├── models/
│   ├── vppm_lstm_ablation_E7_{YS,UTS,UE,TE}_fold{0-4}.pt
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
| Δ < 1σ | Sensor 단독으로 풀-스택 회복 → **시나리오 R** (sensor 가 핵심 정보원). 다른 분기 모두 redundant |
| 1σ ≤ Δ < 2σ | Sensor 단독으로 풀-스택의 80-90 % 회복 — **가장 강한 단일 분기 후보** |
| 2σ ≤ Δ < 3σ | 정상 — sensor 는 글로벌 공정 정보, 잘 작동하지만 per-sample 변동에 한계 |
| 3σ ≤ Δ | sensor 단독 빈약 — 가능성 낮음 |

### 4.2 2차 판정 — E7 vs E4 (DSCNN 단독) 비교

5 개 단일 분기 중 회복도 1-2 위 경합 후보:

| 관계 | 해석 |
|:--|:--|
| RMSE_E7 < RMSE_E4 | **글로벌 공정 시계열** > **결함 분류** — 빌드별 분리가 인장 특성 예측의 핵심 |
| RMSE_E7 ≈ RMSE_E4 | 두 정보 도메인이 유사한 회복도. 결합 (E7+E4 페어) 으로 강한 성능 기대 |
| RMSE_E7 > RMSE_E4 | DSCNN 결함 정보가 sensor 글로벌 정보보다 강함. 픽셀-레벨 결함이 핵심 시그널 |

### 4.3 빌드별 분해

predictions_*.csv 에서 B1.1~B1.5 별 RMSE.

- **B1.4 (가스 유량 변화) 회복도 ≫ 평균** → 가스 유량 자체가 sensor 채널 → 빌드 정보 직접 반영
- **B1.2 (Keyhole/LOF) 회복도 ≫ 평균** → 빌드별 다른 공정 파라미터 (출력/속도) 가 layer_times 등 sensor 채널과 상관
- **B1.3 (오버행) 회복도 평균** → 형상 변화는 sensor 시그널 약함
- **B1.5 (리코터 손상) 회복도 평균** → 리코터 결함은 sensor 에서 직접 미관측 (v1 image 가 핵심 시그널)

### 4.4 빌드 간 vs 빌드 내 분해

Sensor 단독은 빌드별 분류는 잘하지만 빌드 내 sample 변동은 약함. 따라서:

```python
# 같은 빌드 내 RMSE vs 빌드 간 평균 차이
intra_build_rmse = np.mean([rmse(build) for build in builds])     # 빌드 내 변동
total_rmse                                                        # 전체 RMSE
```

| 관계 | 해석 |
|:--|:--|
| total ≈ intra_build | sensor 가 빌드 내 변동도 잘 잡음 — 매우 강한 정보원 |
| total < intra_build | 빌드 분류만 잘함 (빌드 평균 인장 특성 예측) — sensor 가 빌드 정보 압축에 그침 |

### 4.5 채널 ablation 후속 (E7 결과 강하면)

7 채널 중 어느 채널이 핵심인지 후속 실험 (E7a-g, 채널별 leave-one-out) 가능. 단 1 차에는 E7 만 실행.

---

## 5. 예상 결과 시나리오

### 5.1 시나리오 R (가장 강한 가설 — Sensor 우위)

```
E0 풀-스택    : YS 20.1, UTS 28.5, UE 6.5, TE 8.1
E7 only-Sensor: YS 21.0, UTS 30.5, UE 6.9, TE 8.6
ΔE7          : +0.9, +2.0, +0.4, +0.5   (1σ 안팎)
```

**해석**: Sensor 단독으로 풀-스택의 90-95 % 회복 (`feat_static` 없이도). 글로벌 공정 시계열 정보가 인장 특성 예측의 핵심. **다른 모든 분기는 정체된 누적 개선분 일부만 담당** (E1/E2 시나리오 A 와 정합). 모델 단순화 가능 (sensor 분기 단독, 28-d).

### 5.2 시나리오 S (정상 — Sensor 강하지만 부족)

```
E7 only-Sensor: YS 22.5, UTS 33.0, UE 7.4, TE 9.3
ΔE7          : +2.4, +4.5, +0.9, +1.2   (2-3σ)
```

**해석**: Sensor 단독으로 풀-스택의 75-85 % 회복. 단일 분기 시리즈 중 강한 후보지만 결함 정보 (DSCNN, v1) 보완 필수.

### 5.3 시나리오 T (Sensor 빈약 — 빌드 분류만)

```
E7 only-Sensor: YS 24.0, UTS 38.0, UE 8.5, TE 10.5
intra_build_rmse << total_rmse   (빌드 내 변동 잘 못 잡음)
```

**해석**: Sensor 는 빌드별 분리만 가능, 빌드 내 sample 변동에 약함. → SV-수준 spatial 정보 (v0/v1/dscnn/cad/scan) 가 핵심 → sensor 의 가치 제한적.

---

## 6. 후속 실험 분기

| 본 결과 | 다음 실험 |
|:--|:--|
| 시나리오 R (Sensor 우위) | minimal model (sensor 분기 단독 28-d, optional `feat_static` 재추가) 을 새 base 로 두고 다른 분기 추가 가치 재검증 |
| 시나리오 S (정상) | E7 + DSCNN (E4) 페어 실험 — 두 강한 분기 결합 시 풀-스택 회복도 |
| 시나리오 T (Sensor 빈약) | sensor 채널별 leave-one-out (E7a-g) 으로 dominant 채널 식별. 또는 sensor d_per_field sweep (2/4/8) |

---

## 7. E3 ~ E7 종합 비교 (시리즈 결과 정리 시)

5 개 단일 분기 standalone 성능 랭킹 표 (시리즈 모두 풀런 후 작성):

| Rank | 분기 | UTS RMSE | YS RMSE | 회복도 (UTS) | 비고 |
|:-:|:-:|:-:|:-:|:-:|:--|
| 1 | TBD (예상: sensor 또는 dscnn) | ? | ? | ? | 글로벌 공정 vs 압축 결함 |
| 2 | TBD | ? | ? | ? |
| 3 | v0 (img) | ? | ? | ? | raw 카메라 |
| 4 | scan | ? | ? | ? | 간접 공정 |
| 5 | cad | ? | ? | ? | 순수 geometry |

> 본 표 작성은 E3-E7 모두 풀런 (사용자 실행) 후. PLAN.md 인덱스에 결과 요약 추가.

---

## 8. 연관 문서

- 공통 설정: [PLAN.md](./PLAN.md)
- 시리즈 동료 실험: [E3 only_v0_img](./PLAN_E3_only_v0_img.md), [E4 only_dscnn](./PLAN_E4_only_dscnn.md), [E5 only_cad](./PLAN_E5_only_cad.md), [E6 only_scan](./PLAN_E6_only_scan.md)
- Base 풀-스택: [Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md)
- Sensor 캐시 (`lstm_dual_img_4_sensor_7`): `config.LSTM_FULL86_CACHE_SENSOR_DIR`
- TEMPORAL_FEATURES: `Sources/vppm/common/config.py:68-77`
- 관련 실험 (sensor 단독 1D-CNN 비교 base): [Sources/vppm/lstm_dual_img_4_sensor_7_dscnn_8](../lstm_dual_img_4_sensor_7_dscnn_8/)
