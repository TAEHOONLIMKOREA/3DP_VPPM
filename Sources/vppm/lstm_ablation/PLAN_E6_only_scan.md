# E6: Only-Scan (Scan patch 분기 단독) 실험 계획

> **공통 설정** (가설, CV, 모델, 해석 기준) 은 [PLAN.md](./PLAN.md) 참조.
> 코드 변경 (6-flag 토글 모델) 은 [E3 §3.1](./PLAN_E3_only_v0_img.md#31-코드-변경) 참조.
> 본 문서는 E6 실험 고유 정보만 기술.

---

## 1. 실험 정의

| 항목 | 값 |
|:--|:--|
| **실험 ID** | E6 |
| **실험명** | Only-Scan (Scan path patch 분기 단독, `feat_static` 도 제거) |
| **유지 분기** | `branch_scan` |
| **제거 분기** | `feat_static`, `branch_v0`, `branch_v1`, `branch_sensor`, `branch_dscnn`, `branch_cad` |
| **MLP 입력 차원** | 86 → **8** (= 0 + 0 + 0 + 0 + 0 + 0 + **8**) |
| **출력 디렉터리** | `Sources/pipeline_outputs/experiments/lstm_ablation/E6_only_scan/` |
| **flag** | `use_scan=True`, 그 외 모두 False (`use_static=False` 포함) |

### 1.1 유지되는 분기의 물리적 의미

Scan patch = SV 8×8 영역의 **2 채널 스캔 경로 정보** (HDF5 `scans/{layer}` 기반, 빌드 시 패치화):

| 채널 | 의미 | 처리 |
|:-:|:--|:--|
| 0 | return_delay | layer 내 위치별 레이저 재방문 지연 (s), saturation=0.75 s, mask 미적용 |
| 1 | stripe_boundaries | 스캔 stripe 경계 위치 (binary), 0=nominal, mask 미적용 |

- 입력 캐시: `scan_patch_cache_{B}.h5` (`cache_scan_patch.py` 산출물)
- 형상: (B, T≤70, 2, 8, 8) — 2 채널 8×8 패치 시퀀스
- 인코더: spatial CNN (in=2 → d=32) → LSTM(d_hidden=16) → proj(8)

> Scan 분기는 **레이저 운영 패턴** 정보. 결함 직접 정보 없으나, return_delay 가 짧으면 잔열 누적 → keyhole, stripe_boundaries 가 SV 내 위치하면 LOF/표면 결함 가능성. **간접 공정 시그널** 성격.

---

## 2. 가설

> **Scan 단독** 시:
> - **CAD 보다 약간 좋음** 가능성 → return_delay 는 잔열 (= melt mode) 의 직접 인자, stripe_boundaries 는 LOF 위치 시그널. CAD 의 순수 형상 정보보다 인장 특성과 직접 상관
> - **B1.4 (스패터/가스 유량 변화), B1.2 (Keyhole/LOF) 빌드에서 강함** → return_delay 가 melt mode 결함 직접 인자
> - **UE / TE 회복도 낮음** → 결함 종류 결정에는 표면 image (v1, dscnn) 가 핵심
> - 풀-스택 (E0) 의 50-65 % 회복 가능성 — sensor / dscnn 단독보다 분명히 약함

### 2.1 정량 기대치

| 속성 | E0 (풀-스택) | vppm_baseline (21-feat) | E6 예상 RMSE | 회복도 |
|:--:|:--:|:--:|:--:|:--|
| YS  | 20.1 | 24.3 | 22.5 ~ 24.0 | E0 대비 60-70 % 손실 |
| UTS | 28.5 | 42.9 | 34.0 ~ 39.0 | E0 대비 50-65 % 손실 |
| UE  |  6.5 |  9.3 |  8.2 ~  9.0 | E0 대비 60-70 % 손실 |
| TE  |  8.1 | 11.3 | 10.0 ~ 11.0 | E0 대비 60 % 손실 |

> Scan 은 baseline 21-feat 와 **정보 도메인 다름** (baseline 은 scan 특성 미사용). 따라서 baseline 과의 비교는 도메인 ↔ 시간성 두 변수가 동시에 변동하므로 단독 해석 어려움.

---

## 3. 구현

### 3.1 코드 변경

[E3 §3.1](./PLAN_E3_only_v0_img.md#31-코드-변경) 의 7-flag 토글 모델 그대로 사용. `run.py`:

```python
EXPERIMENTS["E6"] = dict(
    use_static=False,
    use_v0=False, use_v1=False,
    use_sensor=False, use_dscnn=False,
    use_cad=False,    use_scan=True,
    out_subdir="E6_only_scan",
    n_total_feats=8,
    kept=["branch_scan"],
)
```

### 3.2 dataset / dataloader

`load_septet_dataset` 그대로. Scan patch 캐시는 base 풀-스택 빌드 시 생성된 것 (`cache_scan_patch.py`) 재사용.

### 3.3 학습 hp

E3 §3.3 와 동일.

> 파라미터 카운트: Scan 분기 spatial CNN (~3k) + LSTM (~3k) + proj (~140) + MLP (~50k) ≈ 56k.
> **주의**: `feat_static` (build_height, laser_module) 도 제거되므로 Scan 분기가 build/laser 정보를 implicit 하게 학습할 수 있는지도 함께 측정.

### 3.4 실행 명령

```bash
./venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E6 --quick    # smoke
./venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E6            # full run (사용자)

# 도커
cd docker/lstm_ablation
docker compose run --rm e6
```

### 3.5 산출물

```
Sources/pipeline_outputs/experiments/lstm_ablation/E6_only_scan/
├── experiment_meta.json       # use_scan=True only (use_static=False), n_total_feats=8
├── features/
│   └── normalization.json     # 8-차원 재정규화 (static 통계 제외)
├── models/
│   ├── vppm_lstm_ablation_E6_{YS,UTS,UE,TE}_fold{0-4}.pt
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
| Δ < 1σ | Scan 단독으로 풀-스택 회복 → **이상** (가능성 매우 낮음). 다른 분기 모두 redundant 의미 |
| 1σ ≤ Δ < 2σ | Scan 시간성 처리가 매우 강력. CAD 보다 명백히 우수 |
| 2σ ≤ Δ < 4σ | 정상 — 단일 분기로서의 한계 |
| 4σ ≤ Δ | Scan 단독 빈약 — return_delay/stripe_boundaries 시그널이 인장 특성과 약한 직접 상관 |

### 4.2 2차 판정 — E6 vs E5 (CAD 단독) 비교

같은 8-d 임베딩, 같은 spatial CNN+LSTM 구조, 같은 입력 형상 (B, T, 2, 8, 8). 직접 비교 가능:

| 관계 | 해석 |
|:--|:--|
| RMSE_E6 < RMSE_E5 | Scan 정보 (process 간접) > CAD 정보 (순수 geometry) — **공정 시그널 우위** 입증 |
| RMSE_E6 ≈ RMSE_E5 | 둘 다 단독으로 빈약. spatial CNN+LSTM 구조의 한계 (또는 양쪽 모두 supplementary 분기) |
| RMSE_E6 > RMSE_E5 | Scan 정보가 CAD 보다 약함 — return_delay/stripe_boundaries 의 인장 특성 직접 상관 약함 |

### 4.3 빌드별 분해

predictions_*.csv 에서 B1.1~B1.5 별 RMSE.

- **B1.4 (스패터/가스 유량) 회복도 ≫ 평균** → return_delay 가 가스 유량과 함께 잔열 거동 결정 → Scan 분기의 작동 영역
- **B1.2 (Keyhole/LOF) 회복도 ≫ 평균** → return_delay 가 melt mode 직접 인자
- **B1.3 (오버행) 회복도 평균** → Scan 은 형상 의존성 약함
- **B1.5 (리코터 손상) 회복도 평균 이하** → Scan 분기는 리코터 결함 미반영

### 4.4 spatial CNN 효과 검증

CAD (E5) 와 동일하게 Scan 도 spatial CNN + LSTM 두 단계. 패치 평균 비교 (E6b 후속) 가능.

---

## 5. 예상 결과 시나리오

### 5.1 시나리오 O (가장 가능성 높음 — Scan 정보 부족)

```
E0 풀-스택  : YS 20.1, UTS 28.5, UE 6.5, TE 8.1
E6 only-Scan: YS 23.5, UTS 37.0, UE 8.5, TE 10.5
ΔE6        : +3.4, +8.5, +2.0, +2.4   (≫ 2σ)
```

**해석**: Scan 단독은 풀-스택의 50-65 % 회복. CAD (E5) 와 비슷하거나 약간 우수. 다른 분기 (특히 sensor / dscnn) 의 보완 필수.

### 5.2 시나리오 P (Scan 시간성 가치)

```
E6 only-Scan: YS 21.8, UTS 32.0, UE 7.5, TE 9.5
ΔE6        : +1.7, +3.5, +1.0, +1.4   (1-2σ)
```

**해석**: return_delay 시간성 + stripe spatial 패턴이 인장 특성과 강한 상관. Scan 분기 단독으로 풀-스택의 75 % 이상 회복 → 본 분기 가치 입증.

### 5.3 시나리오 Q (B1.4 / B1.2 강세)

```
B1.2, B1.4 RMSE_E6 / RMSE_E0 ≈ 1.1 (각각 풀-스택과 거의 동등)
B1.1, B1.3, B1.5 RMSE_E6 / RMSE_E0 ≈ 1.5 (다른 빌드 빈약)
```

**해석**: Scan 분기는 melt mode 결함 빌드 (B1.2 Keyhole/LOF, B1.4 스패터) 에서만 강한 신호. 풀-스택에서 Scan 의 역할은 해당 빌드 한정.

---

## 6. 후속 실험 분기

| 본 결과 | 다음 실험 |
|:--|:--|
| 시나리오 O (Scan 빈약) | E5 vs E6 비교로 CAD/Scan 우열 확인. spatial CNN 무용 검증을 위해 E6b (patch 평균) 후속 |
| 시나리오 P (Scan 시간성 가치) | Scan + 1 개 페어 (Scan+sensor, Scan+dscnn) 로 결합 가치 측정 |
| 시나리오 Q (B1.2/B1.4 강세) | return_delay saturation hp sweep (0.5 / 0.75 / 1.0 s) 로 시그널 압축 한계 확인 |

---

## 7. 연관 문서

- 공통 설정: [PLAN.md](./PLAN.md)
- 시리즈 동료 실험: [E3 only_v0_img](./PLAN_E3_only_v0_img.md), [E4 only_dscnn](./PLAN_E4_only_dscnn.md), [E5 only_cad](./PLAN_E5_only_cad.md), [E7 only_sensor](./PLAN_E7_only_sensor.md)
- Base 풀-스택: [Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md)
- Scan patch 캐시 빌더: [Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache_scan_patch.py](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/cache_scan_patch.py)
- 논문 정의 (return_delay sat=0.75): commit 98fdac1 참조
