# E3: Only-v0-Img (visible/0 카메라 분기 단독) 실험 계획

> **공통 설정** (가설, CV, 모델, 해석 기준) 은 [PLAN.md](./PLAN.md) 참조.
> 본 문서는 E3 실험 고유 정보만 기술. E3 ~ E7 은 **단일 분기 isolation** 시리즈로,
> 풀-스택의 5 개 그룹 (img / dscnn / cad / scan / sensor) 중 한 개만 남기고 나머지는 모두 제거한다.

---

## 1. 실험 정의

| 항목 | 값 |
|:--|:--|
| **실험 ID** | E3 |
| **실험명** | Only-v0-Img (visible/0 단독 잔존, `feat_static` 포함 다른 모든 입력 제거) |
| **유지 분기** | `branch_v0` |
| **제거 분기** | `feat_static`, `branch_v1`, `branch_sensor`, `branch_dscnn`, `branch_cad`, `branch_scan` |
| **MLP 입력 차원** | 86 → **16** (= 0 + **16** + 0 + 0 + 0 + 0 + 0) |
| **출력 디렉터리** | `Sources/pipeline_outputs/experiments/lstm_ablation/E3_only_v0_img/` |
| **flag** | `use_static=False`, `use_v0=True`, `use_v1=False`, `use_sensor=False`, `use_dscnn=False`, `use_cad=False`, `use_scan=False` |

### 1.1 유지되는 분기의 물리적 의미

`visible/0` = **용융 직후** 카메라 이미지 (HDF5 경로: `slices/camera_data/visible/0`).

- 입력 캐시: `crop_stacks_{B}.h5` ([`Sources/vppm/lstm/crop_stacks.py`](../lstm/crop_stacks.py))
- 형상: (B, T≤70, 8, 8) — SV 8×8 패치 시퀀스
- 인코더: per-frame CNN(in=1) → LSTM(d_hidden=16) → proj(16)

본 실험은 v0 단독으로 인장 특성을 어디까지 예측할 수 있는지 (= v0 분기의 standalone 성능) 측정한다.
물리적 정보원: melt pool intensity, 잔열 분포, 용융 모드 (LOF/Keyhole) 1차 시그널, super-elevation/debris 표층 결함.

> 풀-스택 86-d 모델에서 v0 가 차지하는 비중: 16/86 ≈ 18.6 %.
> 본 isolation 실험은 "v0 단독으로 풀-스택 성능의 몇 % 를 회복하는가" 를 정량화한다.

---

## 2. 가설

> **v0 단독** 시:
> - **ΔRMSE 크게 양수** (1σ 훨씬 초과) → 단독으로는 정보 부족. 단, **vppm_baseline (21-feat MLP) 보다 좋다면** 본 데이터에서 v0 시간성 처리만으로도 baseline 21-feat 정적 처리를 능가한다는 증거
> - **B1.2 (Keyhole/LOF) 빌드에서 상대 성능 가장 좋을 가능성** → v0 가 melt mode 결함 1차 시그널이라면 해당 빌드 RMSE 가 다른 빌드 대비 풀-스택과 가장 가까워야 함
> - **UE/TE 보다 UTS/YS 에서 회복도 높음** → v0 는 결함 종류 (UE/TE) 보다 결함 밀도 (YS/UTS) 정보를 더 많이 담음

### 2.1 정량 기대치

| 속성 | E0 (풀-스택) | vppm_baseline (21-feat) | E3 예상 RMSE | 회복도 (vs E0 / baseline) |
|:--:|:--:|:--:|:--:|:--|
| YS  | 20.1 | 24.3 | 22.0 ~ 23.5 | E0 대비 90 % 손실, baseline 보다 약간 좋음 |
| UTS | 28.5 | 42.9 | 33.0 ~ 38.0 | E0 대비 60 % 손실, baseline 대비 큰 개선 |
| UE  |  6.5 |  9.3 |  7.5 ~  8.5 | E0 대비 60 % 손실 |
| TE  |  8.1 | 11.3 |  9.5 ~ 10.5 | E0 대비 65 % 손실 |

> **기준선**: ΔRMSE > 2σ 가 "정보 부족" 의 정상적 결과. ΔRMSE < 1σ 면 "v0 단독으로 거의 풀-스택 성능 회복" → 시나리오 G (v0 가 핵심).

---

## 3. 구현

### 3.1 코드 변경

E1/E2 의 `VPPM_LSTM_FullStack_Ablation` 에 토글을 5 개 추가 (`use_static` + sensor / dscnn / cad / scan).
모든 E3-E7 공통 변경. **본 변경은 E3 PLAN 1회만 기술**, E4-E7 은 본 절을 참조한다.

```python
# Sources/vppm/lstm_ablation/model.py — 확장 시그니처
class VPPM_LSTM_FullStack_Ablation(nn.Module):
    def __init__(self, *,
                 use_static=True,
                 use_v0=True, use_v1=True,
                 use_sensor=True, use_dscnn=True,
                 use_cad=True, use_scan=True,
                 ...):
        super().__init__()
        self.use_static = use_static
        self.use_v0, self.use_v1 = use_v0, use_v1
        self.use_sensor, self.use_dscnn = use_sensor, use_dscnn
        self.use_cad, self.use_scan = use_cad, use_scan

        # 각 분기를 flag 에 따라 None 으로 비활성화
        self.branch_v0     = _LSTMBranch(...) if use_v0 else None
        self.branch_v1     = _LSTMBranch(...) if use_v1 else None
        self.branch_sensor = _PerFieldConv1DBranch(...) if use_sensor else None
        self.branch_dscnn  = _GroupLSTMBranch(...) if use_dscnn else None
        self.branch_cad    = _LSTMBranch(...) if use_cad else None
        self.branch_scan   = _LSTMBranch(...) if use_scan else None

        n_total = (
            (n_static if use_static else 0)
            + (d_embed_v0 if use_v0 else 0)
            + (d_embed_v1 if use_v1 else 0)
            + (n_sensor_fields * d_per_sensor_field if use_sensor else 0)
            + (d_embed_d if use_dscnn else 0)
            + (d_embed_c if use_cad else 0)
            + (d_embed_sc if use_scan else 0)
        )
        # E3: n_total = 0 + 16 + 0 + 0 + 0 + 0 + 0 = 16

    def forward(self, feats_static, stacks_v0, stacks_v1, sensors,
                dscnn, cad_patch, scan_patch, lengths):
        embeds = []
        if self.use_static:
            embeds.append(feats_static)
        if self.branch_v0 is not None:
            embeds.append(self.branch_v0(stacks_v0, lengths))
        # ... (다른 분기 동일 None-가드)
        x = torch.cat(embeds, dim=1)
        ...
```

> **주의**: `n_total` 이 0 이 되는 조합 (모든 flag False) 은 valid 가 아니며, fc1 in_features ≥ 1 보장.
> DataLoader 는 모든 캐시 로드 유지 (1차 단순화) — 미사용 입력은 forward 에서 단순 무시.

`run.py`:

```python
EXPERIMENTS["E3"] = dict(
    use_static=False,
    use_v0=True,  use_v1=False,
    use_sensor=False, use_dscnn=False,
    use_cad=False, use_scan=False,
    out_subdir="E3_only_v0_img",
    n_total_feats=16,
    kept=["branch_v0"],
)
```

`_save_experiment_meta` 도 7-flag + `kept`/`removed` 양쪽 기록하도록 확장.

### 3.2 dataset / dataloader

E1/E2 와 동일하게 `load_septet_dataset` 그대로 사용. 모든 캐시는 메모리 로드되지만 모델 forward 가 미사용 분기 입력을 무시한다. 메모리 절감 옵션 (사용 캐시만 로드) 은 1차 풀런 후 검토.

### 3.3 학습 hp (E0 / E1 / E2 동일)

| 항목 | 값 |
|:--|:--|
| optimizer | Adam (lr=1e-3, β=(0.9, 0.999), ε=1e-4) |
| batch_size | 256 |
| max_epochs | 5000 |
| early_stop_patience | 50 |
| grad_clip | 1.0 |
| dropout | 0.1 |
| weight_decay | 0 |
| 손실 | L1 (정규화 공간) |
| 평가 | 원본 스케일 RMSE, sample-level predict_min |

> 입력 차원이 매우 작아 (16-d) MLP fc1 (256-h) 의 표현력은 충분. 과적합 위험은 SV-수준 6,373 샘플 vs 모델 파라미터 ~50k 기준 여유 있음 (≈127 SV/param).
> **주의**: `feat_static` 제거로 build_height / laser_module 정보 부재 → 본 실험은 v0 시간성 분기가 build/laser 정보를 implicit 하게 학습할 수 있는지도 함께 측정.

### 3.4 실행 명령

호스트 (smoke + 단일 실행):

```bash
./venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E3 --quick    # smoke
./venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E3            # full run (사용자)
```

도커 (사용자 풀런):

```bash
cd docker/lstm_ablation
docker compose run --rm e3
```

> docker compose 안내는 [memory feedback](../../../.claude/projects/-home-taehoon-3DP-VPPM/memory/feedback_docker_compose.md) 에 따라 풀런 시 우선 사용.

### 3.5 산출물

```
Sources/pipeline_outputs/experiments/lstm_ablation/E3_only_v0_img/
├── experiment_meta.json       # use_v0=True only (use_static=False), n_total_feats=16
├── features/
│   └── normalization.json     # 16-차원 재정규화 (static 통계 제외)
├── models/
│   ├── vppm_lstm_ablation_E3_{YS,UTS,UE,TE}_fold{0-4}.pt    (20개)
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
| Δ < 1σ | v0 단독으로 거의 풀-스택 성능 → **v0 가 핵심 정보원** (시나리오 G, 매우 드묾) |
| 1σ ≤ Δ < 2σ | v0 단독으로 풀-스택의 70-90 % 회복 — v0 가 강한 단독 신호 |
| 2σ ≤ Δ | 정보 부족 (정상). 다른 분기와의 보완이 필요함을 입증 |

### 4.2 2차 판정 — vs vppm_baseline (21-feat 정적 MLP)

| RMSE_E3 vs RMSE_baseline | 판정 |
|:--|:--|
| E3 ≪ baseline | 시간성 처리 (v0 LSTM) 가 정적 21-feat 처리보다 우수 — 본 모델 패러다임의 우월성 입증 |
| E3 ≈ baseline | v0 16-d 임베딩이 baseline 21-feat 만큼의 정보만 제공 — 카메라 시간성에 추가 가치 없음 |
| E3 ≫ baseline | v0 단독으로는 baseline 보다도 부족 — v0 는 **다른 분기와의 결합** 으로만 가치 발현 |

### 4.3 빌드별 분해

predictions_*.csv 에서 B1.1~B1.5 별 RMSE 분해.

- **B1.2 (Keyhole/LOF) 단독으로 다른 빌드보다 RMSE 회복도가 좋음** → v0 melt mode 결함 1차 시그널 가설 입증
- **B1.5 (리코터 손상) 에서는 v0 가 빈약** → v0 는 분말 도포 직후 (= v1) 시그널 부재로 리코터 결함 미반영. 이 경우 B1.5 RMSE 가 평균보다 훨씬 큼

---

## 5. 예상 결과 시나리오

### 5.1 시나리오 F (가장 가능성 높음 — 정보 부족)

```
E0 풀-스택   : YS 20.1, UTS 28.5, UE 6.5, TE 8.1
baseline    : YS 24.3, UTS 42.9, UE 9.3, TE 11.3
E3 only-v0  : YS 23.0, UTS 36.0, UE 8.0, TE 10.0
ΔE3 (vs E0) : +2.9, +7.5, +1.5, +1.9   (모두 ≥ 2σ)
```

**해석**: v0 단독으로는 풀-스택의 60-70 % 정보만 회복. 단 baseline 보다는 좋아 카메라 시간성 처리의 가치는 입증.

### 5.2 시나리오 G (v0 가 핵심 — 가능성 매우 낮음)

```
E3 only-v0  : YS 21.0, UTS 30.5, UE 6.8, TE 8.5
ΔE3         : +0.9, +2.0, +0.3, +0.4   (1-2σ 사이)
```

**해석**: v0 단독으로 풀-스택 90 % 회복 (`feat_static` 없이도). v0 가 다른 모든 분기를 합한 것과 거의 동등 정보 → 모델 단순화 가능 (v0 분기 단독 16-d).
E1 (no-v0) 결과와 정합성 검증 필수: 시나리오 G 면 ΔE1 도 커야 함.

### 5.3 시나리오 H (v0 빈약 — baseline 보다도 못함)

```
E3 only-v0  : YS 25.5, UTS 45.0, UE 9.8, TE 11.8
ΔE3 (vs base): +1.2, +2.1, +0.5, +0.5
```

**해석**: v0 단독은 baseline 보다도 못함. v0 는 **다른 분기와의 결합** (cross-branch interaction) 으로만 가치 발현. 단독 임베딩으로 인장 특성을 설명할 수 없음.

---

## 6. 후속 실험 분기

| 본 결과 | 다음 실험 |
|:--|:--|
| 시나리오 F (정상) | E4-E7 모두 풀런 후 5-branch standalone 랭킹 표 작성 |
| 시나리오 G (v0 핵심) | E1 (no-v0) 결과 재검토. v0 + 1 개 분기 페어 실험 (E3+sensor, E3+dscnn) 으로 보완 분기 식별 |
| 시나리오 H (baseline 보다 못함) | v0 d_embed sweep (4/8/16/32) 으로 표현력 한계 탐색. 또는 v0 캐시 자체 재검토 (crop 위치, 사이즈) |

---

## 7. 연관 문서

- 공통 설정: [PLAN.md](./PLAN.md)
- 시리즈 동료 실험: [E4 only_dscnn](./PLAN_E4_only_dscnn.md), [E5 only_cad](./PLAN_E5_only_cad.md), [E6 only_scan](./PLAN_E6_only_scan.md), [E7 only_sensor](./PLAN_E7_only_sensor.md)
- 보완 실험: [E1 no_v0](./PLAN_E1_no_v0.md), [E2 no_cameras](./PLAN_E2_no_cameras.md)
- Base 풀-스택: [Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md)
- v0 캐시 빌더: [Sources/vppm/lstm/crop_stacks.py](../lstm/crop_stacks.py)
- baseline 21-feat 결과 (참조선): [Sources/pipeline_outputs/experiments/vppm_baseline/results/metrics_raw.json](../../pipeline_outputs/experiments/vppm_baseline/results/metrics_raw.json)
