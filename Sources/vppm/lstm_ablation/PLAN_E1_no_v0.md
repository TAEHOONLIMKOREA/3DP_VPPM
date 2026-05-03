# E1: No-v0 (visible/0 카메라 LSTM 분기 제거) 실험 계획

> **공통 설정** (가설, CV, 모델, 해석 기준) 은 [PLAN.md](./PLAN.md) 참조.
> 본 문서는 E1 실험 고유 정보만 기술.

---

## 1. 실험 정의

| 항목 | 값 |
|:--|:--|
| **실험 ID** | E1 |
| **실험명** | No-v0 (visible/0 카메라 분기 제거) |
| **제거 분기** | `branch_v0` (FrameCNN(in=1) + LSTM + proj, d_embed=16) |
| **유지 분기** | `branch_v1`, `branch_sensor`, `branch_dscnn`, `branch_cad`, `branch_scan`, `feat_static` |
| **MLP 입력 차원** | 86 → **70** (= 2 + **0** + 16 + 28 + 8 + 8 + 8) |
| **출력 디렉터리** | `Sources/pipeline_outputs/experiments/lstm_ablation/E1_no_v0/` |
| **flag** | `use_v0=False`, `use_v1=True` |

### 1.1 제거되는 분기의 물리적 의미

`visible/0` = **용융 직후** (laser scan 직후) 카메라 이미지. HDF5 경로: `slices/camera_data/visible/0`.

- 입력 캐시: `crop_stacks_{B}.h5` ([`Sources/vppm/lstm/crop_stacks.py`](../lstm/crop_stacks.py))
- 형상: (B, T≤70, 8, 8) — SV 8×8 패치 시퀀스
- 카메라 분기: per-frame CNN (in=1, 8×8 → 32-d) → LSTM (d_hidden=16) → proj(16)
- 풀-스택에서 차지하는 임베딩 비중: **16 / 86 ≈ 18.6 %**

물리적으로 v0 는 다음 정보를 담을 가능성:
- melt pool intensity / 잔열 분포
- 표면 결함 (super-elevation, debris) 직후 시그널
- LOF/Keyhole 등 용융 모드 지표

> v0 가 시퀀셜로 LSTM 입력될 때 layer-축 변화 (열 축적, 결함 누적) 가 핵심 — 단일 layer 평균이 아닌 **시퀀스 패턴이 정보원**.

---

## 2. 가설

> **v0 단독 제거** 시:
> - **RMSE 변화 < 1σ (fold std)** → v0 분기는 풀-스택의 정체된 누적 개선분 일부만 담당. v1/sensor/dscnn/cad/scan 의 redundancy 로 보완됨 (시나리오 A/B)
> - **B1.2 (Keyhole/LOF) 빌드에서 ΔRMSE 가 가장 큼** → v0 가 melt mode 결함의 1차 시그널이라면 해당 빌드만 악화하고 나머지는 변화 없음
> - **UE/TE 보다 UTS/YS 에서 영향 클 가능성** → 결함 밀도 (UTS 직결) 보다 결함 종류 (UE/TE 직결) 정보가 v0 에 적게 담겼을 것

### 2.1 정량 기대치

| 속성 | E0 (풀-스택) | E1 예상 ΔRMSE | 시나리오별 판정 임계 |
|:--:|:--:|:--:|:--|
| YS  | 20.1 ± 0.9 | +0.0 ~ +0.6 | < 0.9 (1σ) → 시나리오 A |
| UTS | 28.5 ± 0.8 | +0.0 ~ +0.8 | < 0.8 (1σ) → 시나리오 A |
| UE  |  6.5 ± 0.4 | +0.0 ~ +0.3 | < 0.4 (1σ) → 시나리오 A |
| TE  |  8.1 ± 0.4 | +0.0 ~ +0.3 | < 0.4 (1σ) → 시나리오 A |

> 풀-스택까지 누적 개선폭 (vppm_lstm → 풀-스택) 이 YS −0.8, UTS −1.0, UE 0.0, TE −0.3 임을 고려하면 **v0 단독 기여는 그 일부**여야 함. ΔE1 < 누적 개선폭 / 2 정도가 자연스러운 기대.

---

## 3. 구현

### 3.1 코드 변경

**model.py** (신규):

```python
# Sources/vppm/lstm_ablation/model.py
from ..lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.model import (
    FrameCNN, _LSTMBranch, _PerFieldConv1DBranch, _GroupLSTMBranch,
)

class VPPM_LSTM_FullStack_Ablation(nn.Module):
    def __init__(self, *,
                 use_v0: bool = True, use_v1: bool = True,
                 # 나머지 hp 는 풀-스택과 동일 디폴트
                 d_cnn=32, d_hidden_cam=16, d_embed_v0=16, d_embed_v1=16,
                 n_sensor_fields=7, d_per_sensor_field=4,
                 sensor_hidden_ch=16, sensor_kernel=5,
                 n_dscnn_ch=8, d_hidden_d=16, d_embed_d=8,
                 n_cad_ch=2,  d_cnn_c=32,  d_hidden_c=16,  d_embed_c=8,
                 n_scan_ch=2, d_cnn_sc=32, d_hidden_sc=16, d_embed_sc=8,
                 mlp_hidden=(256, 128, 64), dropout=0.1):
        super().__init__()
        self.use_v0, self.use_v1 = use_v0, use_v1

        self.branch_v0 = (_LSTMBranch(in_channels=1, d_cnn=d_cnn,
                                       d_hidden=d_hidden_cam, d_embed=d_embed_v0)
                           if use_v0 else None)
        self.branch_v1 = (_LSTMBranch(in_channels=1, d_cnn=d_cnn,
                                       d_hidden=d_hidden_cam, d_embed=d_embed_v1)
                           if use_v1 else None)
        self.branch_sensor = _PerFieldConv1DBranch(
            n_sensor_fields, d_per_sensor_field, sensor_hidden_ch, sensor_kernel)
        self.branch_dscnn = _GroupLSTMBranch(n_dscnn_ch, d_hidden_d, d_embed_d)
        self.branch_cad = _LSTMBranch(in_channels=n_cad_ch, d_cnn=d_cnn_c,
                                       d_hidden=d_hidden_c, d_embed=d_embed_c)
        self.branch_scan = _LSTMBranch(in_channels=n_scan_ch, d_cnn=d_cnn_sc,
                                        d_hidden=d_hidden_sc, d_embed=d_embed_sc)

        n_total = 2 + (d_embed_v0 if use_v0 else 0) + (d_embed_v1 if use_v1 else 0) \
                  + n_sensor_fields * d_per_sensor_field + d_embed_d \
                  + d_embed_c + d_embed_sc
        # E1 (use_v0=False): n_total = 70
        # E2 (use_v0=False, use_v1=False): n_total = 54

        h1, h2, h3 = mlp_hidden
        self.fc1 = nn.Linear(n_total, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, 1)
        self.dropout = nn.Dropout(dropout)
        self._init_mlp_weights()

    def forward(self, feats_static, stacks_v0, stacks_v1, sensors, dscnn,
                cad_patch, scan_patch, lengths):
        embeds = [feats_static]
        if self.branch_v0 is not None:
            embeds.append(self.branch_v0(stacks_v0, lengths))
        if self.branch_v1 is not None:
            embeds.append(self.branch_v1(stacks_v1, lengths))
        embeds.append(self.branch_sensor(sensors, lengths))
        embeds.append(self.branch_dscnn(dscnn, lengths))
        embeds.append(self.branch_cad(cad_patch, lengths))
        embeds.append(self.branch_scan(scan_patch, lengths))
        x = torch.cat(embeds, dim=1)
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        x = F.relu(self.fc3(x)); x = self.dropout(x)
        return self.fc4(x)
```

**dataset.py** — 풀-스택의 `load_septet_dataset` 그대로 import + dataloader 그대로 재사용. 미사용 분기 입력은 모델 forward 가 무시 (배치별 메모리 약간 낭비되나 단순함).

> 메모리 절감 옵션: `load_septet_dataset(skip_v0=True)` 같은 flag 추가해 stacks_v0 만 None / zeros 로 채워 넘기는 변형. 1차는 단순 처리.

**run.py** — `EXPERIMENTS["E1"] = dict(use_v0=False, use_v1=True, out_subdir="E1_no_v0")`.

### 3.2 학습 hp (E0 와 동일)

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

### 3.3 실행 명령

**호스트 (smoke + 단일 실행)**:

```bash
./venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E1 --quick    # smoke
./venv/bin/python -m Sources.vppm.lstm_ablation.run --experiment E1            # full run (사용자)
```

**도커 (사용자 풀런용)**:

```bash
cd docker/lstm_ablation
docker compose run --rm e1
```

> docker compose 안내는 [memory feedback](../../../.claude/projects/-home-taehoon-3DP-VPPM/memory/feedback_docker_compose.md) 에 따라 풀런 시 우선 사용.

### 3.4 산출물

```
Sources/pipeline_outputs/experiments/lstm_ablation/E1_no_v0/
├── experiment_meta.json       # use_v0=False, use_v1=True, n_total_feats=70, base_model=풀-스택
├── features/
│   └── normalization.json     # 70-차원 (v0 통계 제거) 재정규화
├── models/
│   ├── vppm_lstm_ablation_E1_{YS,UTS,UE,TE}_fold{0-4}.pt    (20개)
│   └── training_log.json      # fold 별 수렴 epoch + val_loss
└── results/
    ├── metrics_raw.json
    ├── metrics_summary.json
    ├── predictions_{YS,UTS,UE,TE}.csv
    ├── correlation_plots.png
    └── scatter_plot_uts.png
```

---

## 4. 판정 기준

### 4.1 1차 판정 (속성별)

| ΔRMSE 범위 | 판정 |
|:--|:--|
| Δ ≤ 0 | v0 가 학습을 방해 (overfitting / 노이즈 분기) |
| 0 < Δ < 1σ | **시나리오 A — v0 분기 제거가 noise 수준** ⭐ 본 가설 검증 |
| 1σ ≤ Δ < 2σ | 경향성 — v0 가 marginal 기여 |
| Δ ≥ 2σ | v0 가 유의 기여 — 시나리오 C (v0 만 핵심) 시사 |

### 4.2 2차 판정 (빌드별 분해)

E1 의 fold-aggregated predictions_*.csv 에서 빌드별 RMSE 분해 → B1.1~B1.5 각각의 ΔRMSE.

- **B1.2 단독 ΔRMSE ≫ 평균** → v0 가 melt mode 결함 (Keyhole/LOF) 의 핵심 시그널
- **모든 빌드 균등하게 작은 변화** → v0 는 redundant (다른 분기로 보완됨)

### 4.3 학습 동학 비교

E0 풀-스택의 fold_epochs (training_log.json) 과 E1 의 fold_epochs 비교:

| E0 평균 epoch | YS 105 | UTS 119 | UE 107 | TE 113 |
|:--:|:--:|:--:|:--:|:--:|

E1 epoch 가 ±20% 이내면 학습 동학 정상. 크게 다르면 (예: ≥2배) ΔRMSE 만으로 판정 어렵고 추가 hyperparameter tune 필요.

---

## 5. 예상 결과 시나리오

### 5.1 시나리오 A (가장 가능성 높음, 본 가설)

```
E0 풀-스택   : YS 20.1, UTS 28.5, UE 6.5, TE 8.1
E1 no-v0    : YS 20.3, UTS 28.7, UE 6.5, TE 8.1
ΔRMSE       : YS +0.2, UTS +0.2, UE 0.0, TE 0.0   (모두 1σ 이내)
```

**해석**: v0 분기는 풀-스택의 정체된 누적 개선분 일부만 담당. 제거해도 다른 분기가 보완 → **카메라 v0 는 본 데이터/모델 구성에서 incremental 가치 없음**.

### 5.2 시나리오 C (반증 — v0 가 핵심)

```
E1 no-v0    : YS 21.5, UTS 30.2, UE 6.9, TE 8.6
ΔRMSE       : YS +1.4, UTS +1.7, UE +0.4, TE +0.5   (1-2σ 사이)
```

**해석**: v0 가 baseline → LSTM 점프의 핵심 출처. 풀-스택의 다른 분기로는 대체 불가능.

---

## 6. 후속 실험 분기

| 본 결과 | 다음 실험 |
|:--|:--|
| ΔRMSE < 1σ (시나리오 A) | E2 (양쪽 카메라 제거) 진행해 카메라 분기 전체 무효성 검증 |
| ΔRMSE > 1σ + B1.2 강함 | v0 d_embed sweep — 압축 한계 탐색 (4/8/16/32) |
| 학습 동학 비정상 | hp tuning — lr/dropout 재조정 후 재시도 |

---

## 7. 연관 문서

- 공통 설정: [PLAN.md](./PLAN.md)
- E2 (양쪽 카메라 제거): [PLAN_E2_no_cameras.md](./PLAN_E2_no_cameras.md)
- Base 풀-스택: [Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md)
- v0 캐시 빌더: [Sources/vppm/lstm/crop_stacks.py](../lstm/crop_stacks.py)
