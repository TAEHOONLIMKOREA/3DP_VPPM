# LSTM-Full-Stack Ablation — 인덱스

풀-스택 LSTM 모델 ([`vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4`](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md)) ablation 시리즈.

## 시리즈 1 — 분기 제거 (특정 분기를 빼고 나머지 유지)

baseline → LSTM 점프(UTS −13) 외 누적 개선폭이 fold std 이내로 정체된 가설을 검증.

| ID | 계획 | 제거 | MLP 입력 |
|:--:|:--|:--|:--:|
| E1 | [PLAN_E1_no_v0.md](./PLAN_E1_no_v0.md) | `branch_v0` | 86 → 70 |
| E2 | [PLAN_E2_no_cameras.md](./PLAN_E2_no_cameras.md) | `branch_v0` + `branch_v1` | 86 → 54 |

## 시리즈 2 — 단일 분기 isolation (한 분기만 남기고 나머지 모두 제거)

5 개 분기 그룹 (img / dscnn / cad / scan / sensor) 각각의 standalone 예측력 측정. **`feat_static` (build_height, laser_module) 도 제거** — MLP 입력 = 해당 분기 임베딩 단독.

| ID | 계획 | 유지 | MLP 입력 |
|:--:|:--|:--|:--:|
| E3 | [PLAN_E3_only_v0_img.md](./PLAN_E3_only_v0_img.md) | `branch_v0` (visible/0, 16-d) | 86 → 16 |
| E4 | [PLAN_E4_only_dscnn.md](./PLAN_E4_only_dscnn.md) | `branch_dscnn` (8-class, 8-d) | 86 → 8 |
| E5 | [PLAN_E5_only_cad.md](./PLAN_E5_only_cad.md) | `branch_cad` (geometry, 8-d) | 86 → 8 |
| E6 | [PLAN_E6_only_scan.md](./PLAN_E6_only_scan.md) | `branch_scan` (laser path, 8-d) | 86 → 8 |
| E7 | [PLAN_E7_only_sensor.md](./PLAN_E7_only_sensor.md) | `branch_sensor` (7-field 1D-CNN, 28-d) | 86 → 28 |

> E3-E7 은 모델에 7-flag 토글 (`use_static/v0/v1/sensor/dscnn/cad/scan`) 추가 필요. 코드 확장 상세는 [PLAN_E3 §3.1](./PLAN_E3_only_v0_img.md#31-코드-변경) 참조.
> v1 (visible/1, 16-d) 단독 isolation 은 본 시리즈에 포함하지 않음 — DSCNN 이 사실상 v1 의 supervised 압축 표현이므로 E4 가 v1 정보의 상한을 측정한다고 간주.

## 공통 기준

- **Base RMSE (E0)**: YS 20.1 / UTS 28.5 / UE 6.5 / TE 8.1.
- **시리즈 1 판정**: ΔRMSE < 1σ → 카메라 노이즈 가설 입증 (시나리오 A).
- **시리즈 2 판정**: 5 개 분기의 standalone 회복도 랭킹 → 본 모델의 핵심 정보원 식별.
- **풀런은 사용자 실행** ([memory feedback](../../../.claude/projects/-home-taehoon-3DP-VPPM/memory/feedback_docker_compose.md): docker compose 우선).
