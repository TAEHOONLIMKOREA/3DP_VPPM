# E34–E36: CAD/좌표 서브 채널 Ablation 실험 계획

> **목적**: E3 (no-cad, v2 ΔUTS **+6.63** / ΔYS +1.85) 의 CAD 3 채널 중 어느 채널이 실제 기여자인지
> 규명. G3 (CAD) 는 단 3 채널인데도 G1 DSCNN 8 채널 (+6.14) 보다 ΔUTS 가 크다 — **채널 효율 1 위**
> 그룹. 어느 단일 좌표 변수가 그 효과를 운반하는지 분해하는 것이 본 실험의 핵심 질문.
>
> **(원래) 가설**: `build_height` (z 누적 높이) 가 누적 열 이력의 결정 변수라 가장 강한 단일 기여자.
> `distance_from_overhang` 은 응고 비등방성·미세조직 prior. `distance_from_edge` 는 빌드 평판 열흡수 prior.

---

## 1. CAD 채널 정리

[features.py:25-28](../baseline/features.py#L25-L28) 및 [config.py CAD 파라미터](../common/config.py#L120-L123) 기준:

| Feature idx | 이름 | HDF5 / 산출 경로 | 물리적 의미 | 예상 기여 |
|:-----------:|-----|:----------------|:-----------|:---------:|
| 0 | `distance_from_edge`     | `slices/part_ids` 거리변환, sat=3 mm | 파트 단면의 외곽까지 최단 거리 (mm) — **빌드 평판 / 자유표면 열흡수 prior** | 중 |
| 1 | `distance_from_overhang` | 레이어간 part 겹침 비교, sat=71 layers (3.55 mm 상당) | 수직 하방 자유표면(오버행) 까지 거리 — **응고 비등방성 / 미세조직 prior** | **상** |
| 2 | `build_height`           | 슈퍼복셀 z 중심 (mm) | 절대 빌드 높이 — **누적 열 이력 / 빌드 시간 proxy** | **최상** |

> 셋 다 [-1, 1] 정규화 후 사용. `distance_from_edge`/`distance_from_overhang` 은 saturation 처리되어
> 큰 값에서 plateau, `build_height` 는 0 ~ 빌드 최대 높이 범위.

---

## 2. 실험 설계

### 2.1 실험 목록 (3 개 단독)

| ID | 제거 채널 | 사용 피처 수 | 의미 |
|:--:|----------|:-----------:|------|
| E34 | `distance_from_edge` (0)     | 20 | 평판/자유표면 열흡수 prior 단독 기여 |
| E35 | `distance_from_overhang` (1) | 20 | **오버행/응고 비등방성 단독** — B1.3 (오버행 빌드) 핵심 후보 |
| E36 | `build_height` (2)           | 20 | **z 누적 높이 단독** — 누적 열 이력 결정 변수 후보 |

> 채널 수가 3 개뿐이므로 묶음 (예: E23 같은 "결함 6 채널 묶음") 단계는 불필요. E3 (전체 3 채널)
> 이 이미 그 역할을 함.

### 2.2 해석 기준 (E3 ΔUTS = +6.63 기준)

| 판정 | 기준 (UTS) | 기준 (YS, ΔE3 = +1.85) |
|:----:|:-----------|:-----------------------|
| **Critical**    | ΔRMSE ≥ 3.32 (50% × ΔE3) | ΔRMSE ≥ 0.93 |
| **Contributing**| 1.33 ≤ ΔRMSE < 3.32       | 0.37 ≤ ΔRMSE < 0.93 |
| **Marginal**    | ΔRMSE < 1.33              | ΔRMSE < 0.37 |

> 단독 채널 fold std (~1.5–2.5 MPa for UTS) 가 Marginal 경계와 비슷 — Marginal 판정 케이스는
> seed 반복으로 통계적 유의성 별도 확인 권장.

### 2.3 가설별 분해 시나리오

| 시나리오 | 패턴 | 해석 |
|:--------|:-----|:-----|
| A: build_height 지배 | E36 ≈ E3, E34/E35 ≈ Marginal | 누적 열 이력이 압도. 다른 두 좌표는 redundant. |
| B: 분산 (additive) | E34 + E35 + E36 ≈ E3 (각 ~+2.2) | 3 채널 각자 독립 기여 — collective 가 아닌 channel-decomposable |
| C: overhang 지배 | E35 >> 나머지 | B1.3 (오버행) 빌드가 평균 RMSE 를 끌어올림 |
| D: 모두 Marginal | 단독 모두 |Δ| < 1.33, but E3 +6.63 | DSCNN 처럼 **collective code** — 단독으로는 분해 불가 |

→ 결과 §8 (실행 후 작성) 에서 위 4 가지 중 어느 패턴인지 판별.

### 2.4 빌드별 특이성 검증

per-build 분해 후 다음 매핑 예측이 맞는지 확인:

| 빌드 | 핵심 예상 채널 | 이유 |
|:----:|:--------------|:----|
| B1.3 | `distance_from_overhang` (1) | 오버행 구조 빌드 — 자유표면까지 거리가 미세조직 직결 |
| B1.5 | `build_height` (2)           | 리코터 손상이 누적 → 후반 레이어일수록 영향 증폭 |
| 모든 빌드 | `distance_from_edge` (0) | 단면 외곽 효과는 빌드 무관 — 보편 prior |

---

## 3. 구현

### 3.1 config.py 에 서브 그룹 추가 (이미 반영)

```python
# Sources/vppm/common/config.py — FEATURE_GROUPS_SCAN_SUB 다음
FEATURE_GROUPS_CAD_SUB = {
    "cad_distance_edge":     [0],   # E34
    "cad_distance_overhang": [1],   # E35
    "cad_build_height":      [2],   # E36
}
FEATURE_GROUPS.update(FEATURE_GROUPS_CAD_SUB)
```

### 3.2 run.py 확장 (이미 반영)

```python
# Sources/vppm/baseline_ablation_with_lstm/run.py — EXPERIMENTS dict 끝
"E34": ("cad_distance_edge",     "No-DistEdge — distance_from_edge 단독 제거"),
"E35": ("cad_distance_overhang", "No-DistOverhang — distance_from_overhang 단독 제거"),
"E36": ("cad_build_height",      "No-BuildHeight — build_height 단독 제거"),
```

### 3.3 도커 실행 (3-GPU 병렬)

```bash
cd docker/ablation
docker compose --profile cad_sub up -d --build      # E34/E35/E36 GPU 0/1/2 병렬
docker compose logs -f                              # 실시간 로그
docker compose down                                 # 종료/정리

# 단일 실험 / smoke test
ABLATION_EXTRA=--quick docker compose --profile E34 up
```

배치 스케줄: 3 실험 × GPU 0/1/2 동시 실행 (batching 없음 — scan_sub 와 동일 패턴).

---

## 4. 결과 산출물

```
Sources/pipeline_outputs/experiments/baseline_ablation_with_lstm/
├── E34_no_cad_distance_edge/
├── E35_no_cad_distance_overhang/
├── E36_no_cad_build_height/
└── (summary.md 자동 갱신 — `docker compose --profile summary up`)
```

각 폴더 레이아웃은 기존 ablation 실험과 동일 (`models/` + `results/` + `features/` + `experiment_meta.json`).

### 4.1 summary 표 템플릿

```markdown
| Exp | Channel | ΔYS | ΔUTS | ΔUE | ΔTE | UTS 판정 |
|:---:|---------|:---:|:----:|:---:|:---:|:--------:|
| E34 | distance_from_edge     | ?   | ?    | ?   | ?   | ?        |
| E35 | distance_from_overhang | ?   | ?    | ?   | ?   | ?        |
| E36 | build_height           | ?   | ?    | ?   | ?   | ?        |
| E3  | **전체 3ch (ref)**     | +1.85 | +6.63 | (TBD) | (TBD) | Critical |
```

---

## 5. 리소스 및 일정

- **학습 시간**: 3 실험 × 4 속성 × 5 fold × ~1 분 = **~60 분** (GPU 1 대 순차) 또는 GPU 3 장 병렬 시
  **~20 분**.
- **디스크**: ~60 MB
- **per-build 분석** (선택): 3 실험 × ~30 초 = ~2 분.

---

## 6. 성공 기준

- [ ] E34/E35/E36 모두 완주 (4 속성 × 5 fold = 60 모델)
- [ ] 채널별 ΔYS / ΔUTS / ΔUE / ΔTE 표 완성
- [ ] 각 채널 Marginal / Contributing / Critical 판정
- [ ] §2.3 4 가지 시나리오 (A–D) 중 실제 패턴 식별
- [ ] (선택) per-build 분해로 §2.4 의 B1.3 / B1.5 가설 검증

---

## 7. 리스크 및 한계

- **채널 수가 3 개로 적음 → 단독 효과가 큼**: DSCNN/Sensor 서브와 달리 redundancy 가 약할 가능성 높음.
  단독 ΔRMSE 가 Critical 판정에 도달할 가능성 충분.
- **Saturation 처리의 비선형성**: `distance_from_edge` (3 mm cap) / `distance_from_overhang`
  (71 layer cap) 은 큰 값에서 정보가 잘림. 실제 구조에 비해 ablation 효과가 과소평가될 수 있음.
- **build_height 의 데이터셋 편향**: 5 빌드의 z 분포가 균일하지 않으면 (특정 빌드가 더 높이 쌓음 등)
  ablation 결과가 일부 빌드 RMSE 변화에 의해 좌우될 수 있음 — per-build 분해 권장.
- **B1.3 (오버행) 의존**: E35 (`distance_from_overhang` 제거) 의 효과는 B1.3 에 집중될 가능성 큼.
  전체 평균 ΔRMSE 가 작더라도 B1.3 에서만 큰 영향이면 그 채널은 "B1.3 특화 정보원" 으로 해석.
- **재현성**: seed 고정 안 함 — Marginal 판정 시 ±1 MPa 수준 fold std 변동 가능. 필요 시 seed 2~3 회 반복.

---

## 8. v2 실행 결과

(실행 후 작성 — RMSE 표, ΔRMSE 표, 시나리오 판별, per-build 분해, 후속 권장)

---

## 9. 연관 문서

- 상위 그룹 실험: [PLAN_E3_no_cad.md](./PLAN_E3_no_cad.md) — 전체 3 채널 제거 (ΔUTS +6.63, ΔYS +1.85)
- 다른 서브 실험: [PLAN_dscnn_subablation.md](./PLAN_dscnn_subablation.md) /
  [PLAN_sensor_subablation.md](./PLAN_sensor_subablation.md) — 동일 2단계 설계 패턴 참조
- 공통 설정: [PLAN.md](./PLAN.md)
- 도커 실행: [docker/ablation/README.md](../../../docker/ablation/README.md) — `--profile cad_sub`
