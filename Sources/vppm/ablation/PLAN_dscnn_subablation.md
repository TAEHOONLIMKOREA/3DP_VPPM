# E5–E12 + E23/E24: DSCNN 서브 채널 Ablation 실험 계획

> **목적**: E1 (no-dscnn, ΔUTS +7.46 / ΔUE +2.40 / ΔTE +2.67) 의 DSCNN 8 채널 중 **어느 결함
> 클래스가 실제 기여자**인지 규명한다. 특히 **연신율 (UE / TE) 을 지배하는 채널** 식별이 핵심.
>
> **배경**: [E1 결과](./PLAN_E1_no_dscnn.md) 에서 DSCNN 전체 제거 시 UE 가 naive 수준으로 붕괴 (15.19 vs
> 15.00). 즉 **DSCNN 8 채널이 연신율 정보의 거의 전부**를 담고 있다. 다만 어느 채널이 본질이고
> 어느 채널이 redundant 인지는 미규명.
>
> **가설**: (1) `seg_recoater_streaking` (B1.5 리코터 손상 빌드의 핵심) 과 (2) `seg_super_elevation`
> / `seg_excessive_melting` (keyhole · LOF 결함) 이 상위 기여자. "정상" 클래스(`seg_powder`,
> `seg_printed`) 는 inverse 지표라 상대적으로 기여 작음.

---

## 1. DSCNN 채널 정리

[features.py:24-31](../origin/features.py#L24-L31) 및 [config.py:53-63](../common/config.py#L53-L63) 기준
(0-based feature idx, HDF5 class id 는 원본 12 클래스 중 선정된 8 개):

| Feature idx | 이름 | HDF5 class id | 카테고리 | 물리적 의미 | 예상 기여 |
|:-----------:|-----|:-------------:|:--------:|:-----------|:---------:|
| 3  | `seg_powder`            | 0  | Normal  | 미용융 분말 영역 비율 (LOF 결함 inverse proxy) | 중 |
| 4  | `seg_printed`           | 1  | Normal  | 정상 프린트 영역 비율 (전체 품질 proxy)       | 하 |
| 5  | `seg_recoater_streaking`| 3  | Defect  | 리코터 블레이드 줄무늬 — **B1.5 리코터 손상 빌드 핵심** | **최상** |
| 6  | `seg_edge_swelling`     | 5  | Defect  | 스캔 엣지 팽창 (과열) | 중 |
| 7  | `seg_debris`            | 6  | Defect  | 스패터·잔해물 (B1.4 관련) | **상** |
| 8  | `seg_super_elevation`   | 7  | Defect  | 과도 돌출 (local over-melt) | **상** |
| 9  | `seg_soot`              | 8  | Defect  | 매연/그을음 (산화물 응축) | 중 |
| 10 | `seg_excessive_melting` | 10 | Defect  | 과다 용융 — **keyhole 결함 (B1.2 Keyhole 조건) 핵심** | **최상** |

카테고리 별 분류:
- **Normal** (idx 3, 4): 결함 없음 신호
- **Defect 6 클래스** (idx 5~10): 이상 패턴
  - 기계/표면 결함: recoater_streaking, edge_swelling, debris
  - 열 결함: super_elevation, soot, excessive_melting

---

## 2. 실험 설계

### 2.1 실험 목록 (8 개 서브 + 2 개 묶음)

| ID | 제거 채널 | 사용 피처 수 | 의미 |
|:--:|----------|:-----------:|------|
| E5  | `seg_powder` (3)            | 20 | 미용융 분말 비율 단독 기여 |
| E6  | `seg_printed` (4)           | 20 | 정상 영역 비율 단독 기여 |
| E7  | `seg_recoater_streaking` (5)| 20 | **리코터 결함 — B1.5 핵심 후보** |
| E8  | `seg_edge_swelling` (6)     | 20 | 엣지 팽창 단독 |
| E9  | `seg_debris` (7)            | 20 | 잔해물 — B1.4 스패터 빌드 관련 |
| E10 | `seg_super_elevation` (8)   | 20 | 과도 돌출 (local over-melt) |
| E11 | `seg_soot` (9)              | 20 | 매연 단독 |
| E12 | `seg_excessive_melting` (10)| 20 | **과다 용융 — B1.2 Keyhole 핵심 후보** |
| E23 | `dscnn_defects_all` (5,6,7,8,9,10) | 15 | **결함 6 채널 묶음 제거 — Normal 2개만 남김** |
| E24 | `dscnn_normal` (3,4)        | 19 | **Normal 2 채널 묶음 제거 — Defect 6개만 남김** |

### 2.2 설계 논리 (2단계)

**1단계 — 개별 채널 기여도 (E5–E12)**

각 클래스를 단독 제거해 그 클래스 하나가 사라졌을 때의 ΔRMSE 를 측정. 센서 서브 실험에서 관찰된
"단독 채널은 모두 Marginal" 현상이 DSCNN 에서도 나타날 수 있음 — 그 경우 2단계 필요.

**2단계 — 카테고리 묶음 (E23, E24)**

- **E23 (defects_all)**: 결함 6 개를 통째로 제거. Normal 2 개만으로 연신율·강도를 예측할 수 있는지 테스트.
  - 예상: UE 가 naive 에 근접 (연신율 정보 대부분이 defects 에 있다는 가설 검증).
- **E24 (normal_only)**: Normal 2 개 (`seg_powder`, `seg_printed`) 를 제거. Defect 6 개만 남김.
  - 예상: 거의 영향 없음 — Normal 클래스는 "defect 없는 영역" 의 보완 확률로, redundant 정보일 가능성.

이 2 단계를 조합하면 "결함 클래스 집단이 연신율 지배" vs "특정 단일 결함 클래스가 지배" 판별 가능.

### 2.3 해석 기준

| 판정 | 기준 (UTS 기준, ΔE1 = +7.46) | 기준 (UE 기준, ΔE1 = +2.40) |
|:----:|:---------------------------|:---------------------------|
| **Critical**    | ΔRMSE ≥ 3.73 (50% × ΔE1) | ΔRMSE ≥ 1.20 |
| **Contributing**| 1.49 ≤ ΔRMSE < 3.73       | 0.48 ≤ ΔRMSE < 1.20 |
| **Marginal**    | ΔRMSE < 1.49              | ΔRMSE < 0.48 |

> **UE 기준을 주 판정** 으로 사용한다 — DSCNN 의 핵심 기여는 연신율에 있기 때문 (§E1 결과 참조).
> UTS 기준은 보조.

### 2.4 빌드별 특이성 검증

per-build 분해로 아래 예측이 맞는지 확인:

| 빌드 | 핵심 예상 채널 | 이유 |
|:----:|:-------------:|:----|
| B1.2 | `seg_excessive_melting` (idx 10) | Keyhole 파라미터 조건 — 과다 용융 결함 |
| B1.2 | `seg_powder` (idx 3)            | LOF 파라미터 조건 — 미용융 분말 결함 |
| B1.4 | `seg_debris` (idx 7)            | 스패터 빌드 — 잔해물 과다 |
| B1.5 | `seg_recoater_streaking` (idx 5)| 리코터 손상 → 줄무늬 결함 |
| B1.3 | (모든 결함 낮음 예상)            | 오버행은 CAD 지배 |

해당 예상이 맞으면 "**결함 클래스가 빌드별 특성을 직접 반영** 한다" 는 강력한 해석 근거.

---

## 3. 구현

### 3.1 config.py 에 서브 그룹 추가

```python
# Sources/vppm/common/config.py — FEATURE_GROUPS 아래
FEATURE_GROUPS_DSCNN_SUB = {
    # 1단계: 개별 채널
    "dscnn_powder":             [3],
    "dscnn_printed":            [4],
    "dscnn_recoater_streaking": [5],
    "dscnn_edge_swelling":      [6],
    "dscnn_debris":             [7],
    "dscnn_super_elevation":    [8],
    "dscnn_soot":               [9],
    "dscnn_excessive_melting":  [10],
    # 2단계: 카테고리 묶음
    "dscnn_defects_all":        [5, 6, 7, 8, 9, 10],  # E23
    "dscnn_normal":             [3, 4],                # E24
}
FEATURE_GROUPS.update(FEATURE_GROUPS_DSCNN_SUB)
```

### 3.2 run.py 확장

```python
# Sources/vppm/ablation/run.py — EXPERIMENTS 에 추가
EXPERIMENTS.update({
    "E5":  ("dscnn_powder",             "No-Powder — 미용융 분말 제거"),
    "E6":  ("dscnn_printed",            "No-Printed — 정상 영역 제거"),
    "E7":  ("dscnn_recoater_streaking", "No-Streaking — 리코터 줄무늬 제거 (B1.5 핵심 후보)"),
    "E8":  ("dscnn_edge_swelling",      "No-EdgeSwelling — 엣지 팽창 제거"),
    "E9":  ("dscnn_debris",             "No-Debris — 잔해물 제거 (B1.4 관련)"),
    "E10": ("dscnn_super_elevation",    "No-SuperElevation — 과도 돌출 제거"),
    "E11": ("dscnn_soot",               "No-Soot — 매연 제거"),
    "E12": ("dscnn_excessive_melting",  "No-ExcessiveMelt — 과다 용융 제거 (B1.2 Keyhole)"),
    "E23": ("dscnn_defects_all",        "No-DefectsAll — DSCNN 결함 6채널 제거"),
    "E24": ("dscnn_normal",             "No-DSCNNNormal — DSCNN normal 2채널 제거"),
})
```

### 3.3 일괄 실행

**호스트 순차 (GPU 1대):**

```bash
for E in E5 E6 E7 E8 E9 E10 E11 E12 E23 E24; do
  ./venv/bin/python -m Sources.vppm.ablation.run --experiment $E
done
./venv/bin/python -m Sources.vppm.ablation.run --rebuild-summary
```

**도커 병렬 (GPU 4장 × 3배치):**

권장 구조 — `docker/ablation/dscnn_sub/` 을 `sensor_sub/` 와 동일하게 구성 후:

```bash
cd docker/ablation/dscnn_sub
./run_all.sh      # full (~30~45분 예상)
./run_all.sh --quick
```

배치 스케줄 (10 실험):

| Batch | 병렬 | GPU 배정 |
|:-----:|:-----|:--------|
| 1/3 | E5·E6·E7·E8 | 0·1·2·3 |
| 2/3 | E9·E10·E11·E12 | 0·1·2·3 |
| 3/3 | E23·E24    | 0·1 |

---

## 4. 결과 산출물

```
Sources/pipeline_outputs/ablation/
├── E5_no_dscnn_powder/
├── E6_no_dscnn_printed/
├── E7_no_dscnn_recoater_streaking/
├── E8_no_dscnn_edge_swelling/
├── E9_no_dscnn_debris/
├── E10_no_dscnn_super_elevation/
├── E11_no_dscnn_soot/
├── E12_no_dscnn_excessive_melting/
├── E23_no_dscnn_defects_all/
├── E24_no_dscnn_normal/
└── dscnn_sub_summary.md            # 별도 요약 (아래 §5.1)
```

각 폴더 레이아웃은 기존 E1~E22 와 동일 (`models/` + `results/` + `features/` + `experiment_meta.json`).

### 4.1 별도 summary 템플릿

기존 `summary.md` 는 모든 실험을 섞어 표시하므로, DSCNN 서브 결과는 별도 `dscnn_sub_summary.md` 권장:

```markdown
# DSCNN Sub-Channel Ablation Summary

| Exp | Channel | ΔYS | ΔUTS | ΔUE | ΔTE | UE 판정 | UTS 판정 |
|:---:|---------|:---:|:----:|:---:|:---:|:-------:|:--------:|
| E5  | powder             | ? | ? | ? | ? | M/C/Cr | M/C/Cr |
| E6  | printed            | ? | ? | ? | ? | M/C/Cr | M/C/Cr |
| E7  | recoater_streaking | ? | ? | ? | ? | M/C/Cr | M/C/Cr |
| ...
| E23 | defects_all (6ch)  | ? | ? | ? | ? | — | — |
| E24 | normal (2ch)       | ? | ? | ? | ? | — | — |
| E1  | **전체 8ch (ref)**  | +2.38 | +7.46 | +2.40 | +2.67 | — | — |
```

자동 생성 스크립트: `Sources/vppm/ablation/build_dscnn_sub_summary.py` 를 신규로 만들거나, 기존
`run.py --rebuild-summary` 를 확장해 실험 필터(prefix) 를 받도록 수정.

---

## 5. 추가 관찰 포인트

### 5.1 연신율 지배 확인

DSCNN 전체 (E1) 에서 UE 가 naive 수준으로 붕괴 (15.19 vs 15.00) 한 것이 **어느 채널의 부재로
야기**되었는지가 본 실험의 핵심 질문. 단일 실험에서 UE 가 naive 에 근접(≥ 14.5) 하는 결과가
나오면 그 채널이 "연신율 정보의 독점 원천"이라는 강력한 결론.

### 5.2 Fold std 증가 기여도

E1 에서 YS fold std 가 0.62 → 1.27 (2× 증가). 이 증가가 어느 채널 제거에서 주로 발생하는지
확인. B1.2 가 E2 fold std 의 주 원인이었던 것과 유사한 패턴 기대.

### 5.3 E23 ≈ E1 vs E23 << E1 비교

- E23 (defects 6개 제거) ≈ E1 (8개 제거) → **연신율 정보는 결함 클래스에만 있음**. Normal 2개는 redundant.
- E23 < E1 (= Normal 2 개 제거분까지 더 빠져 더 악화) → **Normal 클래스도 기여 존재**.

둘 중 어느 쪽이냐에 따라 DSCNN 경량화 가능성이 달라진다.

### 5.4 E24 ≈ Baseline 여부

E24 (Normal 2개 제거) 가 E0 와 거의 동일하면 → Normal 클래스는 **전부 불필요**. DSCNN 을
"결함 6 클래스 전용" 으로 축소 가능.

---

## 6. 리소스 및 일정

- **학습 시간**: 10 실험 × 4 속성 × 5 fold × ~1분 = **~3.5시간** (GPU 1대 순차)
  또는 GPU 4장 병렬 시 **~1시간** (3 배치).
- **디스크**: ~200 MB
- **per-build 분석**: 10 실험 × ~30초 = ~5분 (GPU 활용)

---

## 7. 성공 기준

- [x] 10 개 실험 모두 완주 (E5~E12 + E23 + E24)
- [x] 채널별 ΔUE / ΔTE / ΔYS / ΔUTS 표 완성
- [x] 각 채널 Marginal / Contributing / Critical 판정 (UE / UTS 기준 각각)
- [x] E23 vs E1, E24 vs E0 비교로 "defect 클래스 지배" 가설 검증
- [x] per-build 분석 실행 후 빌드 ↔ 결함 클래스 매핑 (§2.4) 검증

---

## 8. 리스크 및 한계

- **센서 서브 실험과 동일한 "모든 단독이 Marginal" 패턴 위험**: 8 채널 간 중복 정보가 많으면 단독 제거는 noise-level 결과만. 이 경우 E23/E24 묶음 결과가 해석의 핵심.
- **DSCNN 자체 노이즈**: DSCNN 모델이 완벽한 결함 분류기가 아님. 특정 클래스 채널이 노이즈로만 작용할 가능성 있음 — 그 경우 제거가 오히려 개선 (센서 E20 ventilator 사례 참고).
- **카테고리 정의의 인위성**: "Normal vs Defect" 구분은 데이터 제공자가 정한 것. 모델은 8 클래스 전체를 동일하게 취급하므로, Normal 클래스도 "defect 부재 신호"로서 기여할 수 있음 — §5.4 로 확인.
- **B1.3 오버행 빌드 편향**: B1.3 은 DSCNN 결함이 적어 E7~E12 단독 제거 영향이 미미할 수 있음. per-build 에서 B1.3 은 기준점으로 제외하고 해석하는 것도 방법.
- **Seed 의존성**: 단독 채널 제거의 ΔRMSE 가 fold std 보다 작을 가능성 큼 → Marginal 판정 케이스는 **seed 2~3개 반복** 권장.

---

## 9. 부분 실행 결과 (2026-04-25)

10 개 계획 실험 중 **E5~E8 (4 개) 만 완료**. 나머지 6 개 (E9~E12, E23, E24) 미실행.

### 9.1 RMSE 표

| 실험 | 채널 | 카테고리 | YS (MPa) | UTS (MPa) | UE (%) | TE (%) |
|:---:|:----|:--------:|:--------:|:---------:|:------:|:------:|
| E0 Baseline | —              | —      | 28.66 ± 0.62 | 60.72 ± 2.59 | 12.79 ± 0.27 | 15.46 ± 0.20 |
| **E5** | seg_powder (3)  | Normal | 29.23 ± 1.03 | **64.76 ± 1.57** | **14.19 ± 0.34** | **16.76 ± 0.23** |
| **E6** | seg_printed (4) | Normal | 29.84 ± 1.31 | **64.96 ± 2.23** | **14.26 ± 0.24** | **16.72 ± 0.25** |
| E7 | seg_recoater_streaking (5) | Defect | 29.16 ± 0.89 | 60.34 ± 2.24 | 13.02 ± 0.20 | 15.76 ± 0.23 |
| E8 | seg_edge_swelling (6)      | Defect | 28.50 ± 0.91 | 60.07 ± 2.98 | 13.10 ± 0.27 | 15.78 ± 0.18 |
| **E1** | **전체 8 채널 (ref)** | —  | **31.04 ± 1.27** | **68.19 ± 2.26** | **15.19 ± 0.20** | **18.13 ± 0.28** |

### 9.2 ΔRMSE 및 판정

| 실험 | ΔYS | ΔUTS | ΔUE | ΔTE | UE 판정 | UTS 판정 |
|:---:|:---:|:----:|:---:|:---:|:-------:|:--------:|
| E5 powder              | +0.57 | +4.03 | +1.40 | +1.31 | **Critical**   | Contributing |
| E6 printed             | +1.18 | +4.24 | +1.47 | +1.27 | **Critical**   | Contributing |
| E7 recoater_streaking  | +0.50 | −0.39 | +0.23 | +0.31 | Marginal       | Marginal (개선) |
| E8 edge_swelling       | −0.17 | −0.65 | +0.31 | +0.32 | Marginal       | Marginal (개선) |

판정 기준 (§2.3):
- UE Critical: ΔUE ≥ 1.20  /  Contributing: 0.48 ≤ ΔUE < 1.20  /  Marginal: < 0.48
- UTS Critical: ΔUTS ≥ 3.73  /  Contributing: 1.49 ≤ ΔUTS < 3.73  /  Marginal: < 1.49

### 9.3 핵심 발견 — **가설 부분 반박**

§1 의 가설은 다음과 같았다:

> "Normal 클래스(`powder`, `printed`) 는 inverse 지표라 상대적으로 기여 작음.
> Defect 클래스 (특히 `recoater_streaking`, `excessive_melting`) 가 상위 기여자."

E5~E8 결과는 **반대 패턴**을 보임:

1. **Normal 2 채널이 Critical** — `seg_powder` 단독 제거로 ΔUE +1.40 (E1 전체 8 채널 효과의 58%).
   `seg_printed` 도 동일 수준 (ΔUE +1.47).
2. **Defect 2 채널은 Marginal** — `seg_recoater_streaking` (B1.5 핵심 후보) 와 `seg_edge_swelling` 모두
   noise-level (UE/UTS 모두 fold std 이내).

### 9.4 잠정 해석

**Normal 채널이 통합 결함 신호 역할**:
- `seg_printed` ≈ 1 − Σ(defects). 정상률 = 1 - 결함률 의 대수적 관계로, **단일 Normal 채널이
  6 결함 채널의 정보를 압축**해 담고 있을 가능성.
- `seg_powder` 는 LOF (lack of fusion) 의 직접 inverse 지표. 미용융 분말 비율은 다공성 → 강도/연성
  예측의 핵심 신호.
- 결과적으로 모델은 "어떤 결함이냐" 가 아닌 "결함이 얼마나 있느냐" 를 학습.

### 9.5 미실행 6 실험의 결정적 역할

다음 실험들이 가설 검증에 필수:

| 실험 | 채널 | 검증 포인트 |
|:---:|:----|:-----------|
| **E12** | seg_excessive_melting | B1.2 Keyhole 빌드 핵심 후보. 단독 Critical 이면 §1 가설 부분 회복 |
| **E23** | dscnn_defects_all (6ch 묶음) | **결정적 실험**. ΔUE 가 E1 (+2.40) 에 근접 → "결함 클래스 집단이 지배" / 작음 → "Normal 채널이 결함 정보 인코딩" |
| **E24** | dscnn_normal (2ch 묶음)      | **결정적 실험**. ΔUE 가 크면 (>1.5) → Normal 지배 가설 확정 / 작으면 → §1 가설 회복 |
| E9  | seg_debris (B1.4 스패터) | per-build 분해로 B1.4 ΔRMSE 최대 확인 |
| E10 | seg_super_elevation       | Defect 단독이 모두 Marginal 인지 추가 검증 |
| E11 | seg_soot                  | 〃 |

E5~E8 의 4σ-수준 비교 및 fold std 값을 보면:
- E5/E6 의 Critical 판정은 ΔUE 1.40~1.47 vs fold std 1.27~1.45 (E1 기준) → **1σ 수준** 으로 강한 유의성 아님.
- 따라서 **seed 반복 실험** 을 함께 수행해 신뢰도 확보 권장.

### 9.6 다음 단계

```bash
cd docker/ablation/dscnn_sub
./run_all.sh    # 잔여 6 실험 (E9~E12, E23, E24) 자동 실행 — 4-GPU 병렬 ~45–60분
```

각 실험 완료 후 빌드별 분해:
```bash
for E in E5 E6 E7 E8 E9 E10 E11 E12 E23 E24; do
  ./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment $E
done
```

---

## 10. 연관 문서

- 상위 그룹 실험: [PLAN_E1_no_dscnn.md](./PLAN_E1_no_dscnn.md) — 전체 8 채널 제거 (ΔUE +2.40, ΔTE +2.67)
- 센서 서브 실험: [PLAN_sensor_subablation.md](./PLAN_sensor_subablation.md) — 유사한 2단계 설계
- 조합 실험: [PLAN_E13_combined.md](./PLAN_E13_combined.md) — DSCNN × Sensor 상호작용
- 공통 설정: [PLAN.md](./PLAN.md)
- 종합 보고서: [FULL_REPORT.md](../../pipeline_outputs/ablation/FULL_REPORT.md) §5 (DSCNN 영향), §13 (후속 로드맵 2순위)
