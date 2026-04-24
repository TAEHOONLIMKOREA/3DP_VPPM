# Docker — 스캔(G4) 재구현 Ablation (E31~E33)

[PLAN_G4_scan_reengineering.md](../../../Sources/vppm/ablation/PLAN_G4_scan_reengineering.md) 의
스캔 피처 재구현 후 ablation 실험을 **도커 컨테이너에서** 실행한다.

E1~E4, E13~E22 와 동일한 공용 이미지 `vppm-ablation:gpu` 를 공유
([docker/ablation/Dockerfile](../Dockerfile)).

---

## ⚠️ 전제 조건 (매우 중요)

이 도커 스크립트는 **ablation 학습·평가만** 담당한다. 실행 전에 다음 단계들을 호스트에서 완료해야 한다:

### 1. `scan_features.py` 구현

[features.py:115-116](../../../Sources/vppm/origin/features.py#L115-L116) 의 placeholder(=0) 를
실제 알고리즘으로 교체. 구현 스펙은 PLAN §2 와 [implementation_spec.md:186-200](../../../Sources/implementation_spec.md#L186-L200).

필요 함수 (신규 파일 `Sources/vppm/origin/scan_features.py`):
- `build_melt_time_map()` — `scans/{layer}` 를 1842×1842 melt-time 맵으로 래스터화
- `compute_return_delay_map()` — 1mm 커널 max–min 필터 + saturation
- `compute_stripe_boundaries_map()` — 양축 Sobel RMS

### 2. Feature 재추출 (v2)

```bash
# 기존 v1 백업
mv Sources/pipeline_outputs/features/all_features.npz \
   Sources/pipeline_outputs/features/all_features.v1_placeholder.npz

# v2 재추출 (약 6~10시간, 디스크 캐시 ~170 GB)
./venv/bin/python -m Sources.vppm.run_pipeline --phase features
```

### 3. Baseline v2 (E30) 재학습

```bash
# 기존 v1 baseline 백업
mv Sources/pipeline_outputs/results/vppm_origin \
   Sources/pipeline_outputs/results/vppm_origin_v1

# baseline v2 학습·평가 (~30분, GPU)
./venv/bin/python -m Sources.vppm.run_pipeline --phase train,evaluate
```

> E30 은 일반 파이프라인이라 이 docker 스크립트에 포함하지 않는다.

### 4. (이후) 본 ablation 실행 — 이 README 의 아래 절차.

---

## 실험 3종

| ID | drop_group | 제거 피처 idx | 의미 |
|:--:|:----------:|:-------------:|:----|
| E31 | `scan`                    | [18, 19, 20] | v2 No-Scan — G4 3개 전체 제거 (E4 v2 재실험) |
| E32 | `scan_return_delay`       | [19]         | #20 `laser_return_delay` 단독 제거 |
| E33 | `scan_stripe_boundaries`  | [20]         | #21 `laser_stripe_boundaries` 단독 제거 |

---

## 실행 방법

### 단일 실험

```bash
cd docker/ablation/scan_sub
./run.sh E32                   # GPU 0 기본
./run.sh E32 --gpu 1           # GPU 1 지정
./run.sh E32 --gpu 0 --quick   # smoke test
```

### 3개 실험 전체 (3-GPU 병렬, 단일 배치)

```bash
cd docker/ablation/scan_sub
./run_all.sh              # 전체 (~30분)
./run_all.sh --quick      # smoke (~2분)
```

GPU 배정:

| 실험 | GPU |
|:----:|:---:|
| E31 (v2 No-Scan) | 0 |
| E32 (No-ReturnDelay) | 1 |
| E33 (No-StripeBoundary) | 2 |

### 수동 compose

```bash
cd docker/ablation/scan_sub
EXPERIMENT_ID=E32 NVIDIA_VISIBLE_DEVICES=1 docker compose run --rm ablation-scan-sub
```

---

## 볼륨 매핑 (E1~E22 와 동일)

| 호스트 | 컨테이너 | 모드 | 용도 |
|---|---|:---:|---|
| `venv/` | `/workspace/venv` | ro | torch + cuda |
| `Sources/pipeline_outputs/features/` | 동일 | ro | **v2 `all_features.npz`** |
| `Sources/pipeline_outputs/results/` | 동일 | ro | **baseline v2 참조** |
| `Sources/pipeline_outputs/ablation/` | 동일 | rw | **산출물** |

---

## 산출물

```
Sources/pipeline_outputs/ablation/
├── E31_no_scan/                   # E4 결과와는 별도 폴더로 공존
├── E32_no_scan_return_delay/
├── E33_no_scan_stripe_boundaries/
└── summary.md                     # run_all.sh 마지막에 재생성됨
```

> **주의**: `E31_no_scan/` 는 기존 `E4_no_scan/` (v1 placeholder 결과) 와 **별도 폴더**로 보존된다.
> 디렉터리명이 `{exp_id}_no_{drop_group}` 이므로 exp_id 만 다르면 충돌하지 않는다.
> 비교 시 두 폴더의 `metrics_raw.json` 을 함께 읽어 "placeholder vs 실구현" 효과를 판정.

---

## summary.md 흐름

- 각 컨테이너는 `--skip-summary` 로 실행되어 중간 덮어쓰기 방지.
- 모든 컨테이너 완료 후 호스트 `venv` 로 `Sources.vppm.ablation.run --rebuild-summary` 를 호출해
  기존 E1~E4, E13~E22 까지 포함한 **통합 summary.md** 를 재생성.
- 수동 재생성:
  ```bash
  ./venv/bin/python -m Sources.vppm.ablation.run --rebuild-summary
  ```

---

## 결과 해석 템플릿

실행 후 PLAN §3.3 의 비교 테이블을 채운다:

| 버전 | 실험 | ΔUTS vs 해당 baseline | 해석 |
|:---:|:---:|:-------:|:-----|
| v1 (placeholder) | E0 → E4   | −1.04 (기존) | placeholder 포함이 오히려 방해 |
| v2 (실구현)      | E30 → E31 | ?         | 참된 스캔 피처 가치 |
| v2 (실구현)      | E30 → E32 | ?         | return_delay 단독 기여 |
| v2 (실구현)      | E30 → E33 | ?         | stripe_boundaries 단독 기여 |

판정 기준 (PLAN §3.4):
- E30 이 E0 보다 개선 & E31 ΔUTS > 0 → **G4 유의, v2 채택**
- E30 ≈ E0 & E31 ΔUTS ≈ 0        → **G4 본질 불필요, 전체 폐기 고려**
- E30 이 E0 보다 *악화*            → **재구현 코드 버그 의심**, 단위 테스트 재확인

---

## 후속 분석

```bash
# 빌드별 잔차 분해
./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment E31
./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment E32
./venv/bin/python -m Sources.vppm.ablation.analyze_per_build --experiment E33
```

각 실험 폴더에 `per_build_analysis.md` 생성됨.

---

## 트러블슈팅

- **EXPERIMENT_ID 미지정**: compose 에서 `${EXPERIMENT_ID:?...}` 로 즉시 실패. run.sh 사용 권장.
- **`all_features.npz 없음`**: 전제 단계 1~2 미수행. PLAN §4.1 순서대로 진행.
- **피처 20/21 이 여전히 0 인 경우**: scan_features.py 구현 검증 — `np.load(features/all_features.npz)['features'][:, 19:21].std()` 가 0 이면 재추출 실패.
- **Placeholder 제거가 baseline v2 를 악화시킴**: PLAN §7 리스크 — `all_features.v1_placeholder.npz` 와 비교해 어느 빌드에서 성능 역전이 일어났는지 `analyze_per_build.py` 로 세분 분석.
- **디스크 부족**: melt-time 캐시 (~170 GB) 가 `ORNL_Data_Origin/` 과 충돌 가능. 빌드 단위 순차 처리 + 처리 끝난 빌드 캐시 삭제 전략.
