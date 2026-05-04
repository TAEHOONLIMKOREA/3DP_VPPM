# SV Grid Stride 기반 데이터 증강 (sv_stride_aug) 실험 계획

> **한 줄 요약**: baseline VPPM 의 **non-overlapping** 슈퍼복셀 격자 (xy stride = 8 px ≈ 1.064 mm, z stride = 70 layer = 3.5 mm) 를 **셀 크기는 그대로 유지하면서 stride 만 줄인 sliding-window 격자** 로 바꿔, 같은 sample 라벨에 더 촘촘하게 SV 를 매핑하는 데이터 증강 실험. xy stride 4 px + z stride 35 layer 로 가면 SV 수가 ~8× 늘고 baseline 의 36,047 → ~290K 로 증가. K-fold 가 sample-wise 라 누출 위험 없음 — augmentation 효과 vs over-correlation 위험을 정량 검증.

- **실험 이름**: `sv_stride_aug`
- **결과 위치 (예정)**: `Sources/pipeline_outputs/experiments/sv_stride_aug/{baseline_1x, xy4_z70_4x, xy8_z35_2x, xy4_z35_8x, xy4_z17_16x}/`
- **1차 비교 대상**: baseline 21-feat MLP (`Sources/pipeline_outputs/experiments/vppm_baseline/`) — 동일 모델·동일 학습 hp, 격자만 stride 변경
- **2차 비교 대상 (선택)**: 풀-스택 LSTM (`vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4`) — 캐시 재빌드 비용 검토 후 phase 2/3 로 연기
- **학습 일시 (계획)**: 2026-05-05 ~

---

## 1. 동기 — 왜 stride augmentation 인가

### 1.1 현재 격자의 비효율

`Sources/vppm/common/supervoxel.py::SuperVoxelGrid` 의 격자 정의 (L20-L30):

```python
self.sv_xy_pixels = int(round(self.sv_xy_mm / self.pixel_size_mm))  # = 8 (1.0 mm / 0.133 mm/px)
self.nx = self.image_w // self.sv_xy_pixels                          # = 230
self.ny = self.image_h // self.sv_xy_pixels                          # = 230
self.nz = num_layers // self.sv_z_layers                              # = layers / 70
```

`get_pixel_range(ix, iy)` 는 `c0 = ix * sv_xy_pixels` (L48) — **stride 와 cell size 가 동일** (8 px). z 도 동일 (`l0 = iz * sv_z_layers`, L60). 즉 baseline 격자는 SV 가 서로 인접·**non-overlapping** 한 단순 분할.

이 정의는 논문 Scime et al. (Materials 2023, 16, 7293) Section 2.10 의 1.0×1.0×3.5 mm 직육면체 cell 정의를 충실히 따르지만, **SV 하나가 빌드 볼륨의 한 점만 cover** 한다는 점에서 학습 데이터의 산술적 양은 빌드의 물리 부피로 hard-bound 되어 있다.

### 1.2 자연 augmentation — 셀은 유지, stride 만 축소

같은 1.0×1.0×3.5 mm 셀을 **stride < cell** 로 sliding-window 처럼 빌드 볼륨을 훑으면, 같은 sample (같은 SS-J3 게이지 시편) 의 라벨 (YS/UTS/UE/TE) 은 그대로 두고 입력 SV 의 수만 격자수만큼 증가한다. 인접 SV 간 cell overlap 으로 평균 통계는 강하게 correlated 되지만, 각 SV 는 빌드 볼륨의 약간 다른 위치를 cover → 이미지/스캔/CAD 픽셀 입력은 미세하게 다름.

| 변종 | xy stride (px) | z stride (layer) | xy overlap | z overlap | SV multiplier (이론치) | 추정 총 SV |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| baseline | 8 | 70 | 0% | 0% | 1× | 36,047 |
| **A (xy 만)** | 4 | 70 | 50% | 0% | ~4× | ~144K |
| **B (z 만)** | 8 | 35 | 0% | 50% | ~2× | ~72K |
| **C (xy+z 적당)** | 4 | 35 | 50% | 50% | **~8×** | **~290K** |
| **D (공격적)** | 4 | 17 | 50% | ~76% | ~16× | ~580K |

> SV multiplier 의 이론치는 격자 셀 수 비율 (`(image_w/stride_xy)² × (num_layers/stride_z)`) — `find_valid_supervoxels` 의 CAD/sample 교차 필터를 통과하는 비율은 stride 와 무관 (셀 크기 동일) 하므로 같은 비율이 살아남는다고 가정.

### 1.3 가설 — augmentation 의 두 가지 가능 효과

**가설 H1 (긍정)**: SV 다양성 증가 → 정규화 효과 (특히 cell 경계에 걸친 결함이 더 다양한 위치에서 captured) → MAPE/MAE 개선. 이미지 augmentation (flip/rotate) 과 직교적이라 함께 사용 가능.

**가설 H2 (부정)**: 같은 sample 의 SV 가 8× 늘면 **per-sample SV 수 평균 5.7 → ~45**. mini-batch 안 같은 sample 의 다른 SV 가 함께 학습되면 라벨 노이즈가 implicit ensemble 보다 over-correlation 으로 작용 → over-fitting 또는 plateau. 라벨이 같으니 "같은 라벨 45번 반복 학습" 효과.

본 실험은 H1 vs H2 를 빌드별 RMSE 분해 + train↔val gap 모니터링으로 정량 검증한다.

---

## 2. 격자 정의 변경

### 2.1 `SuperVoxelGrid` 클래스 — stride 와 cell size 분리

현재 (`Sources/vppm/common/supervoxel.py` L20-L62):
- `sv_xy_pixels` 한 변수가 cell size **와** stride 동시 의미
- `nx = image_w // sv_xy_pixels`
- `get_pixel_range(ix, iy)`: `c0 = ix * sv_xy_pixels`, `c1 = c0 + sv_xy_pixels`

변경 (제안):

```python
class SuperVoxelGrid:
    def __init__(self,
                 num_layers: int = None,
                 image_shape: tuple = None,
                 xy_stride_px: int = None,    # 신규 — None 이면 cell size 와 동일 (baseline 호환)
                 z_stride_layers: int = None  # 신규
                 ):
        ...
        # cell size — 셀 정의 자체는 baseline 과 같은 1.0×1.0×3.5 mm 유지
        self.cell_xy_pixels = int(round(self.sv_xy_mm / self.pixel_size_mm))   # = 8
        self.cell_z_layers  = self.sv_z_layers                                  # = 70

        # stride — 인자 없으면 cell size 와 동일 (= baseline non-overlapping)
        self.xy_stride_px    = xy_stride_px    if xy_stride_px    is not None else self.cell_xy_pixels
        self.z_stride_layers = z_stride_layers if z_stride_layers is not None else self.cell_z_layers

        # 격자 크기 — stride 기반
        self.nx = max(1, (self.image_w  - self.cell_xy_pixels) // self.xy_stride_px    + 1)
        self.ny = max(1, (self.image_h  - self.cell_xy_pixels) // self.xy_stride_px    + 1)
        self.nz = max(1, (num_layers    - self.cell_z_layers ) // self.z_stride_layers + 1) if num_layers else 0

    def get_pixel_range(self, ix, iy):
        c0 = ix * self.xy_stride_px
        c1 = min(c0 + self.cell_xy_pixels, self.image_w)
        r0 = iy * self.xy_stride_px
        r1 = min(r0 + self.cell_xy_pixels, self.image_h)
        return r0, r1, c0, c1

    def get_layer_range(self, iz):
        l0 = iz * self.z_stride_layers
        l1 = min(l0 + self.cell_z_layers, self.num_layers)
        return l0, l1
```

> 중요: **cell 크기는 변하지 않음** (`cell_xy_pixels=8`, `cell_z_layers=70`). 논문 정의 유지. `iter_xy()` / `summary()` 시그니처도 호환 — `nx`/`ny` 가 stride 기반으로 더 커질 뿐.

### 2.2 신규 config 상수 (`Sources/vppm/common/config.py` 추가)

```python
# ============================================================
# SV Stride Augmentation (Sources/vppm/sv_stride_aug/PLAN.md)
# baseline 격자는 stride = cell. 본 실험은 stride < cell 로 sliding-window 격자.
# ============================================================
# 변종별 설정 (run.py CLI 가 활성 변종 선택)
SV_STRIDE_VARIANTS = {
    "baseline_1x":   {"xy_stride_px": 8, "z_stride_layers": 70},   # stride = cell
    "xy4_z70_4x":    {"xy_stride_px": 4, "z_stride_layers": 70},   # 변종 A
    "xy8_z35_2x":    {"xy_stride_px": 8, "z_stride_layers": 35},   # 변종 B
    "xy4_z35_8x":    {"xy_stride_px": 4, "z_stride_layers": 35},   # 변종 C ★ 1차 타겟
    "xy4_z17_16x":   {"xy_stride_px": 4, "z_stride_layers": 17},   # 변종 D
}

SV_STRIDE_AUG_EXPERIMENT_BASE_DIR = OUTPUT_DIR / "experiments" / "sv_stride_aug"
```

각 변종은 자기 서브디렉토리 (`{base}/{variant_id}/{cache,features,models,results}`) 에 산출.

---

## 3. 격자 변경이 영향 미치는 곳 — 매트릭스

| 파일 / 함수 | 영향 | 필요 작업 |
|:--|:--|:--|
| **`common/supervoxel.py::SuperVoxelGrid`** (L10-L87) | stride 추가 인자 | §2.1 변경. 디폴트 인자가 없으면 baseline 호환. |
| **`common/supervoxel.py::find_valid_supervoxels`** (L90-L161) | cell size 기반 `sv_area` 그대로 OK (`grid.cell_xy_pixels` 로 명명만 변경) | 로직 동일, 변수명만 cell 기반으로 정리 |
| **`baseline/features.py::FeatureExtractor.extract_features`** (L58-L137) | z-block 단위 처리 — **stride < cell 이면 z-block 들이 overlap**. `for iz in range(self.grid.nz)` 가 같은 layer 를 여러 번 read | 메모리/시간 비용 ↑ — 변종 C 면 z-block 2배, layer 단위 read 도 2배 (overlap 50%). 기존 `_prev_cad_layer` carry-over 는 **z-stride 가 cell 보다 작아질 때 의미가 깨짐** (변종 B/C/D 에서 같은 layer 가 여러 z-block 에 걸침). → carry-over 가 아니라 layer-major loop 으로 재구성 (혹은 z-block 처리 시 매번 처음부터 last_overhang 재계산) — §8 함정 |
| **`baseline/features.py` 의 모든 `for layer in range(l0, l1)` 루프** | overlap 시 같은 layer 의 raster 를 중복 처리 | layer-level cache 도입 (한 build 의 모든 layer 를 한 번씩만 처리하고, SV 별로 patch slicing 만 반복) — §6 phase 1 핵심 작업 |
| **`run_pipeline.py::extract_features_for_build`** (L35-L91) | grid 생성에 stride 인자 전달 + 출력 파일명에 variant 접미사 (`B1.{x}_features_xy4_z35.npz`) | CLI 옵션 `--xy-stride`, `--z-stride`, `--variant` 추가 |
| **`run_pipeline.py::merge_all_builds`** | 변종별 `all_features_{variant}.npz` 생성 | 출력 디렉토리만 변종별 |
| **`common/dataset.py::build_dataset`** (L40-L98) | `f_min`/`f_max` 가 stride 격자의 SV 분포로 재계산 — 다른 변종과 다를 수 있음 | 두 모드 지원: (1) 변종 자체 통계로 새 normalization, (2) baseline normalization.json 으로 inference (transfer 비교) — §8 함정 |
| **`common/dataset.py::create_cv_splits`** (L101-L121) | `unique_samples` 는 `sample_ids` 그대로 받음 — sample 수 동일 → fold 분할 불변 (sample-wise 누출 없음 보장) | 그대로 OK, 단 fold 별 SV 수 불균형 모니터링 추가 |
| **`baseline/train.py`** | DataLoader batch sampling — 같은 sample 의 SV 가 한 batch 에 몰릴 가능성 | sample-balanced sampler 옵션 추가 검토 (§8) |
| **LSTM 캐시 (`lstm/crop_stacks.py`, `lstm_dual/crop_stacks_v1.py`, `lstm_dual_img_4_sensor_7/cache_sensor.py`, `lstm_dual_img_4_sensor_7_dscnn_8/cache_dscnn.py`, `lstm_dual_img_16_.../cache_cad_patch.py`, `cache_scan_patch.py`)** | 모두 `find_valid_supervoxels` 결과의 `voxel_indices` 와 `grid.get_pixel_range/get_layer_range` 를 사용 — stride 변경 시 캐시 자체가 다른 SV 정의 | 변종별 캐시 트리 전체 재빌드 필요. 본 PLAN 1차에서는 **MLP only (21-feat)** 로 한정. LSTM 분기는 phase 2 |
| **K-fold sample 누출** | `create_cv_splits` 가 `unique(sample_ids)` 기반 → SV 수가 8× 늘어도 sample 수는 동일 → 누출 없음 보장 | 검증: 각 fold 의 train/val SV 가 sample 단위로 disjoint 인지 unit test |

---

## 4. 실험 디자인 — 변종 비교

### 4.1 1차 (확정) — baseline vs 변종 C

- baseline (1×, 36,047 SV) vs 변종 C (xy4_z35_8x, ~290K SV) 두 점 비교.
- 동일 모델 (`common/model.py::VPPM`, 21→128→1), 동일 hp (`config.py` L84-L95).
- 측정: YS/UTS/UE/TE 4-target × 5-fold MAE / MAPE / R². baseline `MEASUREMENT_ERROR` (config.py L113-L118) 와의 비율.

### 4.2 2차 (조건부) — 변종 A/B/D 추가

변종 C 가 baseline 대비 +2% 이상 (MAE) 개선되면 A/B/D 도 풀어 stride sweep:
- 변종 A (xy 만): xy 방향 augmentation 단독 효과
- 변종 B (z 만): z 방향 augmentation 단독 효과
- 변종 D (16×): 공격적 stride — H2 (over-correlation) 증상이 명확히 나타나는지

실패하면 (변종 C 가 baseline 과 평탄, std 범위 안) A/B/D 풀런 보류. 디스크/시간 절약.

### 4.3 측정 — 빌드별 RMSE 분해

`Sources/pipeline_outputs/experiments/sv_stride_aug/{variant}/results/metrics_summary.json` 에 baseline 형식 그대로:

```json
{
  "yield_strength":          {"rmse_per_fold": [...], "rmse_mean": ..., "mape": ..., "r2": ...},
  "ultimate_tensile_strength": {...},
  "uniform_elongation":      {...},
  "total_elongation":        {...},
  "per_build_rmse":          {"B1.1": ..., "B1.2": ..., ...},
  "n_supervoxels":           ...,
  "n_unique_samples":        ...,
  "stride_xy_px":            4,
  "stride_z_layers":         35
}
```

baseline `experiments/vppm_baseline/results/metrics_summary.json` 와 1:1 비교.

---

## 5. 캐시 의존 그래프 영향

`CLAUDE.md §5` 의 캐시 의존 그래프:

```
features/all_features.npz
   ├─→ vppm_lstm_single/cache/crop_stacks_*.h5
   │     └─→ vppm_lstm_dual/cache/crop_stacks_v1_*.h5
   │           ├─→ vppm_lstm_dual_img_4_sensor_7/cache/sensor_stacks_*.h5
   │           │     └─→ vppm_lstm_dual_img_4_sensor_7_dscnn_8/cache/dscnn_stacks_*.h5
   │           │           └─→ vppm_lstm_dual_img_16_.../cache/{cad,scan}_patch_stacks_*.h5
   │           │                 └─→ lstm_ablation E1-E7
   └─→ baseline / hidden_sweep / baseline_ablation
```

stride 변경은 `find_valid_supervoxels` 의 SV 인덱스 자체를 바꾸므로 **이 트리 전체가 stride 별로 재빌드** 필요. 디스크/시간 비용:

| 캐시 | baseline 디스크 | 변종 C (~8×) 추정 |
|:--|:--|:--|
| `features/all_features.npz` | ~1 MB | ~8 MB (or ~24 MB 무압축) |
| `vppm_lstm_single/cache/crop_stacks_*.h5` (5 빌드) | 187 MB | ~1.5 GB |
| `vppm_lstm_dual/cache/crop_stacks_v1_*.h5` | 108 MB | ~860 MB |
| `vppm_lstm_dual_img_4_sensor_7/cache/sensor_stacks_*.h5` | 35 MB | ~280 MB |
| `vppm_lstm_dual_img_4_sensor_7_dscnn_8/cache/dscnn_stacks_*.h5` | 25 MB | ~200 MB |
| `vppm_lstm_dual_img_16_.../cache/{cad,scan}_patch_stacks_*.h5` | 278 MB | ~2.2 GB |
| **풀-스택 합계** | **~635 MB** | **~5 GB / variant** |

→ 변종 4종 × 풀-스택 = ~20 GB 추가. 본 PLAN **1차는 21-feat MLP 만** stride 격자 적용 (`features/all_features.npz` ~24 MB / variant). LSTM 분기는 phase 2 로 분리 — 변종 C 가 효과 검증된 후에만 진행.

---

## 6. 단계별 실행 계획

| Phase | 작업 | 산출물 | 예상 시간 |
|:--:|:--|:--|:--:|
| **Phase 0** | `common/supervoxel.py` 에 stride 인자 추가 + cell/stride 분리 + unit test (baseline default 호환 확인 + stride=4/35 시 nx/ny/nz 정확) | supervoxel.py 변경, `Sources/tests/test_supervoxel_stride.py` | 2h |
| **Phase 0.5** | `baseline/features.py` 의 `_prev_cad_layer` carry-over 를 z-block-major → layer-major 로 리팩토링 (overlap 안전). layer-level cache (한 빌드 layer 단위로 dist_edge / dist_overhang / dscnn / scan map 한 번만 계산 → SV slicing 만 반복) | features.py 리팩토링, output 동등성 확인 (baseline 격자에서 기존 features 와 bit-equal) | 4-6h |
| **Phase 1** | 변종별 21-feat MLP 학습 — baseline_1x (재계산 검증) + 변종 C (xy4_z35_8x) | `experiments/sv_stride_aug/{baseline_1x, xy4_z35_8x}/{features, models, results}/` | features 추출 빌드당 1-2h × 5 = 5-10h, train ~30 min |
| **Phase 1.5** | 결과 분석 — baseline vs 변종 C MAE / MAPE / R² / 빌드별 RMSE / fold std. H1 vs H2 진단. train↔val gap 모니터링 | `RESULTS.md` (1차) | 2h |
| **Phase 2 (조건부)** | 변종 A/B/D 추가 풀런 (변종 C 효과 있을 시) | `xy4_z70_4x`, `xy8_z35_2x`, `xy4_z17_16x` 서브디렉토리 | 각 5-10h |
| **Phase 3 (선택)** | 풀-스택 LSTM 트리 변종 C 격자로 재빌드 + 학습. 디스크 ~5 GB, 시간 빌드별 캐시 ~1-2h × 6 캐시 = ~10h, 학습 ~5h | `experiments/sv_stride_aug/xy4_z35_8x/lstm_full86/` | ~1-2일 |

---

## 7. 디스크 / 시간 비용 추정

### 7.1 디스크

| 산출물 | baseline | 변종 C (~8×) |
|:--|:--|:--|
| `features/all_features.npz` (290K SV × 21 feats × float32) | ~1 MB | ~24 MB (gzip) |
| 빌드별 `B1.{x}_features.npz` × 5 | ~600 KB × 5 | ~5 MB × 5 |
| 모델 (`models/*.pt`, 4 prop × 5 fold = 20) | ~250 KB total (2.9k param) | 동일 (모델 변화 없음) |
| 결과 (`results/`) | ~2 MB | 빌드별 RMSE 표가 sample 수 동일이라 비슷 |
| **MLP 1차 합계** | **~5 MB** | **~50 MB / variant** |

→ phase 1 (4 변종) 만 200 MB 미만. phase 3 (풀-스택) 가지 않으면 디스크 부담 거의 없음.

### 7.2 시간

| 단계 | baseline | 변종 C |
|:--|:--|:--|
| `find_valid_supervoxels` (5 빌드 합) | ~5 분 | ~40 분 (격자 8× → loop 8×) |
| 21-feat 추출 (5 빌드) | ~30 분 | layer-level cache 도입 시 ~2-3× (z-block overlap 으로 layer 재읽기 회피 후) — ~1.5h. cache 안 도입 시 ~4-5h |
| MLP 학습 (4 prop × 5 fold, GPU) | ~10 분 | SV 수 8× → epoch 당 step 8× → 학습 ~30-40 분 (early stop 빠르게 도달 가능) |
| **Phase 1 합계 (변종 C)** | — | **~2.5 h** (cache 도입 시) |

> Phase 0.5 의 layer-level cache 가 핵심 — 도입 안 하면 변종 D (16×) 는 features 추출만 ~10h.

---

## 8. 위험 / 함정 / 미해결 질문

### 8.1 sample-wise K-fold 안전성 (재확인)

`common/dataset.py::create_cv_splits` (L101-L121) 는 `unique(sample_ids)` 기반 fold 분할. SV 수가 8× 늘어도 `sample_ids` 의 unique 집합은 변하지 않음 (같은 sample 에 더 많은 SV 가 매핑될 뿐) → fold 누출 위험 0. **단** mini-batch 안에서 같은 sample 의 다른 SV 가 함께 학습되면 over-correlation, gradient bias 가능. 검토 필요:

- batch sampler 옵션: 한 batch 에 같은 sample 의 SV 가 ≤ k 개 들어오게 강제 (e.g., k=3)
- shuffle 강도: `DataLoader(shuffle=True)` 의 random permutation 만으로 충분한지 vs sample-aware sampler 필요한지

1차 실험은 **default DataLoader 그대로** 진행하고 결과가 plateau 면 sample-balanced sampler 도입 검토.

### 8.2 정규화 — 새 통계 vs baseline 통계

변종 C 의 SV 분포는 baseline 의 36K SV 와 다름 (특히 cell 경계에 걸친 SV 비중이 늘어 분포 꼬리가 더 길어질 가능성). 두 모드:

1. **자체 통계** (변종별 새 `f_min`/`f_max` 산출 → 새 normalization.json) — 학습 일관성 좋음
2. **baseline 통계 inference** — baseline normalization.json 그대로 적용. 변종 격자가 baseline 과 정렬되는지 transfer 비교 가능

1차는 **(1) 자체 통계** 로 진행. (2) 모드는 RESULTS 분석 단계에서 추가 inference 로 비교.

### 8.3 측정오차 vs SV 다양성 — implicit ensemble 효과

baseline 의 sample 당 평균 SV 수 = 36,047 / 11,756 ≈ **3.07** SV/sample (필터 후 19,313 / ? sample). 변종 C 면 ~24 SV/sample. 라벨이 같으니 같은 라벨이 24번 반복 — 두 가지 효과:

- **A. Implicit ensemble** : 24 SV 의 다양한 입력으로 모델이 같은 라벨을 robust 하게 학습 → over-fitting 감소
- **B. Label noise saturation** : measurement_error (e.g., YS 16.6 MPa) 가 어차피 측정 한계라 24× 학습으로도 그 아래로 안 내려감 — diminishing returns

baseline 시점 YS 가 measurement_error 의 1.26× 까지 내려와 있어 (`baseline/MODEL.md` 참조), augmentation 의 "추가 RMSE 감소 여지" 자체가 이미 작은 구간. UE (3.76×) / TE (2.88×) 가 가장 큰 개선 여지. → **UE/TE 가 가장 augmentation 효과 잘 드러내는 metric** 으로 1차 결론 도출.

### 8.4 z-block carry-over 깨짐 (중요)

`baseline/features.py` 의 `_prev_cad_layer` / `_last_overhang_layer` 는 z-축 layer-major carry-over (L75-L80). 현재는 `for iz in range(grid.nz):` (L89) 안에서 `for layer in range(l0, l1)` (L176) 가 z-block 단위로 layer 를 한 번씩만 통과 — carry-over 정확.

**변종 B/C/D 처럼 z-stride < cell** 이면 같은 layer 가 여러 z-block 에 등장하고, z-block 의 진행 순서가 layer 의 monotonic 순서와 깨짐. `_last_overhang_layer` 가 같은 layer 를 두 번 보면 같은 픽셀에 두 번 update 되거나, 또는 reverse 방향 z-block 진행 시 **미래 layer** 가 carry-over 에 들어가 인과 위반.

**해결**: features 추출을 z-block-major → **layer-major loop** 으로 재구성. layer 0 부터 num_layers-1 까지 한 번 통과하면서 모든 분기 (CAD, DSCNN, scan, sensor) 를 동시에 갱신 + 활성 SV (해당 layer 가 z-block 에 속하는 모든 SV) 들에 patch 누적. carry-over 상태는 layer-major 루프 진입 시 1회 초기화. → **Phase 0.5 의 핵심 작업**.

### 8.5 feature aggregation 의 SV 간 correlation

z-stride 35 면 인접 z-block 은 35 layer 씩 overlap → 같은 sample 의 SV 두 개가 70 layer 중 35 layer (50%) 를 공유 → 21-feat 의 z-평균 통계가 강하게 correlated. 라벨이 같으니 "같은 입력 ≈ 다른 입력" 으로 보여 **MLP 가 plateau 에 빠질 가능성**. xy stride 4 도 4 px overlap (50%) 으로 동일 효과.

→ training loss 가 일정 epoch 후 평탄해지면서 val loss 도 같이 평탄하면 H2 (over-correlation), val loss 가 baseline 보다 낮으면 H1 (augmentation 효과). **train/val loss 곡선** 이 1차 진단 핵심.

### 8.6 batch shuffle — 같은 sample 의 SV 몰림

DataLoader shuffle 후에도 random permutation 으론 같은 sample 의 24 SV 가 한 batch (1000) 에 평균 24/1000 = 2.4% 비율로 들어감 — 기댓값으론 문제 없음. 단 분산이 크면 일부 batch 가 같은 sample 의 SV 5개+ 포함 → gradient bias. 검증: 처음 100 batch 의 sample_id 분포 히스토그램 출력 → 분산 확인.

### 8.7 cell 가장자리 정렬 — `min(c0 + cell, image_w)` 로 우측 가장자리 SV 가 작아짐

baseline 격자 (stride=cell) 도 동일한 가장자리 처리 (`L49`: `c1 = min(c0 + sv_xy_pixels, self.image_w)`). 변종 C 면 right-most stride 위치가 1842 - 8 + 1 - 1 = 1838 까지 — 가장자리 효과 정량적으론 baseline 과 동일.

---

## 9. 참고 문헌 / 관련 작업

- **VPPM 원논문**: Scime et al., *Materials* 2023, 16, 7293. Section 2.10 의 SV 정의는 셀 크기 1.0×1.0×3.5 mm 를 명시하지만 stride=cell 을 못박지는 않음. 본 실험은 **셀 크기는 그대로**, stride 만 줄이는 augmentation 으로 정의 위반 없이 학습 데이터 확장.
- **Sliding-window augmentation**: 의료영상 (3D MRI patch sampling, stride < patch — Isensee et al. nnU-Net 2020) 와 시계열 (HAR 분야 sliding window with overlap — Jordao et al. 2018) 분야의 표준 기법. 본 실험은 같은 아이디어를 LPBF 빌드 볼륨에 적용한 첫 시도.
- **이미지 augmentation 직교성**: 이후 phase 에서 SV crop 의 horizontal flip / 90° rotate 와 함께 적용 가능 (cell 내부의 spatial 변환 vs cell 위치의 변환 — 직교).

---

## 10. 산출물 / 결과 비교 양식

```
Sources/pipeline_outputs/experiments/sv_stride_aug/
├── baseline_1x/
│   ├── features/{B1.{1..5}_features.npz, all_features.npz, normalization.json}
│   ├── models/vppm_{YS,UTS,UE,TE}_fold{0..4}.pt
│   ├── results/{metrics_raw.json, metrics_summary.json, predictions_{YS,UTS,UE,TE}.csv,
│   │            correlation_plots.png, scatter_plot_uts.png}
│   └── experiment_meta.json     (xy_stride_px=8, z_stride_layers=70, n_sv=...)
├── xy4_z35_8x/                   (변종 C — 1차 타겟)
│   └── (동일 구조)
├── xy4_z70_4x/                   (변종 A, phase 2)
├── xy8_z35_2x/                   (변종 B, phase 2)
├── xy4_z17_16x/                  (변종 D, phase 2)
└── COMPARISON.md                 (변종별 metric 비교 표 + RESULTS.md)
```

### 10.1 비교 표 (RESULTS.md 양식)

```markdown
## MAE (sample-min aggregated, denormalized)

| 속성 | baseline_1x | xy4_z35_8x | Δ vs baseline | measurement_error |
|:--:|:--:|:--:|:--:|:--:|
| YS  (MPa) | 21.0 ± 0.5 | xx.x ± y.y | -z.z% | 16.6 |
| UTS (MPa) | 29.5 ± 1.0 | ... | ... | 15.6 |
| UE  (%)   | 6.5 ± 0.3  | ... | ... | 1.73 |
| TE  (%)   | 8.4 ± 0.2  | ... | ... | 2.92 |

## 빌드별 RMSE 분해 (UE)

| 빌드 | baseline_1x | xy4_z35_8x | 의미 |
|:--:|:--:|:--:|:--|
| B1.1 (기준) | ... | ... | augmentation 효과 미미 예상 |
| B1.2 (Keyhole) | ... | ... | 결함 다양성 ↑ |
| B1.3 (오버행) | ... | ... | overhang 영역 SV 수 8× |
| B1.4 (스패터) | ... | ... | 스패터 spatial 위치 다양성 ↑ |
| B1.5 (리코터) | ... | ... | 리코터 손상 captured 횟수 ↑ |

## fold 별 SV 수 분포 (변종 C)

| fold | train SV | val SV | train sample | val sample | val SV/sample |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 0-4 | ... | ... | ... | ... | ~24 (변종 C) vs ~3 (baseline) |
```

### 10.2 진단 추가 산출물

- `train_val_loss_{property}_fold{i}.png` — train/val loss 곡선 비교 (baseline vs 변종 C). H1 / H2 진단 핵심.
- `batch_sample_distribution.png` — 첫 100 batch 의 sample_id 분포 히스토그램.
- `prediction_correlation_{baseline,variant_c}.png` — 같은 sample 의 baseline 예측과 변종 C 예측의 산점도. ρ ≈ 1 이면 augmentation 이 결과를 거의 안 바꾼 것 (H2 의 한 형태), ρ < 0.95 면 다른 SV 분포 학습.

---

## 11. Open Questions (1차 실행 전 합의 필요)

1. **변종 우선순위**: 1차에 변종 C (8×) 단독 vs A/B/C 셋 다 — 시간 여유 따라 결정.
2. **layer-major 리팩토링 범위**: features.py 만 vs LSTM 캐시 빌더 6종 모두 — 1차는 features.py 만, 캐시 빌더는 phase 3 진입 시 동시 리팩토링.
3. **batch sampler 변경 시점**: 1차에 default 로 진행 후 H2 증상 나오면 2차에 sample-balanced sampler 도입. 사전에 도입할 것인지.
4. **validation set 구성**: stride 격자의 SV 가 fold-internal 로 8× 늘면 validation MAE 의 std 가 작아져 통계적 유의성 판단이 쉬워지지만, val SV 끼리도 correlated → 신뢰구간이 sample 수 기반 (`sqrt(n_sample)`) 인지 SV 수 기반 (`sqrt(n_sv)`) 인지 명시. 본 PLAN 은 **sample-min 집계 후 sample 수 기반** RMSE 로 통일 (baseline 평가와 동일).
5. **풀-스택 phase 3 진입 기준**: 변종 C MLP 결과가 baseline 대비 어느 정도 개선되어야 LSTM 트리 재빌드 비용 (~5 GB, ~1-2일) 을 정당화할 것인가. 제안: UE/TE MAE 에서 −5% 이상 개선 + B1.3/B1.4 빌드별 RMSE 에서 의미 있는 변화.

---

## 12. 참조

- baseline 격자 정의: [`Sources/vppm/common/supervoxel.py`](../common/supervoxel.py) L10-L87
- SV 유효성 필터: [`Sources/vppm/common/supervoxel.py::find_valid_supervoxels`](../common/supervoxel.py) L90-L161
- 21-feat 추출 + z-block carry-over: [`Sources/vppm/baseline/features.py`](../baseline/features.py) L58-L308
- 정규화 + sample-wise K-fold: [`Sources/vppm/common/dataset.py`](../common/dataset.py) L40-L121
- baseline pipeline 진입점: [`Sources/vppm/run_pipeline.py`](../run_pipeline.py) L35-L209
- 캐시 의존 그래프: [`CLAUDE.md` §5](../../CLAUDE.md)
- 결과 표 양식 표준: [`Sources/vppm/baseline/MODEL.md`](../baseline/MODEL.md) §6
- 풀-스택 PLAN (phase 3 의존): [`Sources/vppm/lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md`](../lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4/PLAN.md)
