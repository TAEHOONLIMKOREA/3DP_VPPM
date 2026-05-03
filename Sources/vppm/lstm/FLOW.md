# VPPM-LSTM 코드 동작 흐름

> 이 문서는 `Sources/vppm/lstm/` 의 각 파이썬 파일이 **어떤 순서로 무엇을 입력받아 무엇을 만들어내는지** 따라가며 설명한다.
> 모델/하이퍼파라미터 정의는 [`MODEL.md`](MODEL.md), 설계 결정 근거는 [`PLAN.md`](PLAN.md) 참조.

---

## 0. 사전 조건

LSTM 파이프라인이 돌기 위해선 **baseline 의 21 피처 추출 산출물** 이 먼저 있어야 함:

```
Sources/pipeline_outputs/features/all_features.npz
  ├── features      (N, 21)  float32   ─ 정규화 전 raw
  ├── sample_ids    (N,)     int32
  ├── build_ids     (N,)     int32     ─ 0 = B1.1 ... 4 = B1.5
  └── target_{YS|UTS|UE|TE} (N,) float32
```

생성 명령: `python -m Sources.vppm.run_pipeline --phase features` (LSTM 측에서 다시 돌리지 않음).

---

## 1. CLI 진입점 — `run.py`

```bash
python -m Sources.vppm.lstm.run --all              # cache → train → evaluate
python -m Sources.vppm.lstm.run --phase cache      # L1 만
python -m Sources.vppm.lstm.run --phase train      # L2 (+ 즉시 evaluate)
python -m Sources.vppm.lstm.run --phase evaluate   # L3 만
python -m Sources.vppm.lstm.run --all --quick      # smoke test
```

`run.py::main()` 의 동작 순서:

1. `argparse` 로 `--phase / --all / --builds / --device / --quick` 파싱.
2. `torch.cuda.is_available()` 로 device 자동 결정 (사용자 `--device` 가 우선).
3. `_ensure_dirs()` — `cache/`, `models/`, `results/`, `features/` 디렉터리 생성.
4. `--quick` 시 `config.LSTM_MAX_EPOCHS = 20`, `config.LSTM_EARLY_STOP_PATIENCE = 10` 으로 in-place 덮어쓰기 (smoke test 전용).
5. `_save_experiment_meta()` — 현재 config 스냅샷을 `experiment_meta.json` 으로 저장.
6. 단계 분기:
   - `--all` 또는 `--phase cache` → `run_cache(builds)`
   - `--all` 또는 `--phase train` → `run_train(device)`  *(끝나면 자동으로 evaluate 도 수행)*
   - `--phase evaluate` 단독 → `run_evaluate(device)`

---

## 2. Phase L1 — 크롭 시퀀스 캐시 빌드 (`crop_stacks.py`)

### 2.1 호출 경로

```
run.py::run_cache(builds)
  └─ crop_stacks.py::build_cache(build_ids, out_dir)
       └─ for bid in build_ids:
            └─ _build_one_build(bid, out_dir)
```

이미 `crop_stacks_{bid}.h5` 가 존재하면 해당 빌드는 skip (덮어쓰려면 수동 삭제).

### 2.2 `_build_one_build(build_id, out_dir)` 단계별 처리

```
[1] HDF5 열기
    grid  = SuperVoxelGrid.from_hdf5(hdf5)         # SV 그리드 정보 (xy/z 픽셀-레이어 매핑)
    valid = find_valid_supervoxels(grid, hdf5)     # baseline 과 동일 — voxel_indices, sample_ids, ...
    N     = len(valid["voxel_indices"])

[2] iz 별로 SV 그룹핑 — z-block 단위로 한 번만 카메라/CAD 읽기 (I/O 최적화)
    iz_to_svs : {iz → [sv_i, ...]}

[3] 출력 H5 데이터셋 생성
    out["stacks"]      (N, 70, 8, 8) float16  gzip compression
    out["lengths"]     (N,)          int16
    out["sv_indices"]  (N, 3)        int32   ─ (ix, iy, iz)
    out["sample_ids"]  (N,)          int32

[4] 각 z-block 처리 (tqdm progress)
    for iz in sorted(iz_to_svs):
        l0, l1 = grid.get_layer_range(iz)        # 보통 70 레이어
        block_cam  = cam[l0:l1] / 255.0           # (Tb, 1842, 1842) float16
        block_part = part_ids[l0:l1]              # (Tb, 1842, 1842)

        for sv_i in iz_to_svs[iz]:
            ix, iy, _ = indices[sv_i]
            r0, r1, c0, c1 = grid.get_pixel_range(ix, iy)

            cam_crop  = block_cam [:, r0:r1, c0:c1]    # (Tb, h, w)
            part_crop = block_part[:, r0:r1, c0:c1]    # (Tb, h, w)

            # "유효 레이어" 마스크 — SV xy 안에 part 가 한 픽셀이라도 있는 레이어
            valid_mask = (part_crop > 0).reshape(Tb, -1).any(axis=1)
            if not valid_mask.any():
                valid_mask[Tb // 2] = True             # 안전장치 (이론상 발생 X)

            seq = cam_crop[valid_mask]                  # (T_sv, h, w)

            # 8×8 보다 작으면 zero-pad, 크면 잘라냄 (모서리 SV 방어)
            if h<8 or w<8:    pad → (T_sv, 8, 8)
            elif h>8 or w>8:  seq = seq[:, :8, :8]

            # T_max=70 초과는 클립 (이론상 Tb<=70)
            T_sv_clip = min(T_sv, 70)
            stacks[sv_i, :T_sv_clip] = seq[:T_sv_clip]
            lengths[sv_i] = T_sv_clip

[5] 메타 attrs 저장
    out.attrs[T_max=70, H=8, W=8, camera_channel=0, build_id, n_sv,
              valid_layer_rule="part_ids>0 in SV xy region"]
```

### 2.3 산출물 (빌드별 1 파일, 총 5 파일)

```
Sources/pipeline_outputs/experiments/vppm_lstm/cache/
├── crop_stacks_B1.1.h5
├── crop_stacks_B1.2.h5
├── crop_stacks_B1.3.h5
├── crop_stacks_B1.4.h5
└── crop_stacks_B1.5.h5
```

> **메모리/시간**: z-block 1개 (70 × 1842² × 2B ≈ 470 MB) × 2 (cam + part) ≈ 1 GB 피크. I/O bound 라 빌드당 5–10 분, 5 빌드 1 시간 이내.
>
> **인덱싱 일관성**: SV 순서가 `find_valid_supervoxels()` 의 `voxel_indices` 와 1:1 매칭 → baseline `all_features.npz` 의 빌드 내부 순서와 일치.

---

## 3. Phase L2 — 학습 (`run_train` → `dataset` → `train`)

### 3.1 데이터 로드 — `dataset.py::load_lstm_dataset`

```
[1] features.npz 로드 (baseline 산출물)
    features (N, 21), sample_ids (N,), build_ids (N,), targets (dict)

[2] 5개 빌드 캐시 로드 → concat
    for bid in build_ids:
        cache = h5py.File(cache_dir / f"crop_stacks_{bid}.h5")[...]
        stacks_list.append(cache["stacks"])         # (N_b, 70, 8, 8) float16
        lengths_list.append(cache["lengths"])

    cache_stacks  = concatenate(stacks_list,  axis=0)
    cache_lengths = concatenate(lengths_list, axis=0)

[3] 무결성 검증
    - 빌드별 SV 수 일치 (features.npz vs cache)
    - features.npz 가 build_ids 순으로 정렬돼 있는지 확인 → 아니면 정렬해서 stacks 와 매칭
    - len(cache_stacks) == N

[4] dict 반환
    {features, sample_ids, build_ids, targets, stacks, lengths}
```

### 3.2 정규화 / 필터링 — `dataset.py::build_normalized_dataset`

```
[1] valid mask 계산 (baseline 동일 + LSTM 안전장치)
    valid = (UTS not NaN) & (UTS >= 50)
          & (모든 target not NaN)
          & (모든 feature not NaN)
          & (lengths > 0)                      ← LSTM 추가 안전장치

[2] valid 만 남기고 모두 같은 인덱스로 슬라이스
    feats, sids, bids, stacks, lengths, targets

[3] feature-wise min/max 정규화 → [-1, 1]
    feats_norm = normalize(feats, f_min, f_max)

[4] target 별로 min/max 정규화
    tgt_norm[prop] = normalize(tgts[prop], t_min[prop], t_max[prop])

[5] 반환 dict
    features (norm), features_raw, sample_ids, build_ids,
    stacks, lengths, targets (norm), targets_raw, norm_params
```

### 3.3 K-Fold + 학습 — `train.py::train_all`

```
splits = create_cv_splits(sample_ids)            # baseline 함수 재사용 — sample-wise 5-fold

for prop in [YS, UTS, UE, TE]:
    targets = dataset["targets"][prop]
    for fold, (train_mask, val_mask) in enumerate(splits):
        result = train_single_fold(
            feats[train_mask], stacks[train_mask], lengths[train_mask], targets[train_mask],
            feats[val_mask],   stacks[val_mask],   lengths[val_mask],   targets[val_mask],
            device,
        )
        torch.save(result["model_state"], f"vppm_lstm_{short}_fold{fold}.pt")

# training_log.json — fold별 best_val_loss, epochs
```

### 3.4 단일 fold 학습 루프 — `train.py::train_single_fold`

```
[1] DataLoader 구성
    train_ds = VPPMLSTMDataset(feat_train, stacks_train, lengths_train, targets_train)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    (val_loader 도 동일)

[2] 모델 / optimizer / loss / early-stopper 초기화
    model     = VPPM_LSTM().to(device)
    optimizer = Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-4, weight_decay=0.0)
    criterion = L1Loss()
    stopper   = EarlyStopper(patience=50)        # baseline 의 그것 그대로 import

[3] epoch 루프 (max=5000)
    while epoch < MAX_EPOCHS:
        # train
        model.train()
        for feats, stacks, lengths, ys in train_loader:
            feats, stacks, ys = .to(device)      # lengths 는 cpu (pack_padded 요구사항)
            pred = model(feats, stacks, lengths) # (B, 1)
            loss = criterion(pred, ys)
            loss.backward()
            clip_grad_norm_(parameters, 1.0)
            optimizer.step()

        # validate
        model.eval()
        with no_grad():
            val_loss = mean( criterion(model(...), ys) for batch in val_loader )

        if stopper.check(val_loss, model):
            break                                  # 50 epoch 무개선 → 중단

[4] best state 복원 후 반환
    model.load_state_dict(stopper.best_state)
    return {model_state, history, best_val_loss, epochs}
```

### 3.5 데이터로더 한 배치의 흐름 — `dataset.py::collate_fn`

```
batch = [(feat21_i, stack_i, length_i, target_i), ... B 개]

  → feats   = stack(feats)        # (B, 21) float32
  → stacks  = stack(stacks).float()  # (B, 70, 8, 8) — float16 → float32 캐스팅 (BN/LSTM 호환)
  → lengths = stack(lengths)      # (B,) int64 — cpu (pack_padded_sequence 요구)
  → targets = stack(targets)      # (B, 1) float32
```

### 3.6 모델 forward 한 배치 — `model.py::VPPM_LSTM.forward`

```
inputs : feats21 (B, 21), stacks (B, 70, 8, 8), lengths (B,)

[1] CNN per-frame
    x = stacks.view(B*70, 1, 8, 8)
    x = cnn(x)                           # (B*70, 32)
    x = x.view(B, 70, 32)

[2] 가변 길이 LSTM
    packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, (h_n, _) = lstm(packed)
    h_last = h_n[-1]                     # (B, 16) — 각 시퀀스의 실제 마지막 step
    embed  = embed_proj(h_last)          # (B, 1)

[3] concat → MLP
    x = torch.cat([feats21, embed], 1)   # (B, 22)
    x = relu(fc1(x))
    x = dropout(x)
    return fc2(x)                        # (B, 1)
```

### 3.7 학습 후 즉시 평가 — `run.py::run_train`

```
train_all(...)  →  models/*.pt 20 개 저장
   ↓
evaluate_all(dataset, models_dir, device)
   ↓
save_metrics(results, output_dir=results_dir)        # baseline 함수 재사용
plot_correlation(results, output_dir=results_dir)
plot_scatter_uts(results, output_dir=results_dir)
```

---

## 4. Phase L3 — 평가 (`evaluate.py`)

### 4.1 호출 경로

```
run.py::run_evaluate(device)
  └─ load_lstm_dataset() → build_normalized_dataset()
  └─ evaluate_all(dataset, models_dir, device)
       └─ for prop in TARGET_PROPERTIES:
            for fold in 0..4:
              model = VPPM_LSTM(); model.load_state_dict(torch.load(f"vppm_lstm_{short}_fold{fold}.pt"))
              fr = _evaluate_fold(model, val 데이터, sample_ids, norm_params, prop, device)
              fold_rmses.append(fr.rmse)
       └─ save_metrics / plot_correlation / plot_scatter_uts  (baseline 함수)
```

### 4.2 단일 fold 평가 — `evaluate.py::_evaluate_fold`

```
[1] 배치 단위 추론 (batch_size=1024)
    for i0 in 0..N step 1024:
        f = feats[i0:i1].to(device)
        s = stacks[i0:i1].to(device)
        l = lengths[i0:i1].cpu().int64
        preds_norm[i0:i1] = model(f, s, l).cpu().numpy()

[2] 역정규화 — MPa/% 복원
    pred_raw = denormalize(preds_norm, t_min, t_max)

[3] per-sample min 집계
    sid 별로 SV 예측의 최솟값 = 그 시편의 예측 (가장 약한 지점이 파단을 결정)
    preds[s] = min(pred_raw_of_SVs_in_sample(s))
    trues[s] = ground truth

[4] RMSE
    rmse = sqrt( mean( (preds - trues)^2 ) )
```

### 4.3 fold 통합 통계 — `evaluate_all` 의 후처리

```
mean_rmse = mean(fold_rmses)
std_rmse  = std(fold_rmses)
naive_rmse = sqrt( mean( (target_mean - targets_raw)^2 ) )    # 전체 평균을 예측값으로
reduction = naive_rmse - mean_rmse
reduction_pct = reduction / naive_rmse * 100

results[prop] = {vppm_rmse_mean, vppm_rmse_std, naive_rmse,
                 reduction, reduction_pct, fold_rmses,
                 all_predictions, all_ground_truths}
```

---

## 5. 산출물 시점별 매핑

| 시점 | 산출물 | 위치 |
|---|---|---|
| L1 시작 | `experiment_meta.json` | `experiments/vppm_lstm/` |
| L1 완료 | `crop_stacks_B1.{1..5}.h5` | `experiments/vppm_lstm/cache/` |
| L2 시작 | `normalization.json` | `experiments/vppm_lstm/features/` |
| L2 fold 완료 | `vppm_lstm_{short}_fold{k}.pt` | `experiments/vppm_lstm/models/` |
| L2 완료 | `training_log.json` | `experiments/vppm_lstm/models/` |
| L3 완료 | `metrics_summary.json`, `metrics_raw.json`, `predictions_*.csv`, `correlation_plots.png`, `scatter_plot_uts.png` | `experiments/vppm_lstm/results/` |

---

## 6. 모듈 간 의존성

```
run.py
  ├─ uses ─→ crop_stacks.py::build_cache
  ├─ uses ─→ dataset.py::load_lstm_dataset, build_normalized_dataset
  ├─ uses ─→ train.py::train_all
  ├─ uses ─→ evaluate.py::evaluate_all
  └─ uses ─→ baseline/evaluate.py::save_metrics, plot_correlation, plot_scatter_uts
                                   (재사용 — 변경 없음)

train.py
  ├─ uses ─→ dataset.py::VPPMLSTMDataset, collate_fn
  ├─ uses ─→ model.py::VPPM_LSTM
  ├─ uses ─→ baseline/train.py::EarlyStopper      (재사용)
  └─ uses ─→ common/dataset.py::create_cv_splits  (재사용 — sample-wise k-fold)

evaluate.py
  ├─ uses ─→ model.py::VPPM_LSTM
  └─ uses ─→ common/dataset.py::create_cv_splits, denormalize  (재사용)

dataset.py
  ├─ uses ─→ common/dataset.py::normalize          (재사용)
  └─ uses ─→ common/config.py                      (LSTM_* 상수)

crop_stacks.py
  ├─ uses ─→ common/supervoxel.py::SuperVoxelGrid, find_valid_supervoxels  (재사용)
  └─ uses ─→ common/config.py                      (LSTM_* 상수, hdf5_path)

model.py
  └─ uses ─→ common/config.py                      (모든 LSTM_* 하이퍼파라미터)
```

> 핵심: **신규 코드는 LSTM 임베딩과 데이터 로딩에만 한정**, baseline 의 `EarlyStopper / save_metrics / plot_* / create_cv_splits / normalize / denormalize / SuperVoxelGrid / find_valid_supervoxels` 는 그대로 import 해 사용 → 일관성 유지 + 코드 중복 최소화.

---

## 7. Smoke test 흐름 (`--quick`)

```
python -m Sources.vppm.lstm.run --all --quick
  ↓
run.py 가 config.LSTM_MAX_EPOCHS=20, EARLY_STOP_PATIENCE=10 으로 in-place 덮어쓰기
  ↓
[L1] cache 빌드 — 그대로 (이미 있으면 skip)
[L2] train  — 20 epoch / patience 10 으로 빠르게 종료
[L3] evaluate — 동일
  → 기능 검증 / 디버깅 용도. 실제 성능 측정은 `--quick` 없이 재실행 필요.
```

---

## 8. Docker 흐름 (`docker/lstm/`)

```
docker compose -f docker/lstm/docker-compose.yml up -d --build

  → docker-compose.yml 이 ORNL_Data / pipeline_outputs / venv 를 bind mount
  → entrypoint.sh:
       (a) 마운트 검증 + 쓰기 권한 검증
       (b) experiments/vppm_lstm/{cache,models,results,features} mkdir
       (c) python -m Sources.vppm.lstm.run --phase $LSTM_PHASE $LSTM_EXTRA
  → 컨테이너가 SSH 끊겨도 백그라운드 실행
  → 로그: docker compose ... logs -f
```

`.env` 의 환경변수:
- `UID_GID` — 호스트 권한과 일치
- `NVIDIA_VISIBLE_DEVICES` — GPU 인덱스 (예: `0` 또는 `2`)
- `LSTM_PHASE` — `cache | train | evaluate | all`
- `LSTM_EXTRA` — 추가 인자 (예: `--quick`, `--builds B1.1 B1.2`)

---

## 9. 실패 시 디버깅 포인트

| 증상 | 가능한 원인 | 확인 |
|---|---|---|
| `FileNotFoundError: L1 캐시 누락` | cache phase 가 안 돌았거나 빌드 누락 | `ls Sources/pipeline_outputs/experiments/vppm_lstm/cache/` |
| `features.npz 와 L1 캐시의 빌드별 SV 수가 다릅니다` | features.npz 또는 캐시 중 하나가 stale | features.npz 또는 캐시 한쪽 재생성 |
| `매칭 실패: features N=..., stacks N=...` | 빌드 순서 또는 valid SV 정의 변경 | `find_valid_supervoxels()` 로직 확인, 양쪽 재생성 |
| LSTM loss=NaN | 그래디언트 폭발 / 입력 NaN | `LSTM_GRAD_CLIP` 활성, `valid_mask` 가 NaN 제거하는지 확인 |
| 학습이 안 흐름 (val loss 정체) | LR 너무 큼 / 너무 작음, 또는 임베딩 차원 1 의 표현력 한계 | `--d-embed 16` 시도, `--bidirectional` 시도 |
| GPU OOM | `LSTM_BATCH_SIZE=256` 너무 큼 | 128 또는 64 로 감소 (run.py 에 CLI flag 추가 필요) |
| `pack_padded_sequence` 에러 | lengths 가 cpu 가 아님 / 0 길이 포함 | `lengths > 0` mask 적용 확인 (이미 build_normalized_dataset 에서 처리) |
