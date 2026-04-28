# Docker — Sample-LSTM (v2)

[Sources/vppm/lstm/PLAN_LSTM_v2.md](../../Sources/vppm/lstm/PLAN_LSTM_v2.md) 의 4 단계 파이프라인을
독립 도커 컨테이너로 실행한다.

## 4 가지 LSTM 모드

| Mode | LSTM | hidden | d_embed | n_feats | 의미 |
|:---|:---|:---:|:---:|:---:|:---|
| `fwd1`    | Forward       | 8 | 1  | **22** | 단방향 + 스칼라 |
| `bidir1`  | Bidirectional | 8 | 1  | **22** | 양방향 + 스칼라 |
| `fwd16`   | Forward       | 8 | 16 | **37** | 단방향 + 16-dim 임베딩 |
| **`bidir16`** | **Bidirectional** | **8** | **16** | **37** | **이전 v1 설계** |

같은 데이터로 4 모드를 별도 학습 + 평가. ablation:
- forward vs bidirectional (양방향 효과)
- d_embed=1 vs 16 (임베딩 차원 효과)

## 4 Phase

| Phase | 스크립트 | 도커 서비스 | mode | 산출물 | 예상 시간 |
|:----:|:------|:-----------|:----:|:------|:--------:|
| L1 | `run_cache.sh`   | lstm-cache   | 무관 | `sample_stacks/{B1.x}.h5` | ~30분 (CPU OK) |
| L2 | `run_train.sh`   | lstm-train   | 필요 | `models_lstm/{mode}/lstm_sample_fold{0-4}.pt` | ~1~2시간/모드 (GPU) |
| L3 | `run_extract.sh` | lstm-extract | 필요 | `lstm_embeddings/{mode}/embeddings.npz` + `features/all_features_with_lstm_{mode}.npz` | ~10분/모드 |
| L4 | `run_vppm.sh`    | lstm-vppm    | 필요 | `results/vppm_lstm_{mode}/` (RMSE / 예측) | ~30분/모드 |

전체 4 모드 실행: `./run_all.sh --mode all` (~12시간).

## 전제 조건

- 호스트 NVIDIA GPU + `nvidia-docker2` (compose `runtime: nvidia`)
- 호스트 `venv/` 가 torch + cuda 설치 (컨테이너는 bind-mount 로 사용)
- `Sources/pipeline_outputs/features/all_features.npz` 존재 (baseline 21 피처)
- `ORNL_Data_Origin/*.hdf5` 5 빌드 파일

## 실행

### 한 번에 (권장)

```bash
cd docker/lstm
./run_all.sh --mode all              # 4 모드 모두 (~12시간)
./run_all.sh --mode bidir16          # 이전 37 차원 설계만 (~3시간)
./run_all.sh --mode fwd1             # 새 22 차원 단방향만
./run_all.sh --mode all --gpu 2      # GPU 2 사용
./run_all.sh --mode all --quick      # smoke test (4 모드)
./run_all.sh --mode bidir16 --skip-cache  # 캐시 이미 있으면
```

### 단계별

```bash
# 1) 캐시 (mode 무관, 한 번만)
./run_cache.sh
./run_cache.sh --builds B1.2            # 빌드 1개만

# 2) 4 모드 각각 L2~L4
for M in fwd1 bidir1 fwd16 bidir16; do
  ./run_train.sh   --mode $M
  ./run_extract.sh --mode $M
  ./run_vppm.sh    --mode $M
done

# 또는 특정 모드만
./run_train.sh   --mode bidir16 --gpu 2 --quick  # smoke
./run_train.sh   --mode bidir16 --folds 0 1      # 특정 fold
./run_extract.sh --mode bidir16
./run_vppm.sh    --mode bidir16
```

## 산출물

```
Sources/pipeline_outputs/
├── sample_stacks/                # L1 (mode 무관)
│   ├── B1.1.h5 .. B1.5.h5
│   └── normalization.json
├── models_lstm/
│   ├── fwd1/                     bidir1/
│   ├── fwd16/                    bidir16/
│   │   ├── lstm_sample_fold{0-4}.pt
│   │   ├── training_summary.json
│   │   └── vppm_{YS,UTS,UE,TE}_fold{0-4}.pt
├── lstm_embeddings/
│   ├── fwd1/embeddings.npz       bidir1/embeddings.npz
│   ├── fwd16/embeddings.npz      bidir16/embeddings.npz
├── features/
│   ├── all_features.npz                       # 21 차원 baseline
│   ├── all_features_with_lstm_fwd1.npz        # 22 차원
│   ├── all_features_with_lstm_bidir1.npz      # 22 차원
│   ├── all_features_with_lstm_fwd16.npz       # 37 차원
│   └── all_features_with_lstm_bidir16.npz     # 37 차원 (이전 v1)
└── results/
    ├── vppm_lstm_fwd1/           vppm_lstm_bidir1/
    └── vppm_lstm_fwd16/          vppm_lstm_bidir16/
```

## Baseline 비교

각 L4 가 끝나면 baseline (21-feat) 과 LSTM ({22 또는 37}-feat) RMSE 가 콘솔에 출력됨:

```
[L4] === Baseline (21-feat) vs LSTM-bidir16 (37-feat) RMSE ===
  YS:  baseline 24.28 → lstm-bidir16 <X>  ΔRMSE=<...>
  UTS: baseline 42.88 → lstm-bidir16 <Y>  ΔRMSE=<...>
  UE:  baseline 9.34  → lstm-bidir16 <Z>  ΔRMSE=<...>
  TE:  baseline 11.27 → lstm-bidir16 <W>  ΔRMSE=<...>
```

## 트러블슈팅

- **`venv not mounted`** — 호스트 `venv/bin/python` 실행 가능 확인
- **`all_features.npz 없음`** — `./venv/bin/python -m Sources.vppm.run_pipeline --phase features` 먼저
- **`LSTM_MODE 환경변수 필요`** — `./run_*.sh --mode {fwd1|bidir1|fwd16|bidir16}`
- **권한 에러 (UID mismatch)** — `chown -R $(id -u):$(id -g) Sources/pipeline_outputs/{sample_stacks,models_lstm,lstm_embeddings}`
- **L1 메모리 부족** — `./run_cache.sh --builds B1.2` 로 한 빌드씩
- **체크포인트 모드 불일치** — extract 시 "checkpoint (bidir=X, d_embed=Y) 와 mode=Z 불일치" 에러
  → wrong directory. `models_lstm/{mode}/` 에는 해당 mode 모델만.
