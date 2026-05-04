# vppm_lstm/cache/ — VPPM-LSTM 크롭 시퀀스 캐시 (visible/0)

> 본 캐시는 원본 HDF5 의 **`slices/camera_data/visible/0` (용융 직후 카메라) 채널만** 추출한 산출물.
> `visible/1` (분말 도포 후) 은 포함되지 않으며, 별도 실험인 `vppm_lstm_dual` 에서 따로 캐시함.

각 슈퍼복셀(SV) 의 z-범위 동안 카메라 이미지 8×8 크롭을 시간순으로 쌓은 가변 길이 시퀀스를 저장.

---

## 파일 목록

| 파일 | 빌드 | SV 수 | T_sv (min/med/max) | 크기 |
|---|:---:|---:|:---:|---:|
| `crop_stacks_B1.1.h5` | B1.1 (기준 공정) | 10,173 | 70 / 70 / 70 | 56.2 MB |
| `crop_stacks_B1.2.h5` | B1.2 (다양한 공정 파라미터) | 10,173 | 70 / 70 / 70 | 46.4 MB |
| `crop_stacks_B1.3.h5` | B1.3 (오버행 형상) | 2,840 | **36** / 70 / 70 | 13.2 MB |
| `crop_stacks_B1.4.h5` | B1.4 (스패터/가스) | 3,126 | 70 / 70 / 70 | 16.7 MB |
| `crop_stacks_B1.5.h5` | B1.5 (리코터 손상) | 9,735 | 70 / 70 / 70 | 53.5 MB |
| **합계** | | **36,047** | | **186.1 MB** |

> B1.3 (오버행) 만 일부 SV 의 `T_sv < 70` — 오버행 형상에서 파트 두께가 얇은 SV 는 z 방향 70 레이어 중 일부만 유효.

---

## HDF5 구조 (빌드별 동일)

```
crop_stacks_B1.x.h5
├── /stacks        (N_sv, 70, 8, 8)   float16   — 카메라 크롭 시퀀스 (uint8/255, zero-padded, gzip)
├── /lengths       (N_sv,)            int16     — 실제 시퀀스 길이 T_sv ∈ [1, 70]
├── /sv_indices    (N_sv, 3)          int32     — (ix, iy, iz) SV 그리드 좌표
├── /sample_ids    (N_sv,)            int32     — 시편 ID
└── attrs: T_max=70, H=W=8, camera_channel=0, build_id, n_sv,
          valid_layer_rule="part_ids>0 in SV xy region"
```

생성: `python -m Sources.vppm.lstm.run --phase cache`
사용: `Sources/vppm/lstm/dataset.py`, `Sources/vppm/lstm_dual/dataset.py`
