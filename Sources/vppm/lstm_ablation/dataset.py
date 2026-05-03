"""LSTM-Full-Stack Ablation 데이터셋 — base 의 7-입력 dataloader 를 그대로 재사용.

모델 forward 가 미사용 카메라 분기를 알아서 skip 하므로 dataset 단에서 입력을
빼낼 필요가 없다. 정규화 통계도 base 와 동일하게 8-입력 모두 정규화 (실제 학습에
들어가는 차원만 모델 fc1 입력으로 들어감).

> 메모리 절감 옵션 (PLAN §3.2): E2 에서 v0/v1 캐시를 미로드. 1차는 단순 처리.
"""
from __future__ import annotations

from ..lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.dataset import (
    VPPMLSTMSeptetDataset,
    build_normalized_dataset,
    collate_fn,
    load_septet_dataset,
)

__all__ = [
    "VPPMLSTMSeptetDataset",
    "build_normalized_dataset",
    "collate_fn",
    "load_septet_dataset",
]
