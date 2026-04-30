"""dual_4 데이터셋 — dual 과 동일 구조 재사용 (캐시 공유, feature 차원만 다름).

stacks_v0/stacks_v1 의 shape 와 normalize 로직은 dual 과 비트 단위 동일.
실제 모델에서 d_embed 만 1 → 4 로 바뀌므로 데이터 파이프라인은 변경 없음.
"""
from ..lstm_dual.dataset import (  # noqa: F401
    VPPMLSTMDualDataset,
    build_normalized_dataset,
    collate_fn,
    load_dual_dataset,
)
