"""VPPM-LSTM-Dual-4 모델 — `VPPM_LSTM_Dual` 의 d_embed=4 변형.

projection 통로만 1 → 4 로 확장 (16-dim LSTM hidden 출력은 그대로).
구조/forward 시그니처는 dual 과 동일 — train/evaluate 코드를 그대로 재사용 가능.
"""
from __future__ import annotations

from ..common import config
from ..lstm_dual.model import VPPM_LSTM_Dual


class VPPM_LSTM_Dual_4(VPPM_LSTM_Dual):
    """d_embed_v0=4, d_embed_v1=4 default — 21 + 4 + 4 = 29-feat MLP 입력."""

    def __init__(self, **kwargs):
        kwargs.setdefault("d_embed_v0", config.LSTM_DUAL_4_D_EMBED_V0)
        kwargs.setdefault("d_embed_v1", config.LSTM_DUAL_4_D_EMBED_V1)
        super().__init__(**kwargs)
