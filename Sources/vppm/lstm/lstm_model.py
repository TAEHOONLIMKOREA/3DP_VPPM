"""Phase L2 — LSTM 회귀 모델 (mode = 'fwd' or 'bidir').

Architecture:
    seq (B, T, 1, H, W)
       │
       ▼ PatchEncoder (CNN)
    (B, T, d_cnn=64)
       │
       ▼ LSTM(hidden=LSTM_D_HIDDEN, bidirectional=mode_bidir)
    h_n  (B, hidden_out)   # hidden_out = LSTM_D_HIDDEN * (2 if bidir else 1)
       │
       ▼ LayerNorm + Linear(hidden_out → LSTM_D_EMBED=1)
    extra_feat (B, 1)        # 21 차원에 concat 될 *추가 스칼라*
       │
       ▼ Linear(LSTM_D_EMBED=1 → 4)        (학습용 헤드)
    preds (B, 4) — YS / UTS / UE / TE
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..common import config
from .encoder import PatchEncoder


class SampleLSTMRegressor(nn.Module):
    """CNN encoder → LSTM → 1-dim feature → 4-output head."""

    def __init__(
        self,
        bidirectional: bool,
        d_cnn: int = config.LSTM_D_CNN,
        d_hidden: int = config.LSTM_D_HIDDEN,
        d_embed: int = config.LSTM_D_EMBED,
        num_layers: int = config.LSTM_NUM_LAYERS,
        pooling: str = config.LSTM_POOLING,
        n_targets: int = 4,
    ):
        super().__init__()
        self.encoder = PatchEncoder(in_channels=1, d_cnn=d_cnn)
        self.lstm = nn.LSTM(
            input_size=d_cnn,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.bidir = bidirectional
        self.num_layers = num_layers
        self.pooling = pooling
        self.d_hidden = d_hidden

        n_dir = 2 if bidirectional else 1
        hidden_out = d_hidden * n_dir

        # hidden → 추가 피처 (스칼라, d_embed=1)
        self.norm = nn.LayerNorm(hidden_out)
        self.proj = nn.Linear(hidden_out, d_embed)

        # 추가 피처 → 4 물성 회귀 (학습용 헤드)
        self.head = nn.Linear(d_embed, n_targets)

    def encode(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """(B, T, 1, H, W) → (B, d_embed). 학습 / 추출 공용."""
        B, T, C, H, W = seq.shape
        flat = seq.reshape(B * T, C, H, W)
        feats = self.encoder(flat)                          # (B*T, d_cnn)
        feats = feats.reshape(B, T, -1)

        lengths_cpu = lengths.cpu()
        packed = pack_padded_sequence(feats, lengths_cpu, batch_first=True, enforce_sorted=False)
        out, (h_n, _) = self.lstm(packed)

        if self.pooling == "last":
            n_dir = 2 if self.bidir else 1
            last = h_n.view(self.num_layers, n_dir, B, -1)[-1]   # (n_dir, B, hidden)
            if n_dir == 2:
                hidden_concat = torch.cat([last[0], last[1]], dim=1)  # (B, 2*hidden)
            else:
                hidden_concat = last[0]                                # (B, hidden)
        elif self.pooling == "mean":
            from torch.nn.utils.rnn import pad_packed_sequence
            unpacked, _ = pad_packed_sequence(out, batch_first=True)   # (B, T, hidden_out)
            mask = torch.arange(unpacked.shape[1], device=unpacked.device)[None, :] < lengths[:, None]
            hidden_concat = (unpacked * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(1).clamp_min(1)
        else:
            raise ValueError(f"unknown pooling: {self.pooling}")

        emb = self.proj(self.norm(hidden_concat))            # (B, d_embed=1)
        return emb

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """returns (preds (B, 4), embedding (B, d_embed))."""
        emb = self.encode(seq, lengths)
        preds = self.head(emb)
        return preds, emb
