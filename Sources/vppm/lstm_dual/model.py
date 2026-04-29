"""
VPPM-LSTM-Dual 모델 — visible/0 + visible/1 두 채널 각각 CNN+LSTM 임베딩 → concat → 23-feat MLP

흐름:
  stack_v0 (B, T, 8, 8) ──[CNN_v0+LSTM_v0+Linear]──> embed_v0 (B, 1)
  stack_v1 (B, T, 8, 8) ──[CNN_v1+LSTM_v1+Linear]──> embed_v1 (B, 1)
  feat21 ⊕ embed_v0 ⊕ embed_v1 ──> (B, 23)
                          ──MLP(23→128→1)──>           (B, 1)

`FrameCNN` 은 기존 `lstm/model.py` 에서 import 해 재사용. 채널별로 별도 instance.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from ..common import config
from ..lstm.model import FrameCNN


class _LSTMBranch(nn.Module):
    """단일 채널 분기: FrameCNN(per-frame) → LSTM(가변 길이) → Linear(d_embed)."""

    def __init__(self, d_cnn: int, d_hidden: int, d_embed: int,
                 num_layers: int, bidirectional: bool):
        super().__init__()
        self.bidirectional = bidirectional
        self.d_hidden = d_hidden

        self.cnn = FrameCNN(d_cnn=d_cnn)
        self.lstm = nn.LSTM(
            input_size=d_cnn,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        d_lstm_out = d_hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(d_lstm_out, d_embed)

    def forward(self, stacks: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # stacks: (B, T, H, W), lengths: (B,) int64 (cpu)
        B, T, H, W = stacks.shape
        x = stacks.view(B * T, 1, H, W)
        x = self.cnn(x)                  # (B*T, d_cnn)
        x = x.view(B, T, -1)             # (B, T, d_cnn)

        lengths_cpu = lengths.detach().cpu()
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]
        return self.proj(h_last)         # (B, d_embed)


class VPPM_LSTM_Dual(nn.Module):
    """visible/0 + visible/1 두 채널 CNN+LSTM 임베딩 + 결합 MLP."""

    def __init__(self,
                 n_baseline_feats: int = config.N_FEATURES,
                 d_cnn: int = config.LSTM_D_CNN,
                 d_hidden: int = config.LSTM_D_HIDDEN,
                 d_embed_v0: int = config.LSTM_DUAL_D_EMBED_V0,
                 d_embed_v1: int = config.LSTM_DUAL_D_EMBED_V1,
                 num_layers: int = config.LSTM_NUM_LAYERS,
                 bidirectional: bool = config.LSTM_BIDIRECTIONAL,
                 share_cnn: bool = config.LSTM_DUAL_SHARE_CNN,
                 share_lstm: bool = config.LSTM_DUAL_SHARE_LSTM,
                 hidden_dim: int = config.HIDDEN_DIM,
                 dropout: float = config.DROPOUT_RATE):
        super().__init__()
        self.share_cnn = share_cnn
        self.share_lstm = share_lstm

        self.branch_v0 = _LSTMBranch(
            d_cnn=d_cnn, d_hidden=d_hidden, d_embed=d_embed_v0,
            num_layers=num_layers, bidirectional=bidirectional,
        )

        if share_cnn or share_lstm:
            # 공유 모드 — 두 채널이 같은 CNN/LSTM 사용 (proj 만 분리해 채널별 임베딩 의미 보존)
            self.branch_v1 = _LSTMBranch(
                d_cnn=d_cnn, d_hidden=d_hidden, d_embed=d_embed_v1,
                num_layers=num_layers, bidirectional=bidirectional,
            )
            if share_cnn:
                self.branch_v1.cnn = self.branch_v0.cnn
            if share_lstm:
                self.branch_v1.lstm = self.branch_v0.lstm
        else:
            self.branch_v1 = _LSTMBranch(
                d_cnn=d_cnn, d_hidden=d_hidden, d_embed=d_embed_v1,
                num_layers=num_layers, bidirectional=bidirectional,
            )

        # 결합 MLP — baseline VPPM 동일 골격, 입력만 (21 + d_embed_v0 + d_embed_v1)
        n_total = n_baseline_feats + d_embed_v0 + d_embed_v1
        self.fc1 = nn.Linear(n_total, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self._init_mlp_weights()

    def _init_mlp_weights(self):
        """결합 MLP 와 두 proj 만 baseline 과 동일하게 N(0, 0.1) 초기화."""
        for m in (self.fc1, self.fc2,
                  self.branch_v0.proj, self.branch_v1.proj):
            nn.init.normal_(m.weight, mean=0.0, std=config.WEIGHT_INIT_STD)
            nn.init.zeros_(m.bias)

    def encode_dual(self, stacks_v0: torch.Tensor, stacks_v1: torch.Tensor,
                    lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed_v0 = self.branch_v0(stacks_v0, lengths)
        embed_v1 = self.branch_v1(stacks_v1, lengths)
        return embed_v0, embed_v1

    def forward(self, feats21: torch.Tensor,
                stacks_v0: torch.Tensor, stacks_v1: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        embed_v0, embed_v1 = self.encode_dual(stacks_v0, stacks_v1, lengths)
        x = torch.cat([feats21, embed_v0, embed_v1], dim=1)   # (B, 21 + d_embed_v0 + d_embed_v1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
