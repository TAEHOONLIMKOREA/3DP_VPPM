"""
VPPM-LSTM 모델 — CNN per-frame encoder + 가변 길이 LSTM + 결합 MLP

흐름:
  stack (B, T_max, 8, 8)  ──reshape──> (B*T_max, 1, 8, 8)
                          ──CNN──>     (B*T_max, d_cnn=32)
                          ──reshape──> (B, T_max, 32)
                          ──pack(lengths)──> packed
                          ──LSTM(hidden=16)──> h_n[-1] : (B, 16)
                          ──Linear(16→1)──>            (B, 1)
  feat21 (B, 21) ⊕ embed (B, 1) ──> (B, 22)
                          ──MLP(22→128→1)──>           (B, 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from ..common import config


class FrameCNN(nn.Module):
    """8×8 단일 프레임 → d_cnn 차원 임베딩.

    Conv 3×3 (1→16) → BN → ReLU → Conv 3×3 (16→32) → BN → ReLU
    → AdaptiveAvgPool(1) → Linear(32 → d_cnn=32)
    """

    def __init__(self, d_cnn: int = config.LSTM_D_CNN,
                 ch1: int = config.LSTM_CNN_CH1,
                 ch2: int = config.LSTM_CNN_CH2,
                 kernel: int = config.LSTM_CNN_KERNEL):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(1, ch1, kernel, padding=pad)
        self.bn1 = nn.BatchNorm2d(ch1)
        self.conv2 = nn.Conv2d(ch1, ch2, kernel, padding=pad)
        self.bn2 = nn.BatchNorm2d(ch2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(ch2, d_cnn)

    def forward(self, x):
        # x: (B*T, 1, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.gap(x).flatten(1)         # (B*T, ch2)
        x = self.proj(x)                   # (B*T, d_cnn)
        return x


class VPPM_LSTM(nn.Module):
    """CNN(per-frame) + LSTM(variable length) + 결합 MLP."""

    def __init__(self,
                 n_baseline_feats: int = config.N_FEATURES,
                 d_cnn: int = config.LSTM_D_CNN,
                 d_hidden: int = config.LSTM_D_HIDDEN,
                 d_embed: int = config.LSTM_D_EMBED,
                 num_layers: int = config.LSTM_NUM_LAYERS,
                 bidirectional: bool = config.LSTM_BIDIRECTIONAL,
                 hidden_dim: int = config.HIDDEN_DIM,
                 dropout: float = config.DROPOUT_RATE):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.d_hidden = d_hidden
        self.d_embed = d_embed

        self.cnn = FrameCNN(d_cnn=d_cnn)
        self.lstm = nn.LSTM(
            input_size=d_cnn,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # bidirectional 이면 forward+backward hidden concat → 2*d_hidden
        d_lstm_out = d_hidden * (2 if bidirectional else 1)
        self.embed_proj = nn.Linear(d_lstm_out, d_embed)

        # 결합 MLP — baseline VPPM 과 동일 구조 (입력만 22 또는 더 큼)
        n_total = n_baseline_feats + d_embed
        self.fc1 = nn.Linear(n_total, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self._init_mlp_weights()

    def _init_mlp_weights(self):
        """결합 MLP 만 baseline 과 동일하게 N(0, 0.1) 초기화.
        CNN/LSTM 은 PyTorch 기본 (Xavier-uniform 류) — 너무 작은 std 면 활성 죽음."""
        for m in (self.fc1, self.fc2):
            nn.init.normal_(m.weight, mean=0.0, std=config.WEIGHT_INIT_STD)
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.embed_proj.weight, mean=0.0, std=config.WEIGHT_INIT_STD)
        nn.init.zeros_(self.embed_proj.bias)

    def encode_sequence(self, stacks: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """(B, T_max, H, W) + lengths → (B, d_embed) LSTM 임베딩.

        T_max=70, H=W=8. lengths(B,) int64.
        """
        B, T, H, W = stacks.shape
        # CNN per-frame
        x = stacks.view(B * T, 1, H, W)              # (B*T, 1, 8, 8)
        x = self.cnn(x)                              # (B*T, d_cnn)
        x = x.view(B, T, -1)                         # (B, T, d_cnn)

        # 가변 길이 처리 — pack_padded_sequence 는 길이가 cpu int64 여야 함
        lengths_cpu = lengths.detach().cpu()
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * num_directions, B, d_hidden)
        if self.bidirectional:
            # 마지막 layer 의 forward + backward 만 concat
            h_fwd = h_n[-2]   # (B, d_hidden)
            h_bwd = h_n[-1]   # (B, d_hidden)
            h_last = torch.cat([h_fwd, h_bwd], dim=1)   # (B, 2*d_hidden)
        else:
            h_last = h_n[-1]   # (B, d_hidden)

        embed = self.embed_proj(h_last)              # (B, d_embed)
        return embed

    def forward(self, feats21: torch.Tensor,
                stacks: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        embed = self.encode_sequence(stacks, lengths)        # (B, d_embed)
        x = torch.cat([feats21, embed], dim=1)               # (B, 21+d_embed)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
