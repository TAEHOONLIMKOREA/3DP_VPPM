"""VPPM-1DCNN 모델 — channel-wise depthwise 1D CNN + baseline MLP.

흐름:
    x (B, 21, 70)
       ──depthwise Conv1d(k=3, groups=21)──> BN → ReLU
       ──depthwise Conv1d(k=3, groups=21)──> BN → ReLU
       ──AdaptiveAvgPool1d(1)──> squeeze       (B, 21)
       ──Linear(21→128) → ReLU → Dropout(0.1)──> Linear(128→1)

baseline VPPM 의 MLP 부분(``fc1``/``fc2``) 과 동일한 N(0, 0.1) 가중치 초기화 사용.
Conv/BN 은 PyTorch 기본 (Kaiming-uniform) — std=0.1 로 줄이면 활성이 죽는 위험.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from ..common import config as common_config
from . import config as exp_config


class VPPM_1DCNN(nn.Module):
    """Channel-wise depthwise 1D CNN + baseline MLP."""

    def __init__(
        self,
        n_channels: int = exp_config.N_CHANNELS,
        seq_len: int = exp_config.SEQ_LENGTH,
        kernel_size: int = exp_config.KERNEL_SIZE,
        hidden_dim: int = common_config.HIDDEN_DIM,
        dropout: float = common_config.DROPOUT_RATE,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len

        pad = kernel_size // 2
        # depthwise Conv1d × 2 (groups=n_channels → 채널 간 가중치 비공유)
        self.conv1 = nn.Conv1d(
            n_channels, n_channels, kernel_size,
            padding=pad, groups=n_channels,
        )
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.conv2 = nn.Conv1d(
            n_channels, n_channels, kernel_size,
            padding=pad, groups=n_channels,
        )
        self.bn2 = nn.BatchNorm1d(n_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # baseline MLP — 입력 차원 21
        self.fc1 = nn.Linear(n_channels, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self._init_mlp_weights()

    def _init_mlp_weights(self) -> None:
        """결합 MLP 만 baseline 과 동일하게 N(0, 0.1) 초기화."""
        for m in (self.fc1, self.fc2):
            nn.init.normal_(m.weight, mean=0.0, std=common_config.WEIGHT_INIT_STD)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 21, 70)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)           # (B, 21)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
