"""Phase L2 — CNN 패치 인코더.

샘플 1 layer 의 64×64 raw camera 이미지 → d_cnn 차원 vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..common import config


class PatchEncoder(nn.Module):
    """64×64 raw → 64-dim vector."""

    def __init__(self, in_channels: int = 1, d_cnn: int = config.LSTM_D_CNN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2, bias=False),  # 32×32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),  # 16×16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),  # 8×8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, d_cnn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*T, 1, H, W) → (B*T, d_cnn)"""
        z = self.net(x).flatten(1)
        return self.proj(z)
