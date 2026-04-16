"""
Phase L2 — CNN 패치 인코더.

입력:  (B, C, Hs, Ws)
출력:  (B, d_cnn)
"""
import torch
import torch.nn as nn


class PatchEncoder(nn.Module):
    """경량 CNN: 슈퍼복셀 패치 하나 → d_cnn 벡터.

    입력 패치는 작기 때문에 (8×8 ~ 10×10) 깊은 net 이 불필요.
    """

    def __init__(self, in_channels: int = 9, d_cnn: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(64, d_cnn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))
