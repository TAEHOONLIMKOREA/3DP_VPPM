"""
Phase 4: VPPM 퍼셉트론 모델
논문 Table 6, Section 2.11

VPPM_LSTM: IMPLEMENTATION_PLAN_LSTM.md 에 따른 업그레이드.
기존 21-feat + CNN/LSTM 임베딩 concat → 기존 VPPM 헤드
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from ..lstm.sequence_model import SupervoxelLSTM


class VPPM(nn.Module):
    """Voxelized Property Prediction Model

    Architecture (논문 Table 6):
        FC(n_feats → 128) → ReLU → Dropout(0.1) → FC(128 → 1)
    """

    def __init__(self, n_feats: int = config.N_FEATURES,
                 hidden_dim: int = config.HIDDEN_DIM,
                 dropout: float = config.DROPOUT_RATE):
        super().__init__()
        self.fc1 = nn.Linear(n_feats, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화: N(0, 0.1)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.WEIGHT_INIT_STD)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class VPPM_LSTM(nn.Module):
    """VPPM 업그레이드 — 21개 핸드크래프트 피처 + CNN/LSTM 임베딩.

    입력:
        x21:  (B, 21)
        img:  (B, T, C, Hs, Ws)
        mask: (B, T) bool
    출력:
        pred: (B, 1)
    """

    def __init__(
        self,
        n_hand: int = config.N_FEATURES,
        in_channels: int = 9,
        d_cnn: int = config.LSTM_D_CNN,
        d_lstm: int = config.LSTM_D_EMBED,
        hidden_dim: int = config.HIDDEN_DIM,
        dropout: float = config.DROPOUT_RATE,
        bidirectional: bool = config.LSTM_BIDIRECTIONAL,
        num_lstm_layers: int = config.LSTM_NUM_LAYERS,
        pooling: str = config.LSTM_POOLING,
    ):
        super().__init__()
        self.seq = SupervoxelLSTM(
            in_channels=in_channels,
            d_cnn=d_cnn,
            d_lstm=d_lstm,
            bidirectional=bidirectional,
            num_layers=num_lstm_layers,
            pooling=pooling,
        )
        # LayerNorm 로 21-feat 과 LSTM 임베딩 신호 레벨을 맞춤
        self.norm_hand = nn.LayerNorm(n_hand)
        self.norm_lstm = nn.LayerNorm(d_lstm)

        self.fc1 = nn.Linear(n_hand + d_lstm, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self._init_fc()

    def _init_fc(self):
        for m in [self.fc1, self.fc2]:
            nn.init.normal_(m.weight, mean=0.0, std=config.WEIGHT_INIT_STD)
            nn.init.zeros_(m.bias)

    def forward(self, x21: torch.Tensor, img: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        emb = self.seq(img, mask)                        # (B, d_lstm)
        x = torch.cat([self.norm_hand(x21), self.norm_lstm(emb)], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    @torch.no_grad()
    def embed(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """학습 후 LSTM 임베딩만 추출 (export 용)."""
        return self.seq(img, mask)
