"""
Phase 4: VPPM 퍼셉트론 모델
논문 Table 6, Section 2.11
"""
import torch.nn as nn
import torch.nn.functional as F

from . import config


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
