"""VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-Sensor-1 모델.

7-분기 통합 (fullstack 와 동일하되 sensor 분기만 LSTM 으로 swap):
  stack_v0    (B, T, 8, 8)        ──[CNN(in=1)+LSTM+proj(d=16)]────> embed_v0   (B, 16)
  stack_v1    (B, T, 8, 8)        ──[CNN(in=1)+LSTM+proj(d=16)]────> embed_v1   (B, 16)
  sensors     (B, T, 7)           ──[multi-ch LSTM+proj(d=1)]──────> embed_s    (B, 1)   ★ 변경
  dscnn       (B, T, 8)           ──[LSTM+proj(d=8)]────────────────> embed_d    (B, 8)
  cad_patch   (B, T, 2, 8, 8)     ──[CNN(in=2)+LSTM+proj(d=8)]──────> embed_c    (B, 8)
  scan_patch  (B, T, 2, 8, 8)     ──[CNN(in=2)+LSTM+proj(d=8)]──────> embed_sc   (B, 8)
  feat_static (B, 2)              ── (build_height, laser_module) ──> feat2      (B, 2)

  feat2 ⊕ embed_v0 ⊕ embed_v1 ⊕ embed_s ⊕ embed_d ⊕ embed_c ⊕ embed_sc
  = 2 + 16 + 16 + 1 + 8 + 8 + 8 = 59
                       │
                       ▼
              MLP(59 → 256 → 128 → 64 → 1)

설계: PLAN.md
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import config
from ..lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.model import (
    _GroupLSTMBranch,
    _LSTMBranch,
)
from ..lstm_dual_img_4_sensor_7.model import _SensorLSTMBranch


class VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_Sensor_1(nn.Module):
    """7-분기 통합 모델. MLP 입력 59 = 2 + 16 + 16 + 1 + 8 + 8 + 8.

    fullstack `_1dcnn_sensor_4` 와의 유일한 차이는 sensor 분기:
      fullstack: per-field 1D-CNN ×7, d_per_field=4 → 28 dim
      본 실험:   단일 multi-channel LSTM (in=7, hid=16) → proj(d=1) → 1 dim
    """

    def __init__(
        self,
        # 카메라 (d_embed 16, in_channels=1)
        d_cnn: int = config.LSTM_D_CNN,
        d_hidden_cam: int = config.LSTM_FULL59_D_HIDDEN_CAM,
        d_embed_v0: int = config.LSTM_FULL59_D_EMBED_V0,
        d_embed_v1: int = config.LSTM_FULL59_D_EMBED_V1,
        # Sensor — 단일 multi-channel LSTM (★ 변경)
        n_sensor_ch: int = config.LSTM_FULL59_N_SENSOR_CH,
        d_hidden_s: int = config.LSTM_FULL59_D_HIDDEN_S,
        d_embed_s: int = config.LSTM_FULL59_D_EMBED_S,
        num_layers_s: int = config.LSTM_FULL59_NUM_LAYERS_S,
        bidirectional_s: bool = config.LSTM_FULL59_BIDIRECTIONAL_S,
        # DSCNN LSTM (스칼라 시퀀스)
        n_dscnn_ch: int = config.LSTM_FULL59_N_DSCNN_CH,
        d_hidden_d: int = config.LSTM_FULL59_D_HIDDEN_D,
        d_embed_d: int = config.LSTM_FULL59_D_EMBED_D,
        # CAD spatial-CNN+LSTM (in_channels=2, d=8)
        n_cad_ch: int = config.LSTM_FULL59_N_CAD_CH,
        d_cnn_c: int = config.LSTM_FULL59_D_CNN_C,
        d_hidden_c: int = config.LSTM_FULL59_D_HIDDEN_C,
        d_embed_c: int = config.LSTM_FULL59_D_EMBED_C,
        # Scan spatial-CNN+LSTM (in_channels=2, d=8)
        n_scan_ch: int = config.LSTM_FULL59_N_SCAN_CH,
        d_cnn_sc: int = config.LSTM_FULL59_D_CNN_SC,
        d_hidden_sc: int = config.LSTM_FULL59_D_HIDDEN_SC,
        d_embed_sc: int = config.LSTM_FULL59_D_EMBED_SC,
        # 결합 MLP — 59 → 256 → 128 → 64 → 1
        mlp_hidden: tuple[int, int, int] = config.LSTM_FULL59_MLP_HIDDEN,
        dropout: float = config.DROPOUT_RATE,
    ):
        super().__init__()

        # 카메라 v0/v1 (in_channels=1) — fullstack 모듈 재사용
        self.branch_v0 = _LSTMBranch(
            in_channels=1, d_cnn=d_cnn, d_hidden=d_hidden_cam, d_embed=d_embed_v0,
        )
        self.branch_v1 = _LSTMBranch(
            in_channels=1, d_cnn=d_cnn, d_hidden=d_hidden_cam, d_embed=d_embed_v1,
        )
        # Sensor 단일 multi-channel LSTM (★ fullstack 의 per-field 1D-CNN 대체)
        self.branch_sensor = _SensorLSTMBranch(
            n_channels=n_sensor_ch,
            d_hidden=d_hidden_s,
            d_embed=d_embed_s,
            num_layers=num_layers_s,
            bidirectional=bidirectional_s,
        )
        # DSCNN scalar 시퀀스 LSTM
        self.branch_dscnn = _GroupLSTMBranch(n_dscnn_ch, d_hidden_d, d_embed_d)
        # CAD spatial-CNN+LSTM (in_channels=2)
        self.branch_cad = _LSTMBranch(
            in_channels=n_cad_ch, d_cnn=d_cnn_c, d_hidden=d_hidden_c, d_embed=d_embed_c,
        )
        # Scan spatial-CNN+LSTM (in_channels=2)
        self.branch_scan = _LSTMBranch(
            in_channels=n_scan_ch, d_cnn=d_cnn_sc, d_hidden=d_hidden_sc, d_embed=d_embed_sc,
        )

        # MLP 입력 차원 계산
        n_static = len(config.LSTM_FULL59_STATIC_IDX)                            # = 2
        n_total = (
            n_static
            + d_embed_v0 + d_embed_v1
            + d_embed_s                                                          # 1 (★ fullstack 28 → 1)
            + d_embed_d + d_embed_c + d_embed_sc
        )                                                                         # = 59
        # MLP: 59 → 256 → 128 → 64 → 1
        h1, h2, h3 = mlp_hidden
        self.fc1 = nn.Linear(n_total, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, 1)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """MLP fc1~fc4 + 모든 분기 proj 를 baseline 과 동일 N(0, 0.1) 초기화."""
        projs = [
            self.branch_v0.proj, self.branch_v1.proj,
            self.branch_sensor.proj,                                              # 단일 proj (per-field 7개 → 1개)
            self.branch_dscnn.proj, self.branch_cad.proj, self.branch_scan.proj,
        ]
        for m in (self.fc1, self.fc2, self.fc3, self.fc4, *projs):
            nn.init.normal_(m.weight, mean=0.0, std=config.WEIGHT_INIT_STD)
            nn.init.zeros_(m.bias)

    def encode_all(
        self,
        stacks_v0: torch.Tensor, stacks_v1: torch.Tensor,
        sensors: torch.Tensor, dscnn: torch.Tensor,
        cad_patch: torch.Tensor, scan_patch: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        e_v0 = self.branch_v0(stacks_v0, lengths)
        e_v1 = self.branch_v1(stacks_v1, lengths)
        e_s = self.branch_sensor(sensors, lengths)
        e_d = self.branch_dscnn(dscnn, lengths)
        e_c = self.branch_cad(cad_patch, lengths)
        e_sc = self.branch_scan(scan_patch, lengths)
        return e_v0, e_v1, e_s, e_d, e_c, e_sc

    def forward(
        self,
        feats_static: torch.Tensor,           # (B, 2)
        stacks_v0: torch.Tensor,              # (B, T, 8, 8)
        stacks_v1: torch.Tensor,              # (B, T, 8, 8)
        sensors: torch.Tensor,                # (B, T, 7)
        dscnn: torch.Tensor,                  # (B, T, 8)
        cad_patch: torch.Tensor,              # (B, T, 2, 8, 8)
        scan_patch: torch.Tensor,             # (B, T, 2, 8, 8)
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        e_v0, e_v1, e_s, e_d, e_c, e_sc = self.encode_all(
            stacks_v0, stacks_v1, sensors, dscnn, cad_patch, scan_patch, lengths,
        )
        x = torch.cat([feats_static, e_v0, e_v1, e_s, e_d, e_c, e_sc], dim=1)    # (B, 59)
        x = F.relu(self.fc1(x)); x = self.dropout(x)                              # → 256
        x = F.relu(self.fc2(x)); x = self.dropout(x)                              # → 128
        x = F.relu(self.fc3(x)); x = self.dropout(x)                              # → 64
        return self.fc4(x)                                                        # → 1
