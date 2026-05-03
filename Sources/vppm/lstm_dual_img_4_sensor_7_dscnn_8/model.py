"""VPPM-LSTM-Dual-Img-4-Sensor-7-DSCNN-8 모델.

4-분기 LSTM:
  stack_v0    (B, T, 8, 8) ──[CNN+LSTM+proj(d_embed_v0=4)]──> embed_v0 (B, 4)
  stack_v1    (B, T, 8, 8) ──[CNN+LSTM+proj(d_embed_v1=4)]──> embed_v1 (B, 4)
  sensors     (B, T, 7)    ──[LSTM+proj(d_embed_s=7)]──>      embed_s  (B, 7)
  dscnn       (B, T, 8)    ──[LSTM+proj(d_embed_d=8)]──>      embed_d  (B, 8)

  feats6 (G3+G4) ⊕ embed_v0 ⊕ embed_v1 ⊕ embed_s ⊕ embed_d ──> (B, 29) ──MLP──> (B, 1)

설계: PLAN.md
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from ..common import config
from ..lstm_dual.model import _LSTMBranch                      # 카메라 분기 재사용
from ..lstm_dual_img_4_sensor_7.model import _SensorLSTMBranch  # sensor 분기 재사용


class _DSCNNLSTMBranch(nn.Module):
    """8-channel DSCNN segmentation 시퀀스 → LSTM → proj(d_embed_d).

    sensor_7 의 _SensorLSTMBranch 와 코드 본질이 동일 (채널 수와 d_embed 만 다름).
    이번 실험에서는 통합 미루고 별도 클래스로 둠 (PLAN.md §4 명시).
    """

    def __init__(self, n_channels: int, d_hidden: int, d_embed: int,
                 num_layers: int, bidirectional: bool):
        super().__init__()
        self.bidirectional = bidirectional
        self.d_hidden = d_hidden

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        d_lstm_out = d_hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(d_lstm_out, d_embed)

    def forward(self, dscnn: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # dscnn: (B, T, n_channels) float32, lengths: (B,) int64 (cpu)
        lengths_cpu = lengths.detach().cpu()
        packed = pack_padded_sequence(
            dscnn, lengths_cpu, batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]
        return self.proj(h_last)                   # (B, d_embed)


class VPPM_LSTM_Dual_Img_4_Sensor_7_DSCNN_8(nn.Module):
    """카메라 dual(v0+v1) + sensor + DSCNN 4분기 LSTM. G1·G2 평균 제거 → 6-feat baseline 사용.

    MLP 입력: 6 (G3+G4) + d_embed_v0(4) + d_embed_v1(4) + d_embed_s(7) + d_embed_d(8) = 29.
    """

    def __init__(self,
                 d_cnn: int = config.LSTM_D_CNN,
                 d_hidden: int = config.LSTM_D_HIDDEN,
                 d_embed_v0: int = config.LSTM_DUAL_4_D_EMBED_V0,
                 d_embed_v1: int = config.LSTM_DUAL_4_D_EMBED_V1,
                 n_sensor_channels: int = config.LSTM_DUAL_IMG_4_SENSOR_7_N_CHANNELS,
                 d_hidden_s: int = config.LSTM_DUAL_IMG_4_SENSOR_7_D_HIDDEN_S,
                 d_embed_s: int = config.LSTM_DUAL_IMG_4_SENSOR_7_D_EMBED_S,
                 n_dscnn_channels: int = config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_N_CHANNELS,
                 d_hidden_d: int = config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_D_HIDDEN_D,
                 d_embed_d: int = config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_D_EMBED_D,
                 num_layers: int = config.LSTM_NUM_LAYERS,
                 bidirectional: bool = config.LSTM_BIDIRECTIONAL,
                 num_layers_s: int = config.LSTM_DUAL_IMG_4_SENSOR_7_NUM_LAYERS,
                 bidirectional_s: bool = config.LSTM_DUAL_IMG_4_SENSOR_7_BIDIRECTIONAL,
                 num_layers_d: int = config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_NUM_LAYERS,
                 bidirectional_d: bool = config.LSTM_DUAL_IMG_4_SENSOR_7_DSCNN_8_BIDIRECTIONAL,
                 share_cnn: bool = config.LSTM_DUAL_SHARE_CNN,
                 share_lstm: bool = config.LSTM_DUAL_SHARE_LSTM,
                 hidden_dim: int = config.HIDDEN_DIM,
                 dropout: float = config.DROPOUT_RATE,
                 n_baseline_feats: int = config.N_FEATURES,
                 n_sensor_feats_dropped: int = len(config.FEATURE_GROUPS["sensor"]),
                 n_dscnn_feats_dropped: int = len(config.FEATURE_GROUPS["dscnn"])):
        super().__init__()
        self.share_cnn = share_cnn
        self.share_lstm = share_lstm

        # 카메라 v0 / v1 — lstm_dual.model._LSTMBranch 재사용
        self.branch_v0 = _LSTMBranch(
            d_cnn=d_cnn, d_hidden=d_hidden, d_embed=d_embed_v0,
            num_layers=num_layers, bidirectional=bidirectional,
        )
        self.branch_v1 = _LSTMBranch(
            d_cnn=d_cnn, d_hidden=d_hidden, d_embed=d_embed_v1,
            num_layers=num_layers, bidirectional=bidirectional,
        )
        if share_cnn:
            self.branch_v1.cnn = self.branch_v0.cnn
        if share_lstm:
            self.branch_v1.lstm = self.branch_v0.lstm

        # Sensor 분기 — lstm_dual_img_4_sensor_7.model._SensorLSTMBranch 재사용
        self.branch_sensor = _SensorLSTMBranch(
            n_channels=n_sensor_channels,
            d_hidden=d_hidden_s, d_embed=d_embed_s,
            num_layers=num_layers_s, bidirectional=bidirectional_s,
        )

        # DSCNN 분기 (신규)
        self.branch_dscnn = _DSCNNLSTMBranch(
            n_channels=n_dscnn_channels,
            d_hidden=d_hidden_d, d_embed=d_embed_d,
            num_layers=num_layers_d, bidirectional=bidirectional_d,
        )

        # 결합 MLP — 입력 = 6 (G3+G4) + d_embed_v0 + d_embed_v1 + d_embed_s + d_embed_d
        n_baseline_kept = (
            n_baseline_feats - n_sensor_feats_dropped - n_dscnn_feats_dropped
        )                                                       # 21 - 7 - 8 = 6
        n_total = (
            n_baseline_kept + d_embed_v0 + d_embed_v1 + d_embed_s + d_embed_d
        )                                                       # 6 + 4 + 4 + 7 + 8 = 29
        self.fc1 = nn.Linear(n_total, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self._init_mlp_weights()

    def _init_mlp_weights(self):
        """MLP 와 네 proj 만 baseline 과 동일하게 N(0, 0.1) 초기화."""
        for m in (self.fc1, self.fc2,
                  self.branch_v0.proj, self.branch_v1.proj,
                  self.branch_sensor.proj, self.branch_dscnn.proj):
            nn.init.normal_(m.weight, mean=0.0, std=config.WEIGHT_INIT_STD)
            nn.init.zeros_(m.bias)

    def encode_all(self, stacks_v0: torch.Tensor, stacks_v1: torch.Tensor,
                   sensors: torch.Tensor, dscnn: torch.Tensor,
                   lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor, torch.Tensor]:
        embed_v0 = self.branch_v0(stacks_v0, lengths)
        embed_v1 = self.branch_v1(stacks_v1, lengths)
        embed_s = self.branch_sensor(sensors, lengths)
        embed_d = self.branch_dscnn(dscnn, lengths)
        return embed_v0, embed_v1, embed_s, embed_d

    def forward(self,
                feats6: torch.Tensor,
                stacks_v0: torch.Tensor, stacks_v1: torch.Tensor,
                sensors: torch.Tensor, dscnn: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        # feats6:     (B, 6) — G1·G2 제거된 baseline (CAD 3 + Scan 3)
        # stacks_v*:  (B, T, 8, 8)
        # sensors:    (B, T, 7)
        # dscnn:      (B, T, 8)
        # lengths:    (B,) int64 — 네 분기 공통
        embed_v0, embed_v1, embed_s, embed_d = self.encode_all(
            stacks_v0, stacks_v1, sensors, dscnn, lengths,
        )
        x = torch.cat([feats6, embed_v0, embed_v1, embed_s, embed_d], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
