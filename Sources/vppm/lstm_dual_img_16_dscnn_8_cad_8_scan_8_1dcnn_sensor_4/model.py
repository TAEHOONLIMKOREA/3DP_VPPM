"""VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-1DCNN-Sensor-4 모델.

7-분기 통합:
  stack_v0    (B, T, 8, 8)        ──[CNN(in=1)+LSTM+proj(d=16)]────> embed_v0   (B, 16)
  stack_v1    (B, T, 8, 8)        ──[CNN(in=1)+LSTM+proj(d=16)]────> embed_v1   (B, 16)
  sensors     (B, T, 7)           ──[per-field 1D-CNN, 7×4]────────> embed_s    (B, 28)
  dscnn       (B, T, 8)           ──[LSTM+proj(d=8)]────────────────> embed_d    (B, 8)
  cad_patch   (B, T, 2, 8, 8)     ──[CNN(in=2)+LSTM+proj(d=8)]──────> embed_c    (B, 8)
  scan_patch  (B, T, 2, 8, 8)     ──[CNN(in=2)+LSTM+proj(d=8)]──────> embed_sc   (B, 8)
  feat_static (B, 2)              ── (build_height, laser_module) ──> feat2      (B, 2)

  feat2 ⊕ embed_v0 ⊕ embed_v1 ⊕ embed_s ⊕ embed_d ⊕ embed_c ⊕ embed_sc
  = 2 + 16 + 16 + 28 + 8 + 8 + 8 = 86
                       │
                       ▼
              MLP(86 → 256 → 128 → 64 → 1)

설계: PLAN.md
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from ..common import config


# ───────────────────────────────────────────────────────────────────────
# Per-frame CNN — in_channels 일반화
# ───────────────────────────────────────────────────────────────────────


class FrameCNN(nn.Module):
    """SV 8×8 패치 → d_cnn 차원 임베딩. in_channels 파라미터화.

    기존 `lstm/model.py:FrameCNN` 의 in_channels=1 하드코딩을 본 실험에서 일반화.
    카메라 (in=1) / CAD (in=2) / Scan (in=2) 공용.

    구조: Conv 3×3 (in→ch1) → BN → ReLU → Conv 3×3 (ch1→ch2) → BN → ReLU
        → AdaptiveAvgPool(1) → Linear(ch2 → d_cnn)
    """

    def __init__(self, d_cnn: int = config.LSTM_D_CNN,
                 in_channels: int = 1,
                 ch1: int = config.LSTM_CNN_CH1,
                 ch2: int = config.LSTM_CNN_CH2,
                 kernel: int = config.LSTM_CNN_KERNEL):
        super().__init__()
        pad = kernel // 2
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, ch1, kernel, padding=pad)
        self.bn1 = nn.BatchNorm2d(ch1)
        self.conv2 = nn.Conv2d(ch1, ch2, kernel, padding=pad)
        self.bn2 = nn.BatchNorm2d(ch2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(ch2, d_cnn)

    def forward(self, x):
        # x: (B*T, in_channels, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.gap(x).flatten(1)
        x = self.proj(x)
        return x


# ───────────────────────────────────────────────────────────────────────
# Spatial-CNN + LSTM 분기 — 카메라 / CAD / Scan 공용
# ───────────────────────────────────────────────────────────────────────


class _LSTMBranch(nn.Module):
    """spatial 패치 시퀀스 → per-frame CNN → LSTM → proj.

    재사용 케이스:
      - 카메라 v0:  in_channels=1, d_embed=16, 입력 (B, T, H, W)        — channel dim auto-add
      - 카메라 v1:  in_channels=1, d_embed=16, 입력 (B, T, H, W)
      - CAD-patch:  in_channels=2, d_embed=8,  입력 (B, T, 2, H, W)
      - Scan-patch: in_channels=2, d_embed=8,  입력 (B, T, 2, H, W)
    """

    def __init__(self, in_channels: int, d_cnn: int, d_hidden: int, d_embed: int,
                 num_layers: int = config.LSTM_NUM_LAYERS,
                 bidirectional: bool = config.LSTM_BIDIRECTIONAL):
        super().__init__()
        self.in_channels = in_channels
        self.bidirectional = bidirectional
        self.cnn = FrameCNN(d_cnn=d_cnn, in_channels=in_channels)
        self.lstm = nn.LSTM(
            input_size=d_cnn,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        d_lstm_out = d_hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(d_lstm_out, d_embed)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: 카메라 (B, T, H, W)  또는  멀티채널 (B, T, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(2)                     # (B, T, 1, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.cnn(x)                            # (B*T, d_cnn)
        x = x.reshape(B, T, -1)                    # (B, T, d_cnn)

        lengths_cpu = lengths.detach().cpu()
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]
        return self.proj(h_last)                   # (B, d_embed)


# ───────────────────────────────────────────────────────────────────────
# 다채널 스칼라 시퀀스 LSTM — DSCNN 전용
# ───────────────────────────────────────────────────────────────────────


class _GroupLSTMBranch(nn.Module):
    """다채널 스칼라 시퀀스 → LSTM → proj. **DSCNN 전용** (CAD/Scan 은 spatial 분기 사용)."""

    def __init__(self, n_channels: int, d_hidden: int, d_embed: int,
                 num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=d_hidden,
            num_layers=num_layers, batch_first=True, bidirectional=bidirectional,
        )
        d_lstm_out = d_hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(d_lstm_out, d_embed)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_channels), lengths: (B,) int64 (cpu)
        packed = pack_padded_sequence(
            x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]
        return self.proj(h_last)


# ───────────────────────────────────────────────────────────────────────
# Sensor 필드별 1D-CNN
# ───────────────────────────────────────────────────────────────────────


class _PerFieldConv1DBranch(nn.Module):
    """필드별 독립 1D-CNN. (B, T, n_fields) → (B, n_fields * d_per_field).

    각 필드를 1채널 1D-CNN 으로 통과 → AdaptiveAvgPool → flatten → linear.
    필드간 weight sharing 없음 (필드별 ModuleList).

    AdaptiveAvgPool 이 패딩(0)을 평균에 포함시킴 — T_sv 중간값 ~50/70 가정 시 신호 ~0.7× 약화.
    1차 결과 평탄하면 lengths-aware mean 으로 업그레이드 검토 (PLAN §11.8).
    """

    def __init__(self, n_fields: int, d_per_field: int = 4,
                 hidden_ch: int = 16, kernel_size: int = 5):
        super().__init__()
        self.n_fields = n_fields
        self.d_per_field = d_per_field
        pad = kernel_size // 2
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden_ch, kernel_size, padding=pad),
                nn.ReLU(),
                nn.Conv1d(hidden_ch, hidden_ch, kernel_size, padding=pad),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(d_per_field),    # (B, hidden_ch, d_per_field)
            )
            for _ in range(n_fields)
        ])
        self.proj = nn.ModuleList([
            nn.Linear(hidden_ch * d_per_field, d_per_field)
            for _ in range(n_fields)
        ])

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_fields), lengths 미사용 (AdaptiveAvgPool 압축)
        outs = []
        for fi in range(self.n_fields):
            xf = x[:, :, fi].unsqueeze(1)                    # (B, 1, T)
            yf = self.convs[fi](xf)                          # (B, hidden_ch, d_per_field)
            yf = yf.flatten(1)                               # (B, hidden_ch * d_per_field)
            yf = self.proj[fi](yf)                           # (B, d_per_field)
            outs.append(yf)
        return torch.cat(outs, dim=1)                        # (B, n_fields * d_per_field)


# ───────────────────────────────────────────────────────────────────────
# 메인 모델
# ───────────────────────────────────────────────────────────────────────


class VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_1DCNN_Sensor_4(nn.Module):
    """7-분기 통합 모델. MLP 입력 86 = 2 + 16 + 16 + 28 + 8 + 8 + 8."""

    def __init__(
        self,
        # 카메라 (d_embed 16, in_channels=1)
        d_cnn: int = config.LSTM_D_CNN,
        d_hidden_cam: int = config.LSTM_FULL86_D_HIDDEN_CAM,
        d_embed_v0: int = config.LSTM_FULL86_D_EMBED_V0,
        d_embed_v1: int = config.LSTM_FULL86_D_EMBED_V1,
        # Sensor per-field 1D-CNN
        n_sensor_fields: int = config.LSTM_FULL86_N_SENSOR_FIELDS,
        d_per_sensor_field: int = config.LSTM_FULL86_D_PER_SENSOR_FIELD,
        sensor_hidden_ch: int = config.LSTM_FULL86_SENSOR_HIDDEN_CH,
        sensor_kernel: int = config.LSTM_FULL86_SENSOR_KERNEL,
        # DSCNN LSTM (스칼라 시퀀스)
        n_dscnn_ch: int = config.LSTM_FULL86_N_DSCNN_CH,
        d_hidden_d: int = config.LSTM_FULL86_D_HIDDEN_D,
        d_embed_d: int = config.LSTM_FULL86_D_EMBED_D,
        # CAD spatial-CNN+LSTM (in_channels=2, d=8)
        n_cad_ch: int = config.LSTM_FULL86_N_CAD_CH,
        d_cnn_c: int = config.LSTM_FULL86_D_CNN_C,
        d_hidden_c: int = config.LSTM_FULL86_D_HIDDEN_C,
        d_embed_c: int = config.LSTM_FULL86_D_EMBED_C,
        # Scan spatial-CNN+LSTM (in_channels=2, d=8)
        n_scan_ch: int = config.LSTM_FULL86_N_SCAN_CH,
        d_cnn_sc: int = config.LSTM_FULL86_D_CNN_SC,
        d_hidden_sc: int = config.LSTM_FULL86_D_HIDDEN_SC,
        d_embed_sc: int = config.LSTM_FULL86_D_EMBED_SC,
        # 결합 MLP — 86 → 256 → 128 → 64 → 1
        mlp_hidden: tuple[int, int, int] = config.LSTM_FULL86_MLP_HIDDEN,
        dropout: float = config.DROPOUT_RATE,
    ):
        super().__init__()

        # 카메라 v0/v1 (in_channels=1)
        self.branch_v0 = _LSTMBranch(
            in_channels=1, d_cnn=d_cnn, d_hidden=d_hidden_cam, d_embed=d_embed_v0,
        )
        self.branch_v1 = _LSTMBranch(
            in_channels=1, d_cnn=d_cnn, d_hidden=d_hidden_cam, d_embed=d_embed_v1,
        )
        # Sensor per-field 1D-CNN
        self.branch_sensor = _PerFieldConv1DBranch(
            n_sensor_fields, d_per_sensor_field, sensor_hidden_ch, sensor_kernel,
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
        n_static = len(config.LSTM_FULL86_STATIC_IDX)                            # = 2
        n_total = (
            n_static
            + d_embed_v0 + d_embed_v1
            + n_sensor_fields * d_per_sensor_field
            + d_embed_d + d_embed_c + d_embed_sc
        )                                                                         # = 86
        # MLP: 86 → 256 → 128 → 64 → 1
        h1, h2, h3 = mlp_hidden
        self.fc1 = nn.Linear(n_total, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, 1)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """MLP fc1~fc4 + 모든 분기 proj 를 baseline 과 동일 N(0, 0.1) 초기화.

        CNN/LSTM/Conv1d 본체는 PyTorch 기본 초기화 그대로 (N(0, 0.1) 너무 작음).
        """
        projs = [
            self.branch_v0.proj, self.branch_v1.proj,
            self.branch_dscnn.proj, self.branch_cad.proj, self.branch_scan.proj,
        ]
        # _PerFieldConv1DBranch 의 필드별 proj 도 포함
        for proj in self.branch_sensor.proj:
            projs.append(proj)
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
        x = torch.cat([feats_static, e_v0, e_v1, e_s, e_d, e_c, e_sc], dim=1)    # (B, 86)
        x = F.relu(self.fc1(x)); x = self.dropout(x)                              # → 256
        x = F.relu(self.fc2(x)); x = self.dropout(x)                              # → 128
        x = F.relu(self.fc3(x)); x = self.dropout(x)                              # → 64
        return self.fc4(x)                                                        # → 1
