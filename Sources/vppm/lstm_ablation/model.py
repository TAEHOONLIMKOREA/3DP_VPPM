"""LSTM-Full-Stack Ablation 모델.

`vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.model` 의
`FrameCNN`, `_LSTMBranch`, `_PerFieldConv1DBranch`, `_GroupLSTMBranch` 를
import 재사용 (코드 복붙 금지) 하고, 7-flag 로 모든 분기 / static 피처를
토글할 수 있는 메인 클래스 `VPPM_LSTM_FullStack_Ablation` 을 정의한다.

시리즈 1 — 카메라 분기 제거 (use_v0/v1 toggle, 나머지 True):
- E1 (use_v0=False, use_v1=True):  MLP 입력 70 = 2 + 0 + 16 + 28 + 8 + 8 + 8
- E2 (use_v0=False, use_v1=False): MLP 입력 54 = 2 + 0 +  0 + 28 + 8 + 8 + 8

시리즈 2 — 단일 분기 isolation (use_static=False, 하나만 True):
- E3 (use_v0=True  only): MLP 입력 16 = 0 + 16 + 0 + 0 + 0 + 0 + 0
- E4 (use_dscnn=True only): MLP 입력  8 = 0 + 0 + 0 + 0 + 8 + 0 + 0
- E5 (use_cad=True  only): MLP 입력  8 = 0 + 0 + 0 + 0 + 0 + 8 + 0
- E6 (use_scan=True only): MLP 입력  8 = 0 + 0 + 0 + 0 + 0 + 0 + 8
- E7 (use_sensor=True only): MLP 입력 28 = 0 + 0 + 0 + 28 + 0 + 0 + 0

forward 시그니처는 base 와 동일 (8-arg) — 미사용 입력은 단순 무시.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import config
from ..lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4.model import (
    FrameCNN,                # noqa: F401  (재사용 표시용 — 다른 모듈이 import 할 수 있음)
    _GroupLSTMBranch,
    _LSTMBranch,
    _PerFieldConv1DBranch,
)


class VPPM_LSTM_FullStack_Ablation(nn.Module):
    """풀-스택 LSTM 의 7-flag ablation 모델.

    base 의 7-분기 모델 (`VPPM_LSTM_Dual_Img_16_DSCNN_8_CAD_8_Scan_8_1DCNN_Sensor_4`) 와
    동일하나 7 개 flag (`use_static`, `use_v0`, `use_v1`, `use_sensor`, `use_dscnn`,
    `use_cad`, `use_scan`) 로 각 분기를 비활성화할 수 있다.

    flag 가 False 인 분기는 None 으로 설정되어 메모리를 차지하지 않으며,
    forward 시 해당 입력 텐서는 단순 무시된다.

    모든 flag 가 False 인 경우 fc1 입력이 0 이 되어 __init__ 에서 AssertionError 를 일으킨다.

    forward 시그니처는 base 와 동일 (8-arg) — 미사용 입력은 단순 무시.
    """

    def __init__(
        self,
        *,
        use_static: bool = True,
        use_v0: bool = True,
        use_v1: bool = True,
        use_sensor: bool = True,
        use_dscnn: bool = True,
        use_cad: bool = True,
        use_scan: bool = True,
        # 카메라
        d_cnn: int = config.LSTM_D_CNN,
        d_hidden_cam: int = config.LSTM_FULL86_D_HIDDEN_CAM,
        d_embed_v0: int = config.LSTM_FULL86_D_EMBED_V0,
        d_embed_v1: int = config.LSTM_FULL86_D_EMBED_V1,
        # Sensor per-field 1D-CNN
        n_sensor_fields: int = config.LSTM_FULL86_N_SENSOR_FIELDS,
        d_per_sensor_field: int = config.LSTM_FULL86_D_PER_SENSOR_FIELD,
        sensor_hidden_ch: int = config.LSTM_FULL86_SENSOR_HIDDEN_CH,
        sensor_kernel: int = config.LSTM_FULL86_SENSOR_KERNEL,
        # DSCNN
        n_dscnn_ch: int = config.LSTM_FULL86_N_DSCNN_CH,
        d_hidden_d: int = config.LSTM_FULL86_D_HIDDEN_D,
        d_embed_d: int = config.LSTM_FULL86_D_EMBED_D,
        # CAD
        n_cad_ch: int = config.LSTM_FULL86_N_CAD_CH,
        d_cnn_c: int = config.LSTM_FULL86_D_CNN_C,
        d_hidden_c: int = config.LSTM_FULL86_D_HIDDEN_C,
        d_embed_c: int = config.LSTM_FULL86_D_EMBED_C,
        # Scan
        n_scan_ch: int = config.LSTM_FULL86_N_SCAN_CH,
        d_cnn_sc: int = config.LSTM_FULL86_D_CNN_SC,
        d_hidden_sc: int = config.LSTM_FULL86_D_HIDDEN_SC,
        d_embed_sc: int = config.LSTM_FULL86_D_EMBED_SC,
        # MLP
        mlp_hidden: tuple[int, int, int] = config.LSTM_FULL86_MLP_HIDDEN,
        dropout: float = config.DROPOUT_RATE,
    ):
        super().__init__()
        self.use_static = use_static
        self.use_v0 = use_v0
        self.use_v1 = use_v1
        self.use_sensor = use_sensor
        self.use_dscnn = use_dscnn
        self.use_cad = use_cad
        self.use_scan = use_scan

        # 카메라 분기 (flag 가 False 면 None)
        self.branch_v0 = (
            _LSTMBranch(
                in_channels=1, d_cnn=d_cnn,
                d_hidden=d_hidden_cam, d_embed=d_embed_v0,
            )
            if use_v0
            else None
        )
        self.branch_v1 = (
            _LSTMBranch(
                in_channels=1, d_cnn=d_cnn,
                d_hidden=d_hidden_cam, d_embed=d_embed_v1,
            )
            if use_v1
            else None
        )

        # Sensor, DSCNN, CAD, Scan 분기 (flag 가 False 면 None)
        self.branch_sensor = (
            _PerFieldConv1DBranch(
                n_sensor_fields, d_per_sensor_field, sensor_hidden_ch, sensor_kernel,
            )
            if use_sensor
            else None
        )
        self.branch_dscnn = (
            _GroupLSTMBranch(n_dscnn_ch, d_hidden_d, d_embed_d)
            if use_dscnn
            else None
        )
        self.branch_cad = (
            _LSTMBranch(
                in_channels=n_cad_ch, d_cnn=d_cnn_c,
                d_hidden=d_hidden_c, d_embed=d_embed_c,
            )
            if use_cad
            else None
        )
        self.branch_scan = (
            _LSTMBranch(
                in_channels=n_scan_ch, d_cnn=d_cnn_sc,
                d_hidden=d_hidden_sc, d_embed=d_embed_sc,
            )
            if use_scan
            else None
        )

        # MLP 입력 차원 자동 계산
        n_static = len(config.LSTM_FULL86_STATIC_IDX)                            # 2
        n_total = (
            (n_static if use_static else 0)
            + (d_embed_v0 if use_v0 else 0)
            + (d_embed_v1 if use_v1 else 0)
            + (n_sensor_fields * d_per_sensor_field if use_sensor else 0)
            + (d_embed_d if use_dscnn else 0)
            + (d_embed_c if use_cad else 0)
            + (d_embed_sc if use_scan else 0)
        )
        assert n_total >= 1, (
            f"모든 flag 가 False — MLP 입력 차원이 0 입니다. "
            f"use_static/v0/v1/sensor/dscnn/cad/scan 중 하나 이상을 True 로 설정하세요."
        )
        self.n_total = n_total

        h1, h2, h3 = mlp_hidden
        self.fc1 = nn.Linear(n_total, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, 1)
        self.dropout = nn.Dropout(dropout)

        self._init_mlp_weights()

    def _init_mlp_weights(self):
        """MLP fc1~fc4 + 활성화된 분기의 proj 를 N(0, 0.1) 초기화 (base 와 동일).

        None 으로 비활성화된 분기 proj 는 skip.
        """
        projs = []
        if self.branch_v0 is not None:
            projs.append(self.branch_v0.proj)
        if self.branch_v1 is not None:
            projs.append(self.branch_v1.proj)
        if self.branch_sensor is not None:
            for proj in self.branch_sensor.proj:
                projs.append(proj)
        if self.branch_dscnn is not None:
            projs.append(self.branch_dscnn.proj)
        if self.branch_cad is not None:
            projs.append(self.branch_cad.proj)
        if self.branch_scan is not None:
            projs.append(self.branch_scan.proj)
        for m in (self.fc1, self.fc2, self.fc3, self.fc4, *projs):
            nn.init.normal_(m.weight, mean=0.0, std=config.WEIGHT_INIT_STD)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        feats_static: torch.Tensor,           # (B, 2)    — use_static=False 면 무시
        stacks_v0: torch.Tensor,              # (B, T, 8, 8) — use_v0=False 면 무시
        stacks_v1: torch.Tensor,              # (B, T, 8, 8) — use_v1=False 면 무시
        sensors: torch.Tensor,                # (B, T, 7)    — use_sensor=False 면 무시
        dscnn: torch.Tensor,                  # (B, T, 8)    — use_dscnn=False 면 무시
        cad_patch: torch.Tensor,              # (B, T, 2, 8, 8) — use_cad=False 면 무시
        scan_patch: torch.Tensor,             # (B, T, 2, 8, 8) — use_scan=False 면 무시
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        embeds = []
        if self.use_static:
            embeds.append(feats_static)
        if self.branch_v0 is not None:
            embeds.append(self.branch_v0(stacks_v0, lengths))
        if self.branch_v1 is not None:
            embeds.append(self.branch_v1(stacks_v1, lengths))
        if self.branch_sensor is not None:
            embeds.append(self.branch_sensor(sensors, lengths))
        if self.branch_dscnn is not None:
            embeds.append(self.branch_dscnn(dscnn, lengths))
        if self.branch_cad is not None:
            embeds.append(self.branch_cad(cad_patch, lengths))
        if self.branch_scan is not None:
            embeds.append(self.branch_scan(scan_patch, lengths))

        x = torch.cat(embeds, dim=1)
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        x = F.relu(self.fc3(x)); x = self.dropout(x)
        return self.fc4(x)
