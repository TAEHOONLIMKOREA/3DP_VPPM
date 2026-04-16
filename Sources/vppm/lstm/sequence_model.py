"""
Phase L3 — 시퀀스 모델: (B, T, C, Hs, Ws) → (B, d_lstm).
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .encoder import PatchEncoder


class SupervoxelLSTM(nn.Module):
    """CNN+LSTM — 70 레이어 이미지 시퀀스를 임베딩.

    Args:
        in_channels: 입력 채널 수 (raw+dscnn → 9)
        d_cnn:       CNN 출력 차원
        d_lstm:      최종 임베딩 차원 (양방향인 경우 hidden = d_lstm // 2)
        bidirectional: 양방향 여부
        num_layers:  LSTM 스택 수
        pooling:     "last" | "mean"
    """

    def __init__(
        self,
        in_channels: int = 9,
        d_cnn: int = 64,
        d_lstm: int = 16,
        bidirectional: bool = True,
        num_layers: int = 1,
        pooling: str = "last",
    ):
        super().__init__()
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.encoder = PatchEncoder(in_channels=in_channels, d_cnn=d_cnn)

        hidden_per_dir = d_lstm // (2 if bidirectional else 1)
        if bidirectional and d_lstm % 2 != 0:
            raise ValueError("d_lstm must be even when bidirectional=True")
        self.lstm = nn.LSTM(
            input_size=d_cnn,
            hidden_size=hidden_per_dir,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.d_out = d_lstm

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img:  (B, T, C, H, W) float
            mask: (B, T)          bool, True = 유효 레이어
        Returns:
            emb:  (B, d_lstm)
        """
        B, T, C, H, W = img.shape
        flat = img.reshape(B * T, C, H, W)
        feats = self.encoder(flat).view(B, T, -1)           # (B, T, d_cnn)

        lengths = mask.sum(dim=1).clamp(min=1).to("cpu").long()
        packed = pack_padded_sequence(
            feats, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, (h_n, _) = self.lstm(packed)

        if self.pooling == "last":
            if self.bidirectional:
                emb = torch.cat([h_n[-2], h_n[-1]], dim=1)      # (B, d_lstm)
            else:
                emb = h_n[-1]
            return emb

        # mean pooling over valid timesteps
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        mask_f = mask.unsqueeze(-1).float()                      # (B, T, 1)
        summed = (out * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1e-6)
        return summed / denom
