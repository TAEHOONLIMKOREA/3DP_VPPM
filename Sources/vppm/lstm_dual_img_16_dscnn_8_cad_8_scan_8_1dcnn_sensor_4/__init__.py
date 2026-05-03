"""VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-1DCNN-Sensor-4 실험 패키지.

설계: PLAN.md
주요 차이점 (dscnn_8 대비):
  - 카메라 v0/v1 d_embed: 4 → 16
  - Sensor: 7-ch LSTM(d=7) → 필드별 1D-CNN(필드당 4-dim, 총 28)
  - CAD (#1, #2): scalar 평균 → spatial-CNN+LSTM, 8×8 패치 + inversion + cad_mask 픽셀곱 (d=8)
  - Scan (#20, #21): scalar 평균 → spatial-CNN+LSTM, 8×8 패치 raw (d=8)
  - MLP: 29→128→1 → 86→256→128→64→1 (3 hidden)
"""
