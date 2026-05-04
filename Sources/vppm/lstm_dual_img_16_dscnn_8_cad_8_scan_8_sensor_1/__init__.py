"""VPPM-LSTM-Dual-Img-16-DSCNN-8-CAD-8-Scan-8-Sensor-1 실험 패키지.

설계: PLAN.md
주요 차이점 (fullstack `_1dcnn_sensor_4` 대비):
  - Sensor: per-field 1D-CNN(필드당 4-dim, 총 28) → 단일 multi-channel LSTM(d_embed_s=1)
  - 다른 6개 분기 (카메라 v0/v1, DSCNN, CAD, Scan, 정적 2-feat) 동일
  - MLP: 86→256→128→64→1 → 59→256→128→64→1 (fc1 입력만 86→59)
  - 모든 캐시 재사용 — 신규 캐시 빌드 없음
"""
