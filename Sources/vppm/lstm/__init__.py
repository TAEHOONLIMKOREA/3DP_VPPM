"""Sample-LSTM 모듈 — PLAN_LSTM_v2.md 기반.

샘플 단위 raw camera 시퀀스를 LSTM 으로 임베딩해 21 차원 핸드크래프트 피처에 추가한다.

파이프라인:
    Phase L1 — sample_stack.py: HDF5 → 캐시 (.h5)
    Phase L2 — train.py:        5-Fold CV LSTM 학습
    Phase L3 — extract.py:      임베딩 추출 + 21→37 차원 npz 생성
    Phase L4 — run_vppm.py:     37 차원 VPPM 재학습 (origin/ 재사용)
"""
