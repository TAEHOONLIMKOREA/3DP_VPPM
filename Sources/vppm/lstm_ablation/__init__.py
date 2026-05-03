"""LSTM-Full-Stack Ablation 실험 패키지.

설계: PLAN.md (E1: no-v0, E2: no-cameras).
풀-스택 base 모델 (`vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_1dcnn_sensor_4`) 을
import 재사용해 카메라 분기를 토글한 ablation 변형을 학습/평가한다.
"""
