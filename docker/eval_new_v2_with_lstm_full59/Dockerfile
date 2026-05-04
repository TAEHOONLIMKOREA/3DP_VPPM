# syntax=docker/dockerfile:1.6
#
# eval_new_v2_with_lstm_full59 이미지.
# 학습된 LSTM_FULL59 (vppm_lstm_dual_img_16_dscnn_8_cad_8_scan_8_sensor_1) 을
# 신규 build (2023-03-15 AMMTO Spatial Variation Baseline) 에 적용해 part-level 평가.
# 호스트 venv (torch + cuda) 를 bind mount 하여 사용 — pip install 불필요.
#
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# venv python 이 /usr/bin/python3 심볼릭을 참조하므로 최소한 python3 만 설치
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 코드만 복사 (데이터/산출물/venv 는 bind mount). 검증/실행 로직은 compose `command:` 에 인라인.
COPY Sources/ ./Sources/
COPY CLAUDE.md ./

# 기본 명령: usage 출력 (compose command: 가 override)
CMD ["/workspace/venv/bin/python", "-m", "Sources.vppm.eval_new_v2_with_lstm_full59.run", "--help"]
