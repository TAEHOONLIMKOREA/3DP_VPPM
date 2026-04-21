# syntax=docker/dockerfile:1.6
#
# VPPM-LSTM 학습용 이미지 (경량 버전).
# 호스트의 venv 를 bind mount 하여 torch 2.11.0 + sm_120 을 그대로 사용.
# apt-get / pip install 없이 코드 + entrypoint 만 포함 → 빌드 수 초.
#
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LSTM_CACHE_DIR=/tmp/image_stacks

# venv 의 python 심볼릭 링크가 /usr/bin/python3 을 가리키므로 최소한 python3 만 설치
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 코드만 복사 (데이터/산출물/venv 은 bind mount)
COPY Sources/ ./Sources/
COPY CLAUDE.md IMPLEMENTATION_PLAN_LSTM.md ./

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/workspace/venv/bin/python", "-m", "Sources.vppm.run_pipeline", "--use-lstm", "--phase", "all"]
