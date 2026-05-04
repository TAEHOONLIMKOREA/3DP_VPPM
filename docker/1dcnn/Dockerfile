# syntax=docker/dockerfile:1.6
#
# VPPM-1DCNN 전용 이미지 (depthwise 1D CNN z-압축, baseline MLP 재사용).
# 호스트 venv 를 bind mount 하여 사용 (pip install 불필요).
#
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 코드만 복사 (데이터/산출물/venv 는 bind mount). 검증/실행 로직은 compose `command:` 에 인라인.
COPY Sources/ ./Sources/
COPY CLAUDE.md ./

CMD ["/workspace/venv/bin/python", "-m", "Sources.vppm.1dcnn.run", "--help"]
