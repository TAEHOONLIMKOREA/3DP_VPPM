# syntax=docker/dockerfile:1.6
#
# Sample-LSTM (v2) 공용 이미지.
# 호스트 venv (torch 2.11.0 + cuda) 를 bind mount 하여 사용 → pip install 불필요.
# 4 phase (cache / train / extract / vppm) 가 같은 이미지를 공유.
#
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 코드만 복사 (데이터/산출물/venv 는 bind mount)
COPY Sources/ ./Sources/
COPY CLAUDE.md ./

COPY docker/lstm/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/workspace/venv/bin/python", "-m", "Sources.vppm.lstm.train", "--help"]
