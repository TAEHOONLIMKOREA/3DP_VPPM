# syntax=docker/dockerfile:1.6
#
# VPPM Hidden-Dim Sweep 전용 이미지.
# 호스트 venv (torch 2.11.0 + cuda) 를 bind mount 하여 사용 → pip install 불필요.
# entry point: Sources.vppm.ablation.run_hidden_sweep
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

# 기본 명령: usage 출력 (compose 가 --hidden / --all 등으로 override)
CMD ["/workspace/venv/bin/python", "-m", "Sources.vppm.ablation.run_hidden_sweep", "--help"]
