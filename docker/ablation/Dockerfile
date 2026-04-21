# syntax=docker/dockerfile:1.6
#
# VPPM Feature Ablation 공용 이미지.
# 호스트 venv (torch 2.11.0 + cuda) 를 bind mount 하여 사용 → pip install 불필요.
# 4개 그룹(dscnn/sensor/cad/scan) compose 가 같은 이미지를 공유.
#
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# venv python 이 /usr/bin/python3 심볼릭을 참조하므로 최소한 python3 만 설치
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 코드만 복사 (데이터/산출물/venv 는 bind mount)
COPY Sources/ ./Sources/
COPY CLAUDE.md ./

COPY docker/ablation/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# 기본 명령: usage 출력 (각 group compose 가 --experiment E* 로 override)
CMD ["/workspace/venv/bin/python", "-m", "Sources.vppm.ablation.run", "--help"]
