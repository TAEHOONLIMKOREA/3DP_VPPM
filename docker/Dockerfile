# syntax=docker/dockerfile:1.6
#
# VPPM-LSTM 학습용 이미지.
# 호스트 디스크 절약 원칙:
#  - 원본 HDF5, 기존 features/, baseline models/ 는 bind mount 로만 접근 (이미지에 복사 X)
#  - image_stacks 캐시는 컨테이너 내부 /tmp (tmpfs) 에만 존재 → 종료 시 증발
#
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LSTM_CACHE_DIR=/tmp/image_stacks

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10-venv \
        python3-pip \
        libhdf5-dev \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# 시스템 python3 (3.10) 에 직접 설치 — pip 와 python 이 같은 인터프리터
RUN python -m pip install --upgrade pip \
 && python -m pip install \
        numpy==2.2.6 \
        h5py==3.16.0 \
        scipy==1.15.3 \
        scikit-learn==1.7.2 \
        pandas==2.3.3 \
        matplotlib==3.10.8 \
        tqdm==4.67.3 \
 && python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 코드만 복사 (데이터/산출물은 bind mount)
COPY Sources/ ./Sources/
COPY CLAUDE.md IMPLEMENTATION_PLAN_LSTM.md ./

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "Sources.vppm.run_pipeline", "--use-lstm", "--phase", "all"]
