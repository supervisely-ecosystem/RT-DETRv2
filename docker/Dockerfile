FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/src/tensorrt/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libmagic-dev \
    libexiv2-dev \
    python3-dev

RUN ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir onnx==1.16.1 onnxruntime-gpu==1.19.2

RUN pip install --no-cache-dir setuptools==69.5.1 tensorboard imgaug==0.4.0 decord

RUN mkdir -p ~/.cache/supervisely/checkpoints && \
    curl -L -o ~/.cache/supervisely/checkpoints/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/rtdetrv2_r34vd_120e_coco_ema.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/rtdetrv2_r50vd_m_7x_coco_ema.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/rtdetrv2_r50vd_6x_coco_ema.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/rtdetrv2_r101vd_6x_coco_from_paddle.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth

RUN pip install --no-cache-dir supervisely[training]==6.73.424
RUN pip install --no-cache-dir supervisely[model-benchmark]==6.73.424

LABEL python_sdk_version=6.73.424