FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/src/tensorrt/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libmagic-dev \
    libexiv2-dev

# TensorRT Installation
RUN TRT_VERSION="10.9.0.34-1+cuda12.8" && apt-get update && apt-get install -y \
    libnvinfer-dev=${TRT_VERSION} \
    libnvinfer-dispatch-dev=${TRT_VERSION} \
    libnvinfer-dispatch10=${TRT_VERSION} \
    libnvinfer-headers-dev=${TRT_VERSION} \
    libnvinfer-headers-plugin-dev=${TRT_VERSION} \
    libnvinfer-lean-dev=${TRT_VERSION} \
    libnvinfer-lean10=${TRT_VERSION} \
    libnvinfer-plugin-dev=${TRT_VERSION} \
    libnvinfer-plugin10=${TRT_VERSION} \
    libnvinfer-vc-plugin-dev=${TRT_VERSION} \
    libnvinfer-vc-plugin10=${TRT_VERSION} \
    libnvinfer10=${TRT_VERSION} \
    libnvonnxparsers-dev=${TRT_VERSION} \
    libnvonnxparsers10=${TRT_VERSION} \
    tensorrt-dev=${TRT_VERSION} \
    libnvinfer-bin=${TRT_VERSION}
# --------------------- #

RUN ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RT-DETRv2 models
RUN mkdir -p ~/.cache/supervisely/checkpoints && \
    curl -L -o ~/.cache/supervisely/checkpoints/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/rtdetrv2_r34vd_120e_coco_ema.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/rtdetrv2_r50vd_m_7x_coco_ema.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/rtdetrv2_r50vd_6x_coco_ema.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/rtdetrv2_r101vd_6x_coco_from_paddle.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth

# Install Dependencies
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir onnx==1.16.1 onnxruntime-gpu==1.19.2
RUN pip install --no-cache-dir tensorrt-cu12==10.9.0.34

RUN pip install --no-cache-dir setuptools==69.5.1 tensorboard imgaug==0.4.0 decord


# Install the SDK from a git ref. SDK_REF (branch or commit) is referenced in the RUN so that
# passing a new value (--build-arg SDK_REF=<sha>) invalidates the Docker cache and re-pulls it.
# RELEASE_VERSION is required for a git-branch install: setup.py get_version() otherwise reads
# git tags / GitHub API, which fail in pip's shallow clone.
ARG SDK_REF=feat/train-resume-upload-tests
RUN RELEASE_VERSION=6.74.16 pip install --no-cache-dir "supervisely[training,tracking,model-benchmark] @ git+https://github.com/supervisely/supervisely.git@${SDK_REF}"

LABEL python_sdk_version="6.74.16"
