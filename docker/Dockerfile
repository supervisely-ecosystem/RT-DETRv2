FROM supervisely/base-py-sdk:6.73.210
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y curl
RUN apt install -y nvidia-cuda-toolkit

RUN pip3 install torch==2.0.1 torchvision==0.15.2
RUN pip3 install onnx==1.14.0 onnxruntime==1.15.1
RUN pip3 install pycocotools
RUN pip3 install setuptools
RUN pip3 install scipy
RUN pip3 install imgaug
RUN pip3 install PyYAML

RUN mkdir -p /models
RUN curl -L -o /models/rtdetr_r18vd_coco.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth
RUN curl -L -o /models/rtdetr_r34vd_coco.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth
RUN curl -L -o /models/rtdetr_r50vd_m_coco.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth
RUN curl -L -o /models/rtdetr_r50vd_coco.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
RUN curl -L -o /models/rtdetr_r101vd_coco.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth
RUN curl -L -o /models/rtdetr_r18vd_coco_objects365.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth
RUN curl -L -o /models/rtdetr_r50vd_coco_objects365.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth
RUN curl -L -o /models/rtdetr_r101vd_coco_objects365.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth

RUN python3 -m pip install supervisely==6.73.210
RUN python3 -m pip install supervisely[model-benchmark]==6.73.210

LABEL python_sdk_version=6.73.210
