FROM supervisely/base-py-sdk:6.73.209
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt install -y nvidia-cuda-toolkit

RUN pip3 install torch==2.0.1 torchvision==0.15.2
RUN pip3 install onnx==1.14.0 onnxruntime==1.15.1
RUN pip3 install pycocotools
RUN pip3 install setuptools
RUN pip3 install scipy
RUN pip3 install imgaug
RUN pip3 install PyYAML



RUN python3 -m pip install supervisely==6.73.203
RUN python3 -m pip install supervisely[model-benchmark]==6.73.203

LABEL python_sdk_version=6.73.209
