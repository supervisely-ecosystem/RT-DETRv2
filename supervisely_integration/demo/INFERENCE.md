# Local Model Inference: Overview

**Table of Contents:**

- [1. Standalone PyTorch model](#1-standalone-pytorch-model)
- [2. Load Model in your Code](#2-load-model-in-your-code)
- [3. Deploy as a Local Inference Server (no UI)](#3-deploy-as-a-local-inference-server-no-ui)
  - [🐳 Deploy in Docker Container](#-deploy-in-docker-container)
- [4. Deploy as a Local Serving App (with UI)](#4-deploy-as-a-local-serving-app-with-ui)


## 1. Standalone PyTorch model

After you've trained a model in Supervisely, you can download the checkpoint from Team Files and use it as an original PyTorch model.

**Quick start** *(RT-DETRv2 example)*:

1. **Download** your checkpoint and model files from Team Files.

2. **Clone** our [RT-DETRv2](https://github.com/supervisely-ecosystem/RT-DETRv2) fork with the model implementation. Alternatively, you can use the original [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR/tree/0b6972de10bc968045aba776ec1a60efea476165) repository, but you may face some unexpected issues if the authors have updated the code.

```bash
git clone https://github.com/supervisely-ecosystem/RT-DETRv2
cd RT-DETRv2
```

3. **Set up environment:** Install [requirements](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/rtdetrv2_pytorch/requirements.txt) manually, or use our pre-built docker image ([DockerHub](https://hub.docker.com/r/supervisely/rt-detrv2/tags) | [Dockerfile](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/docker/Dockerfile)).
```bash
pip install -r rtdetrv2_pytorch/requirements.txt
```

4. **Run inference:** Refer to our example scripts of how to load RT-DETRv2 and get predictions:

- [demo_pytorch.py](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_pytorch.py)
- [demo_onnx.py](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_onnx.py)
- [demo_tensorrt.py](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_tensorrt.py)


## 2. Load Model in your Code

❌ use of private method

This method allows you to easily load your model and run inference in your code on a local machine with the help of Supervisely SDK. It provides more features and flexibility than the standalone PyTorch model method. For example, you don't need to worry about the model's input and output format, as the our model wrapper will handle it for you.

**Quick start** *(RT-DETRv2 example)*:

1. **Download** your checkpoint and model files from Team Files.

2. **Clone** our [RT-DETRv2](https://github.com/supervisely-ecosystem/RT-DETRv2) fork with the model implementation.

```bash
git clone https://github.com/supervisely-ecosystem/RT-DETRv2
cd RT-DETRv2
```

3. **Set up environment:** Install [requirements](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/rtdetrv2_pytorch/requirements.txt) manually, or use our pre-built docker image ([DockerHub](https://hub.docker.com/r/supervisely/rt-detrv2/tags) | [Dockerfile](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/docker/Dockerfile)). Additionally, you need to install Supervisely SDK.
```bash
pip install -r rtdetrv2_pytorch/requirements.txt
pip install supervisely
```

4. **Predict:** This code snippet demonstrates how to load RT-DETRv2 and get predictions.

```python
import numpy as np
from supervisely_integration.serve.rtdetrv2 import RTDETRv2
from supervisely.nn import ModelSource, RuntimeType
from PIL import Image
import os

# Load model
model = RTDETRv2()
model_info = model.pretrained_models[0]
# ❌ use of private method
model._load_model_headless(
    model_files={
        "config": "rtdetrv2_r18vd_120e_coco.yml",
        "checkpoint": os.path.expanduser("~/.cache/supervisely/checkpoints/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"),
    },
    model_info=model_info,
    model_source=ModelSource.PRETRAINED,
    device="cuda",
    runtime=RuntimeType.PYTORCH,
)

# Load image
image = Image.open("supervisely_integration/serve/scripts/coco_sample.jpg").convert("RGB")
img = np.array(image)

# Inference
# ❌ use of private method
ann = model._inference_auto([img], {"confidence_threshold": 0.5})[0][0]

# Draw predictions
ann.draw_pretty(img)
Image.fromarray(img).save("supervisely_integration/serve/scripts/predict.jpg")
```


## 3. Deploy as a Local Inference Server (no UI)

❌ do not need sly.Api for local deployment

You can run your checkpoints trained in Supervisely locally.

**Quick start** *(RT-DETRv2 example)*:

1. **Download** is optional. You can provide remote path to the custom checkpoint located in Team Files or download checkpoint and model files and place it to the local directory.

2. **Clone** our [RT-DETRv2](https://github.com/supervisely-ecosystem/RT-DETRv2) fork with the model implementation.

```bash
git clone https://github.com/supervisely-ecosystem/RT-DETRv2
cd RT-DETRv2
```

3. **Set up environment:**

For **Local Inference Server**, you need to install the necessary dependencies and tools to run the server.

Install [requirements](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/rtdetrv2_pytorch/requirements.txt) manually, or use our pre-built docker image ([DockerHub](https://hub.docker.com/r/supervisely/rt-detrv2/tags) | [Dockerfile](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/docker/Dockerfile)).

```bash
pip install -r rtdetrv2_pytorch/requirements.txt
pip install supervisely
```

#### 4. **Deploy:**

**Local Inference Server** deployment command:

```bash
PYTHONPATH="${PWD}:${PYTHONPATH}" python ./supervisely_integration/serve/main.py --model ./my_experiments/2315_RT-DETRv2/checkpoints/best.pth
```

You need to pass `--model` argument with the path to the custom checkpoint file or the name of the pretrained model to run the server.
For custom checkpoints path can be local or remote (Team Files).
Additionally, you can pass predict arguments to specify the project, dataset, image, or directory to predict, server will automatically shutdown after prediction.
If no predict arguments are provided, the server will start and wait for the prediction requests via Supervisely Inference API.

#### 5. **Predict**

**Predict with Session API:**

You can use Supervisely [Inference Session API](https://developer.supervisely.com/app-development/neural-network-integration/inference-api-tutorial) with setting server address to `http://0.0.0.0:8000` to make inference in docker container.

```python
import os
from dotenv import load_dotenv
import supervisely as sly

load_dotenv(os.path.expanduser("~/supervisely.env"))
# ❌ (do not need sly.Api for local deployment)
api = sly.Api()

# Create Inference Session
session = sly.nn.inference.Session(api, session_url="http://0.0.0.0:8000")

# local image
pred = session.inference_image_path("image_01.jpg")

# batch of images
pred = session.inference_image_paths(["image_01.jpg", "image_02.jpg"])

# remote image on the platform
pred = session.inference_image_id(17551748)
pred = session.inference_image_ids([17551748, 17551750])

# image url
url = "https://images.unsplash.com/photo-1674552791148-c756b0899dba?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80"
pred = session.inference_image_url(url)
```

**Predict with args:**

Available Arguments:

- `--model` - **(required)** name of the pretrained model or path to custom checkpoint file.
- `--predict-project` ID of the project to predict. New project will be created.
- `--predict-dataset` ID(s) of the dataset(s) to predict. New project will be created
- `--predict-image` - Image ID or path to local image.
- `--predict-dir` - [Not implemented yet] path to the local directory with images to predict

```bash
PYTHONPATH="${PWD}:${PYTHONPATH}" python ./supervisely_integration/serve/main.py --model ./my_experiments/2315_RT-DETRv2/checkpoints/best.pth --predict-image ./supervisely_integration/demo/images/image.jpg
```

### 🐳 Deploy in Docker Container

Inference in Docker Container is similar to local inference, except that it runs in a docker container. This method is useful when you need to run your model on a remote server or in a cloud environment.

For **Docker Container**, you need to pull the pre-built docker image from DockerHub.

```bash
docker pull supervisely/rt-detrv2-gpu-cloud:1.0.3
```
**Docker Container** deployment command:

```bash
docker run \
  --shm-size=1g \
  --runtime=nvidia \
  --cap-add NET_ADMIN \
  --env-file ~/supervisely.env \
  --env ENV=production \
  -v ".:/app" \
  -w /app \
  -p 8000:8000 \
  supervisely/rt-detrv2-gpu-cloud:1.0.3 \
  --model "/experiments/553_42201_Animals/2315_RT-DETRv2/checkpoints/best.pth"
```


## 4. Deploy as a Local Serving App (with UI)

❌ do not need sly.Api for local deployment

You can deploy your model locally as an API Inference Server with the help of Supervisely SDK. It allows you to run inference on your local machine for both local images or videos, and remote supervisely projects and datasets.

**Quick start** *(RT-DETRv2 example)*:

1. **Download** your checkpoint and model files from Team Files.

2. **Clone** our [RT-DETRv2](https://github.com/supervisely-ecosystem/RT-DETRv2) fork with the model implementation.
```bash
git clone https://github.com/supervisely-ecosystem/RT-DETRv2
cd RT-DETRv2
```

3. **Set up environment:** Install [requirements](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/rtdetrv2_pytorch/requirements.txt) manually, or use our pre-built docker image ([DockerHub](https://hub.docker.com/r/supervisely/rt-detrv2/tags) | [Dockerfile](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/docker/Dockerfile)).
```bash
pip install -r rtdetrv2_pytorch/requirements.txt
pip install supervisely
```

4. **Deploy:** To start the inference server, run the following command:

```bash
uvicorn main:model.app --app-dir supervisely_integration/serve --host 0.0.0.0 --port 8000 --ws websockets
```

5. **Predict:** You can use Supervisely [Inference Session API](https://developer.supervisely.com/app-development/neural-network-integration/inference-api-tutorial) with setting server address to `http://localhost:8000` to make inference on your local machine.

```python
import os
from dotenv import load_dotenv
import supervisely as sly

load_dotenv(os.path.expanduser("~/supervisely.env"))
# ❌ (do not need sly.Api for local deployment)
api = sly.Api()

# Create Inference Session
session = sly.nn.inference.Session(api, session_url="http://localhost:8000")

# local image
pred = session.inference_image_path("image_01.jpg")

# batch of images
pred = session.inference_image_paths(["image_01.jpg", "image_02.jpg"])

# remote image on the platform
pred = session.inference_image_id(17551748)
pred = session.inference_image_ids([17551748, 17551750])

# image url
url = "https://images.unsplash.com/photo-1674552791148-c756b0899dba?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80"
pred = session.inference_image_url(url)
```
