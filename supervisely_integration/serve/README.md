<div align="center" markdown>

<img src="https://github.com/user-attachments/assets/af2c8f9c-8de7-4c78-9b4a-13a2627993be"/>

# Serve RT-DETRv2

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#how-to-run">How To Run</a> •
  <a href="#how-to-use-your-checkpoints-outside-supervisely-platform">How to use checkpoints outside Supervisely Platform</a> •
  <a href="#acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/rt-detrv2/supervisely_integration/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/RT-DETRv2)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/rt-detrv2/supervisely_integration/serve.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/rt-detrv2/supervisely_integration/serve.png)](https://supervisely.com)

</div>

# Overview

Serve RT-DETRv2 model as a Supervisely Application. RT-DETRv2 (Real-Time DEtection TRansformer) is fast and accurate object detection model that combines the advantages of DETR and YOLO. RT-DETRv2 builds upon the previous state-of-the-art real-time detector, RT-DETR, and opens up a set of bag-of-freebies for flexibility and practicality, as well as optimizing the training strategy to achieve enhanced performance.

Learn more about RT-DETR and available models [here](https://github.com/lyuwenyu/RT-DETR).

# Model Zoo

|     Model      |     Dataset     | Input Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | #Params(M) | FPS |                                                           checkpoint                                                            |
| :------------: | :-------------: | :--------: | :--------------: | :---------------------------: | :--------: | :-: | :-----------------------------------------------------------------------------------------------------------------------------: |
|  rtdetr_r18vd  |      COCO       |    640     |       46.4       |             63.7              |     20     | 217 |    [url<sup>\*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth)     |
|  rtdetr_r34vd  |      COCO       |    640     |       48.9       |             66.8              |     31     | 161 |    [url<sup>\*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth)     |
| rtdetr_r50vd_m |      COCO       |    640     |       51.3       |             69.5              |     36     | 145 |      [url<sup>\*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth)      |
|  rtdetr_r50vd  |      COCO       |    640     |       53.1       |             71.2              |     42     | 108 |       [url<sup>\*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth)       |
| rtdetr_r101vd  |      COCO       |    640     |       54.3       |             72.8              |     76     | 74  |      [url<sup>\*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth)       |
|  rtdetr_18vd   | COCO+Objects365 |    640     |       49.0       |             66.5              |     20     | 217 | [url<sup>\*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth)  |
|  rtdetr_r50vd  | COCO+Objects365 |    640     |       55.2       |             73.4              |     42     | 108 | [url<sup>\*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth)  |
| rtdetr_r101vd  | COCO+Objects365 |    640     |       56.2       |             74.5              |     76     | 74  | [url<sup>\*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth) |

# How to Run

0. Start the application from an app's context menu or the Ecosystem.

1. Select pre-trained model or custom model trained inside Supervisely platform, and a runtime for inference.

<img src="https://github.com/user-attachments/assets/3f95f85c-e02e-4753-98a2-54fbd0a55a25" />

2. Select device and press the `Serve` button, then wait for the model to deploy.

<img src="https://github.com/user-attachments/assets/9ba08388-d3a7-4716-ae7a-dde86df0db4c" />

3. You will see a message once the model has been successfully deployed.

<img src="https://github.com/user-attachments/assets/c1afc3ab-849d-4cf9-a4b6-6268664c7558" />

4. You can now use the model for inference and see model info.

<img src="https://github.com/user-attachments/assets/2c44915e-e1dd-431f-b85c-07ff3bff3df9" />

# How to use your checkpoints outside Supervisely Platform

After you've trained a model in Supervisely, you can download the checkpoint from Team Files and use it as a simple PyTorch model without Supervisely Platform.

**Quick start:**

1. **Set up environment**. Install [requirements](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/rtdetrv2_pytorch/requirements.txt) manually, or use our pre-built docker image [DockerHub](https://hub.docker.com/r/supervisely/rt-detrv2/tags). Clone [RT-DETRv2](https://github.com/supervisely-ecosystem/RT-DETRv2) repository with model implementation.
2. **Download** your checkpoint and model files from Supervisely Platform.
3. **Run inference**. Refer to our demo scripts: [demo_pytorch.py](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_pytorch.py), [demo_onnx.py](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_onnx.py), [demo_tensorrt.py](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_tensorrt.py)

## Step-by-step guide:

### 1. Set up environment

**Manual installation:**

```bash
git clone https://github.com/supervisely-ecosystem/RT-DETRv2
cd RT-DETRv2
pip install -r rtdetrv2_pytorch/requirements.txt
```

**Using docker image (advanced):**

We provide a pre-built docker image with all dependencies installed [DockerHub](https://hub.docker.com/r/supervisely/rt-detrv2/tags). The image includes installed packages for ONNXRuntime and TensorRT inference.

```bash
docker pull supervisely/rt-detrv2:1.0.38-deploy
```

See our [Dockerfile](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/docker/Dockerfile) for more details.

Docker image already includes the source code.

### 2. Download checkpoint and model files from Supervisely Platform

For RT-DETRv2, you need to download the following files:

**For PyTorch inference:**

- `checkpoint.pth` - model weights, for example `best.pth`
- `model_config.yml` - model configuration
- `model_meta.json` - class names

**ONNXRuntime and TensorRT inference require only \*.onnx and \*.engine files respectively.**

- Exported ONNX/TensorRT models can be found in the `export` folder in Team Files after training.

Go to Team Files in Supervisely Platform and download the files.

Files for PyTorch inference:

![team_files_download](https://github.com/user-attachments/assets/796bf915-fbaf-4e93-a327-f0caa51dced4)

### 3. Run inference

We provide several demo scripts to run inference with your checkpoint:

- [demo_pytorch.py](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_pytorch.py) - simple PyTorch inference
- [demo_onnx.py](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_onnx.py) - ONNXRuntime inference
- [demo_tensorrt.py](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_tensorrt.py) - TensorRT inference

# Acknowledgment

This app is based on the great work `RT-DETRv2` ([github](https://github.com/lyuwenyu/RT-DETR)). ![GitHub Org's stars](https://img.shields.io/github/stars/lyuwenyu/RT-DETR?style=social)
