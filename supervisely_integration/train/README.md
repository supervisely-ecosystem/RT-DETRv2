<div align="center" markdown>

<img src="https://github.com/user-attachments/assets/b4554de5-3d2c-4b4f-aba9-95864cb05289"/>

# Train RT-DETRv2

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#how-to-run">How To Run</a> •
  <a href="#obtain-saved-checkpoints">Obtain saved checkpoints</a> •
  <a href="#how-to-use-your-checkpoints-outside-supervisely-platform">How to use checkpoints outside Supervisely Platform</a> •
  <a href="#acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/rt-detrv2/supervisely_integration/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/RT-DETRv2)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/rt-detrv2/supervisely_integration/train.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/rt-detrv2/supervisely_integration/train.png)](https://supervisely.com)

</div>

# Overview

Train RT-DETRv2 models in Supervisely on your custom data. RT-DETRv2 (Real-Time DEtection TRansformer) is fast and accurate object detection model that combines the advantages of DETR and YOLO. RT-DETRv2 builds upon the previous state-of-the-art real-time detector, RT-DETR, and opens up a set of bag-of-freebies for flexibility and practicality, as well as optimizing the training strategy to achieve enhanced performance.

|            Model            | Input shape | Dataset |                $AP^{val}$                |             $AP^{val}_{50}$              | Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS) |
| :-------------------------: | :---------: | :-----: | :--------------------------------------: | :--------------------------------------: | :-------: | :------: | :-------------------: |
|       **RT-DETRv2-S**       |     640     |  COCO   | **48.1** <font color=green>(+1.6)</font> |                 **65.1**                 |    20     |    60    |          217          |
| **RT-DETRv2-M**<sup>\*<sup> |     640     |  COCO   | **49.9** <font color=green>(+1.0)</font> |                 **67.5**                 |    31     |    92    |          161          |
|       **RT-DETRv2-M**       |     640     |  COCO   | **51.9** <font color=green>(+0.6)</font> |                 **69.9**                 |    36     |   100    |          145          |
|       **RT-DETRv2-L**       |     640     |  COCO   | **53.4** <font color=green>(+0.3)</font> |                 **71.6**                 |    42     |   136    |          108          |
|       **RT-DETRv2-X**       |     640     |  COCO   |                   54.3                   | **72.8** <font color=green>(+0.1)</font> |    76     |   259    |          74           |

# How to Run

**Step 0.** Run the app from context menu of the project with annotations or from the Ecosystem

**Step 1.** Select if you want to use cached project or redownload it

<img src="https://github.com/user-attachments/assets/8c2b5f4b-1abd-44f5-8f0e-d2890284d8ba" width="100%" style='padding-top: 10px'>

**Step 2.** Select train / val split

<img src="https://github.com/user-attachments/assets/aec211b8-6eee-46a8-a70f-04bbf6ed4fa5" width="100%" style='padding-top: 10px'>

**Step 3.** Select the classes you want to train RT-DETRv2 on

<img src="https://github.com/user-attachments/assets/55068763-268a-437e-8492-0368fba4fb13" width="100%" style='padding-top: 10px'>

**Step 4.** Select the model you want to train

<img src="https://github.com/user-attachments/assets/753f6f0f-a38f-4d61-accb-dc8ba41777c0" width="100%" style='padding-top: 10px'>

**Step 5.** Configure hyperaparameters and select whether you want to use model evaluation and convert checkpoints to ONNX and TensorRT

<img src="https://github.com/user-attachments/assets/59da4672-2f82-480f-b2c5-fb118878ff29" width="100%" style='padding-top: 10px'>

**Step 6.** Enter experiment name and start training

<img src="https://github.com/user-attachments/assets/252b085f-4597-494e-afaa-ae0a19f0e7ba" width="100%" style='padding-top: 10px'>

**Step 7.** Monitor training progress

<img src="https://github.com/user-attachments/assets/0b157c66-b3a0-4bd2-ba93-c3c8e8ff72bb" width="100%" style='padding-top: 10px'>

# Obtain saved checkpoints

All the trained checkpoints, that are generated through the process of training models are stored in [Team Files](https://app.supervisely.com/files/) in the **experiments** folder.

You will see a folder thumbnail with a link to your saved checkpoints by the end of training process.

<img src="https://github.com/user-attachments/assets/c075107c-6493-401e-8276-14cdf6f61be6" width="100%" style='padding-top: 10px'>

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

This app is based on the great work `RT-DETR` ([github](https://github.com/lyuwenyu/RT-DETR)). ![GitHub Org's stars](https://img.shields.io/github/stars/lyuwenyu/RT-DETR?style=social)
