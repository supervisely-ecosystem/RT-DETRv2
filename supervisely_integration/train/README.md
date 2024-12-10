<div align="center" markdown>

<img src="https://github.com/user-attachments/assets/ae6ec55e-63eb-43d1-99de-f62650939c69"/>  

# Train RT-DETRv2

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Obtain-saved-checkpoints">Obtain saved checkpoints</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/rt-detr/supervisely_integration/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/RT-DETR)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/rt-detr/supervisely_integration/train.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/rt-detr/supervisely_integration/train.png)](https://supervise.ly)

</div>

# Overview

Train RT-DETRv2 models in Supervisely on your custom data. RT-DETRv2 (Real-Time DEtection TRansformer) is fast and accurate object detection model that combines the advantages of DETR and YOLO. RT-DETRv2 builds upon the previous state-of-the-art real-time detector, RT-DETR, and opens up a set of bag-of-freebies for flexibility and practicality, as well as optimizing the training strategy to achieve enhanced performance.

|           Model            | Input shape | Dataset |                $AP^{val}$                |             $AP^{val}_{50}$              | Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS) |
|:--------------------------:|:-----------:|:-------:|:----------------------------------------:|:----------------------------------------:|:---------:|:--------:|:---------------------:|
|      **RT-DETRv2-S**       |     640     |  COCO   | **48.1** <font color=green>(+1.6)</font> |                 **65.1**                 |    20     |    60    |          217          |
| **RT-DETRv2-M**<sup>*<sup> |     640     |  COCO   | **49.9** <font color=green>(+1.0)</font> |                 **67.5**                 |    31     |    92    |          161          |
|      **RT-DETRv2-M**       |     640     |  COCO   | **51.9** <font color=green>(+0.6)</font> |                 **69.9**                 |    36     |   100    |          145          |
|      **RT-DETRv2-L**       |     640     |  COCO   | **53.4** <font color=green>(+0.3)</font> |                 **71.6**                 |    42     |   136    |          108          |
|      **RT-DETRv2-X**       |     640     |  COCO   |                   54.3                   | **72.8** <font color=green>(+0.1)</font> |    76     |   259    |          74           |

# How to Run

**Step 0.** Run the app from context menu of the project with annotations or from the Ecosystem

**Step 1.** Select if you want to use cached project or redownload it

<img src="https://github.com/user-attachments/assets/5d992899-e82f-449a-8817-8a261f033403" width="100%" style='padding-top: 10px'>  

**Step 2.** Select train / val split

<img src="https://github.com/user-attachments/assets/71a682a6-b59a-4338-9697-b37d735a5f58" width="100%" style='padding-top: 10px'>  

**Step 3.** Select the classes you want to train RT-DETRv2 on

<img src="https://github.com/user-attachments/assets/0546e425-453e-4e1d-b47d-892e68da04e5" width="100%" style='padding-top: 10px'>  

**Step 4.** Select the model you want to train

<img src="https://github.com/user-attachments/assets/d58dbe20-ba53-4525-9407-c037fff90655" width="100%" style='padding-top: 10px'>  

**Step 5.** Configure hyperaparameters and select whether you want to use model evaluation and convert checkpoints to ONNX and TensorRT

<img src="https://github.com/user-attachments/assets/d82161c2-2ba5-42af-b651-e4bfc55aad3a" width="100%" style='padding-top: 10px'>  

**Step 6.** Enter experiment name and start training

<img src="https://github.com/user-attachments/assets/6eac9d70-bfd9-4105-949e-2394a13395a2" width="100%" style='padding-top: 10px'>  

**Step 7.** Monitor training progress

<img src="https://github.com/user-attachments/assets/d82216f6-0983-4717-8140-69b589d1b0fa" width="100%" style='padding-top: 10px'>  

# Obtain saved checkpoints

All the trained checkpoints, that are generated through the process of training models are stored in [Team Files](https://app.supervise.ly/files/) in the **experiments** folder.

You will see a folder thumbnail with a link to your saved checkpoints by the end of training process.

<img src="https://github.com/user-attachments/assets/31aab070-343b-4890-ac4b-0f1ca7efcee2" width="100%" style='padding-top: 10px'>  

# Acknowledgment

This app is based on the great work `RT-DETR` ([github](https://github.com/lyuwenyu/RT-DETR)). ![GitHub Org's stars](https://img.shields.io/github/stars/lyuwenyu/RT-DETR?style=social)
