<div align="center" markdown>

<img src="https://github.com/user-attachments/assets/b4554de5-3d2c-4b4f-aba9-95864cb05289"/>  

# Train RT-DETRv2

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Obtain-saved-checkpoints">Obtain saved checkpoints</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/rt-detrv2/supervisely_integration/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/RT-DETRv2)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/rt-detrv2/supervisely_integration/train.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/rt-detrv2/supervisely_integration/train.png)](https://supervisely.com)

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

# Acknowledgment

This app is based on the great work `RT-DETR` ([github](https://github.com/lyuwenyu/RT-DETR)). ![GitHub Org's stars](https://img.shields.io/github/stars/lyuwenyu/RT-DETR?style=social)
