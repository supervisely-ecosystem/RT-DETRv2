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

Train RT-DETRv2 models in Supervisely on your custom data. RT-DETRv2, an improved Real-Time DEtection TRansformer (RT-DETR). RT-DETRv2 builds upon the previous state-of-the-art real-time detector, RT-DETR, and opens up a set of bag-of-freebies for flexibility and practicality, as well as optimizing the training strategy to achieve enhanced performance.

|           Model            | Input shape | Dataset |                $AP^{val}$                |             $AP^{val}_{50}$              | Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS) |
|:--------------------------:|:-----------:|:-------:|:----------------------------------------:|:----------------------------------------:|:---------:|:--------:|:---------------------:|
|      **RT-DETRv2-S**       |     640     |  COCO   | **48.1** <font color=green>(+1.6)</font> |                 **65.1**                 |    20     |    60    |          217          |
| **RT-DETRv2-M**<sup>*<sup> |     640     |  COCO   | **49.9** <font color=green>(+1.0)</font> |                 **67.5**                 |    31     |    92    |          161          |
|      **RT-DETRv2-M**       |     640     |  COCO   | **51.9** <font color=green>(+0.6)</font> |                 **69.9**                 |    36     |   100    |          145          |
|      **RT-DETRv2-L**       |     640     |  COCO   | **53.4** <font color=green>(+0.3)</font> |                 **71.6**                 |    42     |   136    |          108          |
|      **RT-DETRv2-X**       |     640     |  COCO   |                   54.3                   | **72.8** <font color=green>(+0.1)</font> |    76     |   259    |          74           |

# How to Run

**Step 1.** Run the app from context menu of the project with annotations or from the Ecosystem

**Step 2.** Choose the pretrained or custom object detection model

<img src="https://github.com/user-attachments/assets/c236ced3-9165-4d5c-a2f0-2fb29edee05c" width="100%" style='padding-top: 10px'>  

**Step 3.** Select the classes you want to train RT-DETR on

<img src="https://github.com/user-attachments/assets/f0b2c84d-e2a8-4314-af4e-5ec7f784ce1f" width="100%" style='padding-top: 10px'>  

**Step 4.** Define the train/val splits

<img src="https://github.com/user-attachments/assets/3a2ac582-0489-493d-b2ff-8a98c94dfa20" width="100%" style='padding-top: 10px'>  

**Step 5.** Choose either ready-to-use augmentation template or provide custom pipeline

<img src="https://github.com/user-attachments/assets/a053fd89-4acc-44c0-af42-1ec0b84804a6" width="100%" style='padding-top: 10px'>  

**Step 6.** Configure the training parameters

<img src="https://github.com/user-attachments/assets/c5c715f0-836d-4613-a004-d139e2cf9706" width="100%" style='padding-top: 10px'>  

**Step 7.** Click `Train` button and observe the training progress, metrics charts and visualizations 

<img src="https://github.com/user-attachments/assets/703e182f-c84e-47de-8dc3-b01da8457580" width="100%" style='padding-top: 10px'>  

# Obtain saved checkpoints

All the trained checkpoints, that are generated through the process of training models are stored in [Team Files](https://app.supervise.ly/files/) in the folder **RT-DETR**.

You will see a folder thumbnail with a link to you saved checkpoints by the end of training process.

<img src="https://github.com/user-attachments/assets/6dd036f4-41de-4eb9-a87a-3387fb849ff1" width="100%" style='padding-top: 10px'>  

# Acknowledgment

This app is based on the great work `RT-DETR` ([github](https://github.com/lyuwenyu/RT-DETR)). ![GitHub Org's stars](https://img.shields.io/github/stars/lyuwenyu/RT-DETR?style=social)
