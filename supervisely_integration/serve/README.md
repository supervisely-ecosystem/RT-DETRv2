<div align="center" markdown>

<img src="https://github.com/user-attachments/assets/af2c8f9c-8de7-4c78-9b4a-13a2627993be"/>  

# Serve RT-DETRv2

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
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

|     Model      |     Dataset     | Input Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | #Params(M) | FPS |                                                           checkpoint                                                           |
|:--------------:|:---------------:|:----------:|:----------------:|:-----------------------------:|:----------:|:---:|:------------------------------------------------------------------------------------------------------------------------------:|
|  rtdetr_r18vd  |      COCO       |    640     |       46.4       |             63.7              |     20     | 217 |    [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth)     |
|  rtdetr_r34vd  |      COCO       |    640     |       48.9       |             66.8              |     31     | 161 |    [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth)     |
| rtdetr_r50vd_m |      COCO       |    640     |       51.3       |             69.5              |     36     | 145 |      [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth)      |
|  rtdetr_r50vd  |      COCO       |    640     |       53.1       |             71.2              |     42     | 108 |       [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth)       |
| rtdetr_r101vd  |      COCO       |    640     |       54.3       |             72.8              |     76     | 74  |      [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth)       |
|  rtdetr_18vd   | COCO+Objects365 |    640     |       49.0       |             66.5              |     20     | 217 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth)  |
|  rtdetr_r50vd  | COCO+Objects365 |    640     |       55.2       |             73.4              |     42     | 108 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth)  |
| rtdetr_r101vd  | COCO+Objects365 |    640     |       56.2       |             74.5              |     76     | 74  | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth) |

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

# Acknowledgment

This app is based on the great work `RT-DETRv2` ([github](https://github.com/lyuwenyu/RT-DETR)). ![GitHub Org's stars](https://img.shields.io/github/stars/lyuwenyu/RT-DETR?style=social)
