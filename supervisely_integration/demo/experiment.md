# Experiment "835 Mouse Detector"

## Buttons
- 🚀 Deploy (PyTorch)  (открывать модалку с настройками*)
- 🚀 Deploy (TensorRT)  -- Если был экспорт в TRT
- ⚡ Apply model to images/video (открывать модалку с настройками*)
- ⏩ Fine-tune
- 🔄 Re-train
- 📦 Download model
- ❌ Remove permamently

**\* настройки - select project/dataset + additional settings: checkpoint, device, runtime, agent, inference_settings.yaml**

---

## Overview

- 🎓 [Training Task](https://dev.internal.supervisely.com/apps/146/sessions/1089)
- 📊 [Evaluation Report](https://dev.internal.supervisely.com/model-benchmark?id=262839)
- 📈 [TensorBoard Logs](xxx)
- 📂 [Open in Team Files](https://dev.internal.supervisely.com/files/?path=%2Fexperiments%2F835_MP%3A%20Images%20Sample%20for%20Detection%20Task%20%28RTDETR2%20-%20cat%29%20Filtered%20and%20Splitted%2F1089_RT-DETRv2%2F)

---

- **Model**: RT-DETRv2-M
- **Task**: Object Detection
- **Framework**: [Train RT-DETRv2](https://dev.internal.supervisely.com/ecosystem/apps/rt-detr/supervisely_integration/train?id=225)
- **Project**: [MP: Images Sample for Detection Task (RTDETR2)](https://dev.internal.supervisely.com/projects/835/datasets) (6130 images)
- **Train dataset**: [train](Train) (5130 images)
- **Validation dataset**: [val](Val) (1000 images)
- **Classes**: 1
- **Class names**: mouse
- **Train time**: 1h 23m
- **Date**: 21 Feb 2025

| Checkpoints |
|---------|
| best.pt |
| last.pt |
| checkpoint005.pt |
| checkpoint010.pt |
| checkpoint015.pt |
| checkpoint020.pt |

## Predictions

![predictions](img/predictions.png)

## Metrics

| Metrics | Values |
|---------|--------|
| mAP | 0.9421 |
| AP50 | 0.9915 |
| AP75 | 0.9761 |
| f1 | 0.9210 |
| precision | 0.9051 |
| recall | 0.9341 |
| Avg. IoU | 0.9461 |
| Classification Acc. | 1.00 |
| Calibration Score | 0.9318 |
| Optimal confidence threshold | 0.6159 |

## Training Hyperparameters

```yaml
epoches: 100
batch_size: 16
eval_spatial_size: [640, 640]  # height, width

checkpoint_freq: 10
save_optimizer: false
save_ema: false

optimizer:
  type: AdamW
  lr: 0.0001
  betas: [0.9, 0.999]
  weight_decay: 0.0001

clip_max_norm: 1

lr_scheduler:
  type: MultiStepLR  # CosineAnnealingLR | OneCycleLR
  milestones: [80, 95]  # epochs
  gamma: 0.1

lr_warmup_scheduler:
  type: LinearWarmup
  warmup_duration: 300  # steps

use_ema: false
ema:
  type: ModelEMA
  decay: 0.9999
  warmups: 2000

use_amp: false
```

## Training

![chart](img/chart.png)

## Predict via API

Get predictions from your model in a couple lines of code.

🔴🔴🔴 Здесь будут tabs с разными примерами

#### Local Images

```python
import supervisely as sly

api = sly.Api()

# Deploy
model = api.nn.deploy(
    model={"/experiments/9_Animals (Bitmap)/1866_RT-DETRv2/checkpoints/best.pth"},
    device="cuda:0",  # or "cpu"
)

# Predict local images
prediction = model.predict(
    input="image.jpg"  # can also be a directory, np.array, PIL.Image, URL or a list of them
)
```

#### Image ID 🔴🔴🔴 Здесь будут tabs с разными примерами

```python
import supervisely as sly

api = sly.Api()

# Deploy
model = api.nn.deploy(
    model={"/experiments/9_Animals (Bitmap)/1866_RT-DETRv2/checkpoints/best.pth"},
    device="cuda:0",  # or "cpu"
)

# Predict images in Supervisely
prediction = model.predict(
    image_ids=[123, 124]  # Image ids in Supervisely
)
```

#### Dataset 🔴🔴🔴 Здесь будут tabs с разными примерами

```python
import supervisely as sly

api = sly.Api()

# Deploy
model = api.nn.deploy(
    model={"/experiments/9_Animals (Bitmap)/1866_RT-DETRv2/checkpoints/best.pth"},
    device="cuda:0",  # or "cpu"
)

# Predict dataset
prediction = model.predict(
    dataset_id=12,  # Dataset id in Supervisely
)
```

#### Project 🔴🔴🔴 Здесь будут tabs с разными примерами

```python
import supervisely as sly

api = sly.Api()

# Deploy
model = api.nn.deploy(
    model={"/experiments/9_Animals (Bitmap)/1866_RT-DETRv2/checkpoints/best.pth"},
    device="cuda:0",  # or "cpu"
)

# Predict project
prediction = model.predict(
    project_id=21,  # Project id in Supervisely
)
```

#### Video 🔴🔴🔴 Здесь будут tabs с разными примерами

```python
import supervisely as sly

api = sly.Api()

# Deploy
model = api.nn.deploy(
    model={"/experiments/9_Animals (Bitmap)/1866_RT-DETRv2/checkpoints/best.pth"},
    device="cuda:0",  # or "cpu"
)

# Predict video
prediction = model.predict(
    video_id=123,  # Video id in Supervisely
)
```

> For more information, see [Prediction API](https://docs.supervisely.com/neural-networks/overview-1/prediction-api) and [Model API](https://docs.supervisely.com/neural-networks/overview-1/model-api).

## Predict in Docker

You can apply this model in a 🐋 Docker Container with a single `docker run` comand. For this, you need to download a checkpoint, pull the docker image for the corresponding model's framework, and run the `docker run` comand with addtional arguments.

1. Download checkpoint from Supervisely ([best.pt](xxx))

2. Pull the Docker image

```bash
docker pull {supervisely/rt-detrv2:1.0.11}
```
3. Run the Docker container

```bash
docker run \
  --runtime=nvidia \
  -v "./{1089_RT-DETRv2}:/model" \
  -p 8000:8000 \
  {supervisely/rt-detrv2:1.0.11} \
  predict \
  "./image.jpg" \
  --model "/model" \
  --device "cuda:0" \
```

> For more information, see [Deploy in Docker Container](https://docs.supervisely.com/neural-networks/overview-1/deploy_and_predict_with_supervisely_sdk#deploy-in-docker-container) documentation.

## Predict Locally

In the case of local deployment, the model will be deployed outside of Supervisely Platform. This is useful when you're developing applications that are not directly related to the platform, and you can just use the model itself in your code.

Here's how to deploy the model locally:

1. Download checkpoint from Supervisely ([best.pt](xxx))

2. Clone our repository

```bash
git clone https://github.com/supervisely-ecosystem/RT-DETRv2
cd RT-DETRv2
```

3. Install requirements

```bash
pip install -r dev_requirements.txt
pip install supervisely
```

4. Run the inference code

```python
# Be sure you are in the root of the RT-DETRv2 repository
from supervisely_integration.serve.rtdetrv2 import RTDETRv2

# Load model
model = RTDETRv2(
    model="./{1089_RT-DETRv2}/checkpoints/best.pt",  # path to the checkpoint you've downloaded
    device="cuda",  # or "cuda:1", "cpu"
)

# Predict
predictions = model(
    # Input can accpet various formats: image paths, np.arrays, Supervisely IDs and others.
    input=["path/to/image1.jpg", "path/to/image2.jpg"],
    conf=0.5,  # confidence threshold
    # ... additional parameters (see the docs)
)
```

> For more information, see [Prediction API](https://docs.supervisely.com/neural-networks/overview-1/prediction-api) and [Local Deployment](https://docs.supervisely.com/neural-networks/overview-1/local-deployment.md).

### Using ONNX and TensorRT

You can also use the exported ONNX or TensorRT models. For this, you need to specify the `model` parameter as a path to your ONNX or TensorRT model and provide class names in the additional `class_names` parameter.  🔴🔴🔴

```python
# Be sure you are in the root of the RT-DETRv2 repository
from supervisely_integration.serve.rtdetrv2 import RTDETRv2

# Deploy ONNX or TensorRT
model = RTDETRv2(
    model="./{1089_RT-DETRv2}/export/best.onnx",  # or "best.engine"
    device="cuda",
)

# Predict
predictions = model(
    # Input can accpet various formats: image paths, np.arrays, Supervisely IDs and others.
    input=["path/to/image1.jpg", "path/to/image2.jpg"],
    conf=0.5,  # confidence threshold
    # ... additional parameters (see the docs)
)
```

## Use the Model Outside of Supervisely

In this approach you'll completely decouple your model from both the **Supervisely Platform** and **Supervisely SDK**, and you will develop your own code for inference and deployment of that particular model. It's important to understand that for each neural network or a framework, you need to set up an environment and write inference code by yourself, since each model has its own installation instructions and the way of processing inputs and outputs correctly.

We provide a basic instructions and a demo script of how to load {RT-DETRv2} and get predictions using the original code from the authors.

1. Download checkpoint from Supervisely ([best.pt](xxx))

2. Prepare environment following instructions of the original repository [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)

3. Use the demo script for inference:

<details>
<summary>Click to expand</summary>

```python
"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw

from src.core import YAMLConfig


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='blue', )

        im.save(f'results_{i}.jpg')


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
```

</details>