import os
from dotenv import load_dotenv
import supervisely as sly

from supervisely_integration.serve.rtdetrv2 import RTDETRv2

if sly.is_development():
    load_dotenv("local.env")
    # load_dotenv("supervisely.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

model_n = 1

# 1. Pretrained model
if model_n == 1:
    model = RTDETRv2(
        model="RT-DETRv2-S",
        device="cuda",
    )

# 2. Local checkpoint
elif model_n == 2:
    model = RTDETRv2(
        model="my_models/best.pth",
        device="cuda",
    )

# 3. Remote Custom Checkpoint (Team Files)
elif model_n == 3:
    model = RTDETRv2(
        model="/experiments/9_Animals (Bitmap)/47688_RT-DETRv2/checkpoints/best.pth",
        device="cuda:0",
    )

image_path = "supervisely_integration/demo/img/coco_sample.jpg"
predictions = model(input=image_path)
print(f"Predictions: {len(predictions)}")