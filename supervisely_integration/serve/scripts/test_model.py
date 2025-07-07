from dotenv import load_dotenv
import supervisely as sly

from supervisely_integration.serve.rtdetrv2 import RTDETRv2

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv("supervisely.env")

# 1. Pretrained model
# model = RTDETRv2(
#     model="RT-DETRv2-S",
#     device="cuda",
# )

# 2. Local checkpoint
# model_local = RTDETRv2(
#     model="47705_RT-DETRv2/checkpoints/best.pth",
#     device="cuda",
# )

# 3. Remote Custom Checkpoint (Team Files)
# model_remote = RTDETRv2(
    # model="/experiments/9_Animals (Bitmap)/47705_RT-DETRv2/checkpoints/best.pth",
    # device="cuda:0",
# )

model = RTDETRv2(
    model="RT-DETRv2-S",
    device="cuda",
)

image_path = "supervisely_integration/demo/img/coco_sample.jpg"
model(image_path)
