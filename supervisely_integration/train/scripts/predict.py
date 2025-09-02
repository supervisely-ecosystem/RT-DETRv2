import os

from dotenv import load_dotenv

import supervisely as sly

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()

# Deploy model
# model_api = api.nn.deploy(
#     model="RT-DETRv2/RT-DETRv2-M",
#     # model="/experiments/27_Lemons (Rectangle)/2053_RT-DETRv2/checkpoints/best.pth",
#     # model="/experiments/27_Lemons (Rectangle)/2053_RT-DETRv2/export/best.onnx",
#     # model="/experiments/27_Lemons (Rectangle)/2053_RT-DETRv2/export/best.engine",
#     device="cuda:0",
#     # runtime="TensorRT",
# )

# Connect to model
session_id = 50756
model_api = api.nn.connect(session_id)

# Data

# path
# image_path = "supervisely_integration/demo/img/coco_sample.jpg"
# predictions = model_api.predict(input=image_path)

# image id
image_id = 1425234
predictions = model_api.predict(image_id=image_id, upload_mode="append")

# video id
# video_id = 110593
# predictions = model_api.predict(video_id=video_id, upload_mode="append")

print(predictions)
