import os

from dotenv import load_dotenv
import supervisely as sly
from supervisely.nn.tracker.utils import predictions_to_video_annotation
from supervisely.io.json import dump_json_file

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
session_id = 51344 # 50964
model_api = api.nn.connect(session_id)

image_path = False
image_id = False
video_id = True

# Data

# path
if image_path is True:
    image_path = "supervisely_integration/demo/img/coco_sample.jpg"
    predictions = model_api.predict(input=image_path)

# image id
if image_id is True:
    project_id = 3530
    dataset_id = 16776
    image_id = 1475824
    predictions = model_api.predict(image_id=image_id, upload_mode="replace")
    print(predictions)

# video id
if video_id is True:
    project_id = 3598
    dataset_id = 16914
    video_id = 1498079
    session = model_api.predict_detached(video_id=video_id, upload_mode="replace", tracking=True)
    predictions = list(session)
    video_ann = predictions_to_video_annotation(predictions)
    print(video_ann)
    video_ann_json = session.final_result
    print(video_ann_json)
    # dump_json_file(video_ann.to_json(), "video_ann.json", indent=4)
