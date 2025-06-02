import os

from dotenv import load_dotenv

import supervisely as sly
from supervisely.template.experiment.experiment_generator import ExperimentGenerator
from supervisely_integration.serve.main import RTDETRv2

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()


model = api.nn.deploy(
    model="/experiments/27_Lemons (Rectangle)/2053_RT-DETRv2/export/best.onnx",
    device="cuda:0",
    runtime="ONNXRuntime",
)

predictions = model.predict(
    input=["supervisely_integration/train/coco_sample.jpg"],
)
print(predictions)
