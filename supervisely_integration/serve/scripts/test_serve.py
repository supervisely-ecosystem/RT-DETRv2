import os
import time
from dotenv import load_dotenv
import supervisely as sly
from supervisely.nn.benchmark import ObjectDetectionBenchmark, InstanceSegmentationBenchmark
from supervisely_integration.serve.rtdetrv2 import RTDETRv2
from supervisely.nn import ModelSource, RuntimeType
import os

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))


model = RTDETRv2()
model.serve()

model_info = model.pretrained_models[0]

model._load_model_headless(
    model_files={
        "config": "rtdetrv2_r18vd_120e_coco.yml",
        "checkpoint": os.path.expanduser(
            "~/.cache/supervisely/checkpoints/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
        ),
    },
    model_info=model_info,
    model_source=ModelSource.PRETRAINED,
    device="cuda",
    runtime=RuntimeType.PYTORCH,
)

