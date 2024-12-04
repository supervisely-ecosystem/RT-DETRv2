import numpy as np
from supervisely_integration.serve.rtdetrv2 import RTDETRv2
from supervisely.nn import ModelSource, RuntimeType
from PIL import Image
import os


model = RTDETRv2()

model_info = model.pretrained_models[0]

model._load_model_headless(
    model_files={
        "config": "rtdetrv2_r18vd_120e_coco.yml",
        "checkpoint": os.path.expanduser("~/.cache/supervisely/checkpoints/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"),
    },
    model_info=model_info,
    model_source=ModelSource.PRETRAINED,
    device="cuda",
    runtime=RuntimeType.TENSORRT,
)

image = Image.open("supervisely_integration/serve/scripts/coco_sample.jpg").convert("RGB")
img = np.array(image)

ann = model._inference_auto([img], {"confidence_threshold": 0.5})[0][0]

ann.draw_pretty(img)
Image.fromarray(img).save("supervisely_integration/serve/scripts/predict.jpg")
