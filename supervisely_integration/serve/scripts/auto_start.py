import os

from dotenv import load_dotenv

import supervisely as sly
from supervisely.nn.utils import ModelSource, RuntimeType

load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("local.env")

api: sly.Api = sly.Api.from_env()

task_id = 68910  # <---- Change this to your task_id
method = "deploy_from_api"


# Pretrained
pretrained_model_data = {
    "deploy_params": {
        "model_files": {
            "config": "rtdetrv2_r18vd_120e_coco.yml",
            "checkpoint": "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
        },
        "model_source": ModelSource.PRETRAINED,
        "model_info": {
            "Model": "RT-DETRv2-L",
            "dataset": "COCO",
            "AP_val": 53.4,
            "Params(M)": 42,
            "FPS(T4)": 108,
            "meta": {
                "task_type": "object detection",
                "model_name": "RT-DETRv2-L",
                "model_files": {
                    "checkpoint": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth",
                    "config": "rtdetrv2_r50vd_6x_coco.yml",
                },
            },
        },
        "device": "cuda",
        "runtime": RuntimeType.PYTORCH,
    }
}

# Custom
# @TODO: add deploy_params.json to team files for convenience?
custom_model_data = {
    "deploy_params": {
        "model_files": {
            "config": "/experiments/43192_Apples/68908_RT-DETRv2/model_config.yml",
            "checkpoint": "/experiments/43192_Apples/68908_RT-DETRv2/checkpoints/best.pth",
        },
        "model_source": ModelSource.CUSTOM,
        "model_info": {
            "artifacts_dir": "/experiments/43192_Apples/68908_RT-DETRv2",
            "framework_name": "RT-DETRv2",
            "model_name": "RT-DETRv2-S",
            "model_meta": "model_meta.json",
        },
        "device": "cuda",
        "runtime": RuntimeType.PYTORCH,
    }
}

api.app.send_request(task_id, method, custom_model_data)
