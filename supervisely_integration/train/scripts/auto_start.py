import os

from dotenv import load_dotenv

import supervisely as sly

load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("local.env")

api: sly.Api = sly.Api.from_env()
hyperparameters_path = os.path.join(os.path.dirname(__file__), "hyperparameters.yaml")
with open(hyperparameters_path, "r") as f:
    hyper_params = f.read()
    print(hyper_params)

app_config = {
    "input": {
        "project_id": 42201,
        "train_dataset_id": 99198,
        "val_dataset_id": 99199,
    },
    "classes": ["cat", "dog"],
    "model": {
        # Pretrain
        # "source": "Pretrained models",
        # "model_name": "rtdetr_r50vd_coco_objects365",
        # Custom
        "source": "Custom models",
        "task_id": "debug-session",
        "checkpoint": "checkpoint0011.pth",
    },
    "hyperparameters": hyper_params,
    "options": {
        "model_benchmark": {
            "enable": True,
            "speed_test": True,
        },
        "cache_project": True,
    },
}

api.app.send_request()
