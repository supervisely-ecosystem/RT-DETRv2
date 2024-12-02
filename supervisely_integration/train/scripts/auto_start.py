import os

from dotenv import load_dotenv

import supervisely as sly

load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("local.env")

api: sly.Api = sly.Api.from_env()

task_id = 68843  # <---- Change this to your task_id
method = "train_from_api"

hyperparameters_path = os.path.join(os.path.dirname(__file__), "hyperparameters.yaml")
with open(hyperparameters_path, "r") as f:
    hyper_params = f.read()
    print(hyper_params)

data = {
    "app_state": {
        "input": {
            "project_id": 43166,
            "train_dataset_id": 101769,
            "val_dataset_id": 101770,
        },
        "train_val_split": {"method": "random", "split": "train", "percent": 80},
        "classes": ["cat", "dog"],
        "model": {
            # Pretrain
            "source": "Pretrained models",
            "model_name": "RT-DETRv2-S",
            # Custom
            # "source": "Custom models",
            # "task_id": "debug-session",
            # "checkpoint": "checkpoint0011.pth",
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
}

api.app.send_request(task_id, method, data)
