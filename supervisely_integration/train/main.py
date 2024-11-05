import os
import sys

cwd = os.getcwd()

rtdetr_pytorch_path = os.path.join(cwd, "rtdetr_pytorch")
sys.path.insert(0, rtdetr_pytorch_path)
from dotenv import load_dotenv
from models import get_models

import supervisely as sly
import supervisely_integration.train.utils as utils
from supervisely.nn.training.train_app import TrainApp

load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("local.env")

api: sly.Api = sly.Api.from_env()


task_id = 534523  # sly.env.task_id()
team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
project_id = sly.env.project_id()

# Path to configs
# cwd = os.getcwd()
# rtdetr_pytorch_path = os.path.join(cwd, "rtdetr_pytorch")
config_paths_dir = os.path.join(rtdetr_pytorch_path, "configs", "rtdetr")
default_config_path = os.path.join(config_paths_dir, "placeholder.yml")

app_options = {
    "save_best_checkpoint": True,
    "save_last_checkpoint": True,
    "supported_train_modes": ["finetune", "scratch"],
    "supported_optimizers": ["Adam", "AdamW", "SGD"],
    "supported_schedulers": [
        "Without scheduler",
        "CosineAnnealingLR",
        "LinearLR",
        "MultiStepLR",
        "OneCycleLR",
    ],
    "logging": {
        "enable": True,  # Enable logging
        "interval": 1,  # Interval for logging metrics
        "save_to_file": True,  # Save logs to file
        "metrics": [  # Metrics to log
            "accuracy",
            "loss",
            "mAP",
            "AP",
            "AR",
            "memory",
        ],
    },
    "evaluation": {
        "enable": True,  # Enable model evaluation during training
    },
}

app_config = {
    "project_id": 42201,
    "train_dataset_id": 99198,
    "val_dataset_id": 99199,
    "model": "yolov8s-det",
    "classes": ["cat", "dog"],
}

models = get_models()
models_path = os.path.join(os.path.dirname(__file__), "models.json")
hyperparameters_path = os.path.join(os.path.dirname(__file__), "hyperparameters.yaml")


current_file_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_file_dir, "output")
train = TrainApp(output_dir, models_path, hyperparameters_path, app_options)


@train.start
def start_training():
    print("-----------------")
    print("Start training")
    print("-----------------")
    import rtdetr_pytorch.train as train_cli

    custom_config_path = os.path.join(config_paths_dir, "custom.yml")
    hyperparameters = train.hyperparameters
    utils.read_parameters(hyperparameters, custom_config_path)

    finetune = True
    cfg = train_cli.train(
        finetune, custom_config_path, train.progress_bar_epochs, train.progress_bar_iters
    )
    utils.save_config(cfg)

    # upload_model in preprocess?
