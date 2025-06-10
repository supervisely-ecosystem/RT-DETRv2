import os

import supervisely as sly
from dotenv import load_dotenv
from supervisely.template.experiment.experiment_generator import ExperimentGenerator

from supervisely_integration.serve.main import RTDETRv2

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()

experiment_info = {
    "experiment_name": "2053_Lemons (Rectangle)_RT-DETRv2-M",
    "framework_name": "RT-DETRv2",
    "model_name": "RT-DETRv2-M",
    "task_type": "object detection",
    "project_id": 27,
    "task_id": 2053,
    "device": "NVIDIA GeForce RTX 4070 Ti SUPER",
    "model_files": {"config": "model_config.yml"},
    "checkpoints": [
        "checkpoints/best.pth",
        "checkpoints/checkpoint0025.pth",
        "checkpoints/checkpoint0050.pth",
        "checkpoints/last.pth",
    ],
    "best_checkpoint": "best.pth",
    "export": {"ONNXRuntime": "export/best.onnx", "TensorRT": "export/best.engine"},
    "app_state": "app_state.json",
    "model_meta": "model_meta.json",
    "train_val_split": "train_val_split.json",
    "train_size": 4,
    "val_size": 2,
    "hyperparameters": "hyperparameters.yaml",
    "artifacts_dir": "/experiments/27_Lemons (Rectangle)/2053_RT-DETRv2/",
    "datetime": "2025-03-06 11:13:41",
    "evaluation_report_id": 301554,
    "evaluation_report_link": "https://dev.internal.supervisely.com/model-benchmark?id=301554",
    "evaluation_metrics": {
        "mAP": 0.9886588658865886,
        "AP50": 1,
        "AP75": 1,
        "f1": 0.99,
        "precision": 0.99,
        "recall": 0.99,
        "iou": 0.9705370106892338,
        "classification_accuracy": 1,
        "calibration_score": 0.886729476368412,
        "f1_optimal_conf": 0.666441023349762,
        "expected_calibration_error": 0.11327052363158797,
        "maximum_calibration_error": 0.5632345080375671,
    },
    "primary_metric": "mAP",
    "logs": {
        "type": "tensorboard",
        "link": "/experiments/27_Lemons (Rectangle)/2053_RT-DETRv2/logs/",
    },
}

model_meta = {
    "classes": [
        {
            "title": "kiwi",
            "description": "",
            "shape": "rectangle",
            "color": "#FF0000",
            "geometry_config": {},
            "id": 70,
            "hotkey": "",
        },
        {
            "title": "lemon",
            "description": "",
            "shape": "rectangle",
            "color": "#51C6AA",
            "geometry_config": {},
            "id": 71,
            "hotkey": "",
        },
    ],
    "tags": [],
    "projectType": "images",
    "projectSettings": {
        "multiView": {"enabled": False, "tagName": None, "tagId": None, "isSynced": False}
    },
}
model_meta = sly.ProjectMeta.from_json(model_meta)

hyperparameters_yaml = """
epoches: 50
batch_size: 2
eval_spatial_size: [640, 640]  # height, width

checkpoint_freq: 25
save_optimizer: false
save_ema: false

optimizer:
  type: AdamW
  lr: 0.0001
  betas: [0.9, 0.999]
  weight_decay: 0.0001

clip_max_norm: 1.

lr_scheduler:
  type: MultiStepLR  # CosineAnnealingLR | OneCycleLR
  milestones: [350, 450]  # epochs
  gamma: 0.1

lr_warmup_scheduler:
  type: LinearWarmup
  warmup_duration: 10  # steps

use_ema: False
ema:
  type: ModelEMA
  decay: 0.9999
  warmups: 200

use_amp: False
"""
app_options = {
    "demo": {
        "path": "supervisely_integration/demo",
    },
}


experiment = ExperimentGenerator(
    api=api,
    experiment_info=experiment_info,
    hyperparameters=hyperparameters_yaml,
    model_meta=model_meta,
    serving_class=RTDETRv2,
    team_id=team_id,
    output_dir="./experiment_report",
    app_options=app_options,
)

experiment.generate()
experiment.upload_to_artifacts()
