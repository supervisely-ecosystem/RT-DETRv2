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

experiment_info = {
    "experiment_name": "47017_Train dataset - Insulator-Defect Detection_RT-DETRv2-S",
    "framework_name": "RT-DETRv2",
    "model_name": "RT-DETRv2-S",
    "task_type": "object detection",
    "project_id": 1112,
    "task_id": 47017,
    "model_files": {"config": "model_config.yml"},
    "checkpoints": [
        "checkpoints/best.pth",
        "checkpoints/checkpoint0005.pth",
        "checkpoints/checkpoint0010.pth",
        "checkpoints/checkpoint0015.pth",
        "checkpoints/checkpoint0020.pth",
        "checkpoints/checkpoint0025.pth",
        "checkpoints/checkpoint0030.pth",
        "checkpoints/checkpoint0035.pth",
        "checkpoints/checkpoint0040.pth",
        "checkpoints/checkpoint0045.pth",
        "checkpoints/checkpoint0050.pth",
        "checkpoints/checkpoint0055.pth",
        "checkpoints/checkpoint0060.pth",
        "checkpoints/checkpoint0065.pth",
        "checkpoints/checkpoint0070.pth",
        "checkpoints/checkpoint0075.pth",
        "checkpoints/checkpoint0080.pth",
        "checkpoints/last.pth",
    ],
    "best_checkpoint": "best.pth",
    "export": {"ONNXRuntime": "export/best.onnx", "TensorRT": "export/best.engine"},
    "app_state": "app_state.json",
    "model_meta": "model_meta.json",
    "hyperparameters": "hyperparameters.yaml",
    "artifacts_dir": "/experiments/1112_Train dataset - Insulator-Defect Detection/47017_RT-DETRv2/",
    "datetime": "2025-06-10 15:50:53",
    "evaluation_report_id": 629921,
    "evaluation_report_link": "https://dev.internal.supervisely.com/model-benchmark?id=629921",
    "evaluation_metrics": {
        "mAP": 0.6380200624108143,
        "AP50": 0.9148437682250835,
        "AP75": 0.6323428066528294,
        "f1": 0.6662328818472609,
        "precision": 0.7102697041300912,
        "recall": 0.6342230764698951,
        "iou": 0.8560792643796192,
        "classification_accuracy": 1,
        "calibration_score": 0.9041297329673619,
        "f1_optimal_conf": 0.642889142036438,
        "expected_calibration_error": 0.09587026703263817,
        "maximum_calibration_error": 0.2603313753550703,
    },
    "primary_metric": "mAP",
    "logs": {
        "type": "tensorboard",
        "link": "/experiments/1112_Train dataset - Insulator-Defect Detection/47017_RT-DETRv2/logs/",
    },
    "device": "NVIDIA GeForce RTX 4090",
    "train_val_split": "train_val_split.json",
    "train_size": 1296,
    "val_size": 144,
}

model_meta = {
    "classes": [
        {
            "title": "broken",
            "description": "",
            "shape": "rectangle",
            "color": "#FF00DB",
            "geometry_config": {},
            "id": 26627,
            "hotkey": "",
        },
        {
            "title": "insulator",
            "description": "",
            "shape": "rectangle",
            "color": "#0000FF",
            "geometry_config": {},
            "id": 26628,
            "hotkey": "",
        },
        {
            "title": "pollution-flashover",
            "description": "",
            "shape": "rectangle",
            "color": "#09FF00",
            "geometry_config": {},
            "id": 26629,
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
epoches: 80
batch_size: 16
eval_spatial_size: [640, 640]  # height, width

checkpoint_freq: 5
save_optimizer: false
save_ema: false

optimizer:
  type: AdamW
  lr: 0.0001
  betas: [0.9, 0.999]
  weight_decay: 0.0001

clip_max_norm: 0.1

lr_scheduler:
  type: MultiStepLR  # CosineAnnealingLR | OneCycleLR
  milestones: [35, 45]  # epochs
  gamma: 0.1

lr_warmup_scheduler:
  type: LinearWarmup
  warmup_duration: 1000  # steps

use_ema: True 
ema:
  type: ModelEMA
  decay: 0.9999
  warmups: 2000

use_amp: True
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
