import os

from dotenv import load_dotenv

import supervisely as sly
from supervisely.template.experiment.experiment_generator import ExperimentGenerator
from supervisely_integration.serve.rtdetrv2 import RTDETRv2

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()

experiment_info = {
	"experiment_name": "49800 Animals (Rectangle) RT-DETRv2-M",
	"framework_name": "RT-DETRv2",
	"model_name": "RT-DETRv2-M",
	"base_checkpoint": "rtdetrv2_r50vd_m_7x_coco_ema.pth",
	"base_checkpoint_link": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth",
	"task_type": "object detection",
	"project_id": 1322,
	"project_version": {
		"id": 322,
		"version": 13
	},
	"task_id": 49800,
	"model_files": {
		"config": "model_config.yml"
	},
	"checkpoints": [
		"checkpoints/best.pth",
		"checkpoints/checkpoint0005.pth",
		"checkpoints/checkpoint0010.pth",
		"checkpoints/last.pth"
	],
	"best_checkpoint": "best.pth",
	"export": {},
	"app_state": "app_state.json",
	"model_meta": "model_meta.json",
	"hyperparameters": "hyperparameters.yaml",
	"artifacts_dir": "/experiments/1322_Animals (Rectangle)/49800_RT-DETRv2/",
	"datetime": "2025-08-14 08:53:47",
	"evaluation_report_id": None,
	"evaluation_report_link": None,
	"evaluation_metrics": {},
	"primary_metric": None,
	"logs": {
		"type": "tensorboard",
		"link": "/experiments/1322_Animals (Rectangle)/49800_RT-DETRv2/logs/"
	},
	"device": "NVIDIA GeForce RTX 4090",
	"training_duration": 32.86879684301675,
	"train_collection_id": 574,
	"val_collection_id": 575,
	"train_val_split": "train_val_split.json",
	"train_size": 27,
	"val_size": 27
}

model_meta = {
	"classes": [
		{
			"title": "cat",
			"description": "",
			"shape": "rectangle",
			"color": "#A80B10",
			"geometry_config": {},
			"id": 32406,
			"hotkey": ""
		},
		{
			"title": "dog",
			"description": "",
			"shape": "rectangle",
			"color": "#B8E986",
			"geometry_config": {},
			"id": 32407,
			"hotkey": ""
		},
		{
			"title": "horse",
			"description": "",
			"shape": "rectangle",
			"color": "#9F21DE",
			"geometry_config": {},
			"id": 32408,
			"hotkey": ""
		},
		{
			"title": "sheep",
			"description": "",
			"shape": "rectangle",
			"color": "#1EA49B",
			"geometry_config": {},
			"id": 32409,
			"hotkey": ""
		},
		{
			"title": "squirrel",
			"description": "",
			"shape": "rectangle",
			"color": "#F8E71C",
			"geometry_config": {},
			"id": 32410,
			"hotkey": ""
		}
	],
	"tags": [
		{
			"name": "animal age group",
			"value_type": "oneof_string",
			"color": "#F5A623",
			"values": [
				"juvenile",
				"adult",
				"senior"
			],
			"id": 4781,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "animal age group_1",
			"value_type": "any_string",
			"color": "#8A0F59",
			"id": 4782,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "animal count",
			"value_type": "any_number",
			"color": "#E3BE1C",
			"id": 4783,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "cat",
			"value_type": "none",
			"color": "#A80B10",
			"id": 4784,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "dog",
			"value_type": "none",
			"color": "#B8E986",
			"id": 4785,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "horse",
			"value_type": "none",
			"color": "#9F21DE",
			"id": 4786,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "imgtag",
			"value_type": "none",
			"color": "#FF03D6",
			"id": 4787,
			"hotkey": "",
			"applicable_type": "imagesOnly",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "sheep",
			"value_type": "none",
			"color": "#1EA49B",
			"id": 4788,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "squirrel",
			"value_type": "none",
			"color": "#F8E71C",
			"id": 4789,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		}
	],
	"projectType": "images",
	"projectSettings": {
		"multiView": {
			"enabled": False,
			"tagName": None,
			"tagId": None,
			"isSynced": False
		}
	}
}
model_meta = sly.ProjectMeta.from_json(model_meta)

hyperparameters_yaml = """
epoches: 10
batch_size: 2
eval_spatial_size: [640, 640]  # height, width

checkpoint_freq: 5
save_optimizer: false
save_ema: false

optimizer:
  type: AdamW
  lr: 0.0001
  betas: [0.9, 0.999]
  weight_decay: 0.0001

clip_max_norm: 10.0

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
    "original_repository": {
        "name": "RT-DETR",
        "url": "https://github.com/lyuwenyu/RT-DETR",
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
