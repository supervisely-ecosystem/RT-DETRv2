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
	"experiment_name": "49487_Animals (Rectangle) (Old)_RT-DETRv2-S",
	"framework_name": "RT-DETRv2",
	"model_name": "RT-DETRv2-S",
	"base_checkpoint": "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
	"base_checkpoint_link": "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
	"task_type": "object detection",
	"project_id": 30,
	"project_version": {
		"id": 222,
		"version": 4
	},
	"task_id": 49487,
	"model_files": {
		"config": "model_config.yml"
	},
	"checkpoints": [
		"checkpoints/best.pth",
		"checkpoints/checkpoint0005.pth",
		"checkpoints/last.pth"
	],
	"best_checkpoint": "best.pth",
	"export": {},
	"app_state": "app_state.json",
	"model_meta": "model_meta.json",
	"hyperparameters": "hyperparameters.yaml",
	"artifacts_dir": "/experiments/30_Animals (Rectangle) (Old)/49487_RT-DETRv2/",
	"datetime": "2025-08-11 15:42:16",
	"evaluation_report_id": None,
	"evaluation_report_link": None,
	"evaluation_metrics": {},
	"primary_metric": None,
	"logs": {
		"type": "tensorboard",
		"link": "/experiments/30_Animals (Rectangle) (Old)/49487_RT-DETRv2/logs/"
	},
	"device": "NVIDIA GeForce RTX 4090",
	"training_duration": 9.387736642998789,
	"train_collection_id": 530,
	"val_collection_id": 531,
	"train_val_split": "train_val_split.json",
	"train_size": 12,
	"val_size": 12
}

model_meta = {
	"classes": [
		{
			"title": "cat",
			"description": "",
			"shape": "rectangle",
			"color": "#A80B10",
			"geometry_config": {},
			"id": 78,
			"hotkey": ""
		},
		{
			"title": "dog",
			"description": "",
			"shape": "rectangle",
			"color": "#B8E986",
			"geometry_config": {},
			"id": 79,
			"hotkey": ""
		},
		{
			"title": "horse",
			"description": "",
			"shape": "rectangle",
			"color": "#9F21DE",
			"geometry_config": {},
			"id": 80,
			"hotkey": ""
		},
		{
			"title": "sheep",
			"description": "",
			"shape": "rectangle",
			"color": "#1EA49B",
			"geometry_config": {},
			"id": 81,
			"hotkey": ""
		},
		{
			"title": "squirrel",
			"description": "",
			"shape": "rectangle",
			"color": "#F8E71C",
			"geometry_config": {},
			"id": 82,
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
			"id": 62,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "animal age group_1",
			"value_type": "any_string",
			"color": "#8A0F59",
			"id": 63,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "animal count",
			"value_type": "any_number",
			"color": "#E3BE1C",
			"id": 64,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "cat",
			"value_type": "none",
			"color": "#A80B10",
			"id": 65,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "dog",
			"value_type": "none",
			"color": "#B8E986",
			"id": 66,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "horse",
			"value_type": "none",
			"color": "#9F21DE",
			"id": 67,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "imgtag",
			"value_type": "none",
			"color": "#FF03D6",
			"id": 68,
			"hotkey": "",
			"applicable_type": "imagesOnly",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "sheep",
			"value_type": "none",
			"color": "#1EA49B",
			"id": 69,
			"hotkey": "",
			"applicable_type": "all",
			"classes": [],
			"target_type": "all"
		},
		{
			"name": "squirrel",
			"value_type": "none",
			"color": "#F8E71C",
			"id": 70,
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
epoches: 80
batch_size: 8
eval_spatial_size: [640, 640]  # height, width

checkpoint_freq: 40
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
