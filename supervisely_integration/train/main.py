import json
import os
import shutil
import sys
from typing import List

import yaml
from pycocotools.coco import COCO

from supervisely.io.fs import get_file_name, get_file_name_with_ext
from supervisely.nn.task_type import TaskType
from supervisely_integration.train.serve import RTDETRModelMB

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

config_paths_dir = os.path.join(rtdetr_pytorch_path, "configs", "rtdetr")
default_config_path = os.path.join(config_paths_dir, "placeholder.yml")

app_options = {
    "gpu_selector": True,
    "multi_gpu_training": True,
    "data_format": ["COCO", "VOC", "YOLO"],  # etc
    "use_coco_annotation": True,
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
            "Train/loss",
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
work_dir = os.path.join(current_file_dir, "output")
train = TrainApp("rt-detr", models_path, hyperparameters_path, app_options, work_dir)

inference_settings = {"confidence_threshold": 0.4}
train.register_inference_class(RTDETRModelMB, inference_settings)


# train.init_logger(logger="supervisely")

# from src.serve import RTDETR
# def init_logger(self, logger):
# from sly.train import train_logger
#
# train_logger.init(self)


@train.start
def start_training():
    print("-----------------")
    print("Start training")
    print("-----------------")

    import rtdetr_pytorch.train as train_cli

    # Step 0. Clean output dit
    # Step 1. Convert and prepare Project
    converted_project_dir = os.path.join(train.work_dir, "converted_project")
    convert2coco(train.sly_project, converted_project_dir, train.classes)

    # Step 2. Prepare config and read hyperparameters
    custom_config_path = prepare_config(train, converted_project_dir)

    # Step 3. Train
    cfg, best_checkpoint_path = train_cli.train(train, custom_config_path)

    # Step 4. Optional.
    # Move everything you want to upload to output dir
    # cfg.output_dir contain all train generated files
    output_models_dir = os.path.join(cfg.output_dir, "weights")
    os.makedirs(output_models_dir, exist_ok=True)
    for file in os.listdir(cfg.output_dir):
        if file.endswith(".pth"):
            shutil.move(os.path.join(cfg.output_dir, file), os.path.join(output_models_dir, file))

    if train.model_source == "Pretrained models":
        model_name = train.model_parameters["Model"]
    else:
        model_name = train.model_parameters["model_name"]

    best_checkpoint_path = os.path.join(
        output_models_dir, get_file_name_with_ext(best_checkpoint_path)
    )

    experiment_info = {
        "model_name": model_name,
        "task_type": TaskType.OBJECT_DETECTION,
        "model_files": {"config": custom_config_path},
        "checkpoints": output_models_dir,  # or ["output_dir/checkpoints/epoch_10.pt", ...]
        "best_checkpoint": best_checkpoint_path,
    }

    return experiment_info


def convert2coco(project: sly.Project, converted_project_dir: str, selected_classes: List[str]):
    sly.logger.info("Converting project to COCO format")
    for dataset in project.datasets:
        dataset: sly.Dataset
        coco_anno = {"images": [], "categories": [], "annotations": []}
        cat2id = {name: i for i, name in enumerate(selected_classes)}
        img_id = 1
        ann_id = 1
        for name in dataset.get_items_names():
            ann = dataset.get_ann(name, project.meta)
            img_dict = {
                "id": img_id,
                "height": ann.img_size[0],
                "width": ann.img_size[1],
                "file_name": name,
            }
            coco_anno["images"].append(img_dict)

            for label in ann.labels:
                if isinstance(label.geometry, (sly.Bitmap, sly.Polygon)):
                    rect = label.geometry.to_bbox()
                elif isinstance(label.geometry, sly.Rectangle):
                    rect = label.geometry
                else:
                    continue
                class_name = label.obj_class.name
                if class_name not in selected_classes:
                    continue
                x, y, x2, y2 = rect.left, rect.top, rect.right, rect.bottom
                ann_dict = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat2id[class_name],
                    "bbox": [x, y, x2 - x, y2 - y],
                    "area": (x2 - x) * (y2 - y),
                    "iscrowd": 0,
                }
                coco_anno["annotations"].append(ann_dict)
                ann_id += 1

            img_id += 1

        coco_anno["categories"] = [{"id": i, "name": name} for name, i in cat2id.items()]

        # Test:
        coco_api = COCO()
        coco_api.dataset = coco_anno
        coco_api.createIndex()

        if dataset.name == "train":
            converted_ds_dir = os.path.join(converted_project_dir, "train")
        elif dataset.name == "val":
            converted_ds_dir = os.path.join(converted_project_dir, "val")

        converted_img_dir = os.path.join(converted_ds_dir, "img")
        converted_ann_dir = os.path.join(converted_ds_dir, "ann")
        converted_ann_path = os.path.join(converted_ann_dir, "coco_anno.json")
        os.makedirs(converted_img_dir, exist_ok=True)
        os.makedirs(converted_ann_dir, exist_ok=True)
        sly.json.dump_json_file(coco_anno, converted_ann_path)

        # Move items
        for image_name, image_path, _ in dataset.items():
            shutil.move(image_path, os.path.join(converted_img_dir, image_name))
        sly.logger.info(f"Dataset: '{dataset.name}' converted to COCO format")


def prepare_config(train: TrainApp, converted_project_dir: str):

    # Train / Val paths
    train_ds_dir = os.path.join(converted_project_dir, "train")
    train_img_dir = os.path.join(train_ds_dir, "img")
    train_ann_path = os.path.join(train_ds_dir, "ann", "coco_anno.json")

    val_ds_dir = os.path.join(converted_project_dir, "val")
    val_img_dir = os.path.join(val_ds_dir, "img")
    val_ann_path = os.path.join(val_ds_dir, "ann", "coco_anno.json")

    # Detect config from model parameters
    model_parameters = train.model_parameters
    if train.model_source == "Pretrained models":
        selected_model_name = model_parameters["Model"]
        arch = selected_model_name.split("_coco")[0]
        config_name = f"{arch}_6x_coco"
        custom_config_path = os.path.join(config_paths_dir, f"{config_name}.yml")
    else:
        selected_model_name = model_parameters.get("checkpoint_name")
        config_name = get_file_name(model_parameters.get("config_url"))
        custom_config_path = train.model_config_path

    # Read custom config
    with open(custom_config_path, "r") as f:
        custom_config = yaml.safe_load(f)

    # Fill custom config
    custom_config["__include__"] = [f"{config_name}.yml"]
    custom_config["remap_mscoco_category"] = False
    custom_config["num_classes"] = train.num_classes
    if "train_dataloader" not in custom_config:
        custom_config["train_dataloader"] = {
            "dataset": {
                "img_folder": train_img_dir,
                "ann_file": train_ann_path,
            }
        }
    else:
        custom_config["train_dataloader"]["dataset"]["img_folder"] = train_img_dir
        custom_config["train_dataloader"]["dataset"]["ann_file"] = train_ann_path
    if "val_dataloader" not in custom_config:
        custom_config["val_dataloader"] = {
            "dataset": {
                "img_folder": val_img_dir,
                "ann_file": val_ann_path,
            }
        }
    else:
        custom_config["val_dataloader"]["dataset"]["img_folder"] = val_img_dir
        custom_config["val_dataloader"]["dataset"]["ann_file"] = val_ann_path

    # Merge with hyperparameters
    hyperparameters = train.hyperparameters
    custom_config.update(hyperparameters)

    custom_config_path = os.path.join(config_paths_dir, "custom.yml")
    with open(custom_config_path, "w") as f:
        yaml.safe_dump(custom_config, f)

    # Copy to output dir also

    return custom_config_path
