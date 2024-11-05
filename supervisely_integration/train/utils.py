import os
from collections import OrderedDict
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import yaml
from pycocotools.coco import COCO

import supervisely as sly
from supervisely.app import DataJson
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    InputNumber,
    Stepper,
    Switch,
    Widget,
)
from supervisely.io.fs import get_file_name_with_ext


def get_num_workers(batch_size: int):
    num_workers = min(batch_size, 8, cpu_count())
    return num_workers


def read_parameters(hyperparameters: dict, default_config_path: str):
    sly.logger.debug("Reading training parameters...")
    with open(default_config_path, "r") as f:
        custom_config = f.read()
    custom_config = yaml.safe_load(custom_config)

    general_params = {
        "epoches": hyperparameters["general"]["n_epochs"],
        "val_step": hyperparameters["general"]["val_interval"],
        "checkpoint_step": hyperparameters["checkpoint"]["checkpoint_interval"],
        "clip_max_norm": hyperparameters["optimizer"]["clip_grad_norm"],
    }

    train_batch_size = hyperparameters["general"]["train_batch_size"]
    val_batch_size = hyperparameters["general"]["val_batch_size"]

    optimizer_params = {**hyperparameters["optimizer"]}
    scheduler_params = {**hyperparameters["lr_scheduler"]}

    sly.logger.debug(f"General parameters: {general_params}")
    sly.logger.debug(f"Optimizer parameters: {optimizer_params}")
    sly.logger.debug(f"Scheduler parameters: {scheduler_params}")

    custom_config.update(general_params)
    custom_config["optimizer"]["type"] = optimizer_params["type"]
    custom_config["optimizer"]["lr"] = optimizer_params["lr"]
    custom_config["optimizer"]["weight_decay"] = optimizer_params["weight_decay"]
    if optimizer_params.get("momentum"):
        custom_config["optimizer"]["momentum"] = optimizer_params["momentum"]
    elif optimizer_params.get("betas"):
        custom_config["optimizer"]["betas"] = [
            optimizer_params["betas"]["beta1"],
            optimizer_params["betas"]["beta2"],
        ]

    # Set input_size
    w, h = hyperparameters["general"]["input_image_size"]
    for op in custom_config["train_dataloader"]["dataset"]["transforms"]["ops"]:
        if op["type"] == "Resize":
            op["size"] = [w, h]
    for op in custom_config["val_dataloader"]["dataset"]["transforms"]["ops"]:
        if op["type"] == "Resize":
            op["size"] = [w, h]
    if "HybridEncoder" in custom_config:
        custom_config["HybridEncoder"]["eval_spatial_size"] = [w, h]
    else:
        custom_config["HybridEncoder"] = {"eval_spatial_size": [w, h]}
    if "RTDETRTransformer" in custom_config:
        custom_config["RTDETRTransformer"]["eval_spatial_size"] = [w, h]
    else:
        custom_config["RTDETRTransformer"] = {"eval_spatial_size": [w, h]}

    custom_config["train_dataloader"]["batch_size"] = train_batch_size
    custom_config["val_dataloader"]["batch_size"] = val_batch_size
    custom_config["train_dataloader"]["num_workers"] = get_num_workers(train_batch_size)
    custom_config["val_dataloader"]["num_workers"] = get_num_workers(val_batch_size)

    # LR scheduler
    if (
        scheduler_params["scheduler"] == "Without scheduler"
        or scheduler_params["scheduler"] == "empty"
    ):
        custom_config["lr_scheduler"] = None
    else:
        custom_config["lr_scheduler"] = {**scheduler_params}

    if scheduler_params["enable_warmup"]: #
        custom_config["lr_warmup"] = {
            "type": "LinearLR",
            "total_iters": scheduler_params["warmup_iterations"],
            "start_factor": 0.001,
            "end_factor": 1.0,
        }

    return custom_config


def prepare_config(
    project_info: sly.ProjectInfo,
    model_source: str,
    model_name: str,
    train_dataset_path: str,
    val_dataset_path: str,
    classes: List[str],
    custom_config: Dict[str, Any],
    custom_config_path: str,
):
    if model_source == "Pretrained models":
        model_name = model_name
        arch = model_name.split("_coco")[0]
        config_name = f"{arch}_6x_coco"
        sly.logger.info(f"Model name: {model_name}, arch: {arch}, config_name: {config_name}")
    else:
        model_name = get_file_name_with_ext(model_name=model_name)
        config_name = "custom"
        sly.logger.info(f"Model name: {model_name}, config_name: {config_name}")

    if model_source == "Pretrained models":
        custom_config["__include__"] = [f"{config_name}.yml"]
    else:
        custom_config["__include__"] = [
            "../dataset/coco_detection.yml",
            "../runtime.yml",
            "./include/dataloader.yml",
            "./include/optimizer.yml",
            "./include/rtdetr_r50vd.yml",
        ]
    custom_config["remap_mscoco_category"] = False
    custom_config["num_classes"] = len(classes)
    custom_config["train_dataloader"]["dataset"]["img_folder"] = f"{train_dataset_path}/img"
    custom_config["train_dataloader"]["dataset"][
        "ann_file"
    ] = f"{train_dataset_path}/coco_anno.json"
    custom_config["val_dataloader"]["dataset"]["img_folder"] = f"{val_dataset_path}/img"
    custom_config["val_dataloader"]["dataset"]["ann_file"] = f"{val_dataset_path}/coco_anno.json"
    selected_classes = classes
    custom_config["sly_metadata"] = {
        "classes": selected_classes,
        "project_id": project_info.id,
        "project_name": project_info.name,
        "model": model_name,
    }

    with open(custom_config_path, "w") as f:
        yaml.dump(custom_config, f)


def save_config(cfg):
    if "__include__" in cfg.yaml_cfg:
        cfg.yaml_cfg.pop("__include__")

    output_path = os.path.join(cfg.output_dir, "config.yml")

    with open(output_path, "w") as f:
        yaml.dump(cfg.yaml_cfg, f)


def create_trainval():
    # g.splits = splits.trainval_splits.get_splits()
    train_items, val_items = g.splits
    sly.logger.debug(f"Creating trainval datasets from splits: {g.splits}...")
    train_items: List[sly.project.project.ItemInfo]
    val_items: List[sly.project.project.ItemInfo]

    converted_project_dir = os.path.join(g.CONVERTED_DIR, g.project_info.name)
    sly.logger.debug(f"Converted project will be saved to {converted_project_dir}.")
    sly.fs.mkdir(converted_project_dir)
    train_dataset_path = os.path.join(converted_project_dir, "train")
    val_dataset_path = os.path.join(converted_project_dir, "val")
    sly.logger.debug(
        f"Train dataset path: {train_dataset_path}, val dataset path: {val_dataset_path}."
    )

    g.train_dataset_path = train_dataset_path
    g.val_dataset_path = val_dataset_path

    project_meta_path = os.path.join(converted_project_dir, "meta.json")
    sly.json.dump_json_file(g.project.meta.to_json(), project_meta_path)

    for items, dataset_path in zip(
        [train_items, val_items], [train_dataset_path, val_dataset_path]
    ):
        prepare_dataset(dataset_path, items)

    g.converted_project = sly.Project(converted_project_dir, sly.OpenMode.READ)
    sly.logger.info(f"Project created in {converted_project_dir}")

    for dataset_fs in g.converted_project.datasets:
        dataset_fs: sly.Dataset
        selected_classes = g.selected_classes

        coco_anno = get_coco_annotations(dataset_fs, g.converted_project.meta, selected_classes)
        coco_anno_path = os.path.join(dataset_fs.directory, "coco_anno.json")
        sly.json.dump_json_file(coco_anno, coco_anno_path)

    sly.logger.info("COCO annotations created")


def get_coco_annotations(dataset: sly.Dataset, meta: sly.ProjectMeta, selected_classes: List[str]):
    coco_anno = {"images": [], "categories": [], "annotations": []}
    cat2id = {name: i for i, name in enumerate(selected_classes)}
    img_id = 1
    ann_id = 1
    for name in dataset.get_items_names():
        ann = dataset.get_ann(name, meta)
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
    return coco_anno
