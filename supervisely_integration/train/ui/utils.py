import os
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional

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


def read_parameters(train_items_cnt: int):
    sly.logger.debug("Reading training parameters...")
    if advanced_mode_checkbox.is_checked():
        sly.logger.info("Advanced mode enabled, using custom config from the editor.")
        custom_config = advanced_mode_editor.get_value()
    else:
        sly.logger.info("Advanced mode disabled, reading parameters from the widgets.")
        with open(g.default_config_path, "r") as f:
            custom_config = f.read()
        custom_config = yaml.safe_load(custom_config)

        clip_max_norm = (
            clip_gradient_norm_input.get_value() if clip_gradient_norm_checkbox.is_checked() else -1
        )
        general_params = {
            "epoches": number_of_epochs_input.value,
            "val_step": validation_interval_input.value,
            "checkpoint_step": checkpoints_interval_input.value,
            "clip_max_norm": clip_max_norm,
        }

        total_steps = general_params["epoches"] * np.ceil(
            train_items_cnt / train_batch_size_input.value
        )

        optimizer_params = read_optimizer_parameters()
        scheduler_params, scheduler_cls_params = read_scheduler_parameters(total_steps)

        sly.logger.debug(f"General parameters: {general_params}")
        sly.logger.debug(f"Optimizer parameters: {optimizer_params}")
        sly.logger.debug(f"Scheduler parameters: {scheduler_cls_params}")

        custom_config.update(general_params)
        custom_config["optimizer"]["type"] = optimizer_params["optimizer"]
        custom_config["optimizer"]["lr"] = optimizer_params["learning_rate"]
        custom_config["optimizer"]["weight_decay"] = optimizer_params["weight_decay"]
        if optimizer_params.get("momentum"):
            custom_config["optimizer"]["momentum"] = optimizer_params["momentum"]
        else:
            custom_config["optimizer"]["betas"] = [
                optimizer_params["beta1"],
                optimizer_params["beta2"],
            ]

        # Set input_size
        w, h = input_size_input.get_value()
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

        custom_config["train_dataloader"]["batch_size"] = train_batch_size_input.value
        custom_config["val_dataloader"]["batch_size"] = val_batch_size_input.value
        custom_config["train_dataloader"]["num_workers"] = utils.get_num_workers(
            train_batch_size_input.value
        )
        custom_config["val_dataloader"]["num_workers"] = utils.get_num_workers(
            val_batch_size_input.value
        )

        # LR scheduler
        if scheduler_params["type"] == "Without scheduler":
            custom_config["lr_scheduler"] = None
        else:
            custom_config["lr_scheduler"] = scheduler_cls_params

        if scheduler_params["enable_warmup"]:
            custom_config["lr_warmup"] = {
                "type": "LinearLR",
                "total_iters": scheduler_params["warmup_iterations"],
                "start_factor": 0.001,
                "end_factor": 1.0,
            }

    return custom_config


def prepare_config(custom_config: Dict[str, Any]):
    if g.model_mode == g.MODEL_MODES[0]:
        model_name = g.train_mode.pretrained[0]
        arch = model_name.split("_coco")[0]
        config_name = f"{arch}_6x_coco"
        sly.logger.info(f"Model name: {model_name}, arch: {arch}, config_name: {config_name}")
    else:
        model_name = get_file_name_with_ext(g.train_mode.custom)
        config_name = "custom"
        sly.logger.info(f"Model name: {model_name}, config_name: {config_name}")

    if g.model_mode == g.MODEL_MODES[0]:
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
    custom_config["num_classes"] = len(g.selected_classes)
    custom_config["train_dataloader"]["dataset"]["img_folder"] = f"{g.train_dataset_path}/img"
    custom_config["train_dataloader"]["dataset"][
        "ann_file"
    ] = f"{g.train_dataset_path}/coco_anno.json"
    custom_config["val_dataloader"]["dataset"]["img_folder"] = f"{g.val_dataset_path}/img"
    custom_config["val_dataloader"]["dataset"]["ann_file"] = f"{g.val_dataset_path}/coco_anno.json"
    selected_classes = g.selected_classes
    custom_config["sly_metadata"] = {
        "classes": selected_classes,
        "project_id": g.PROJECT_ID,
        "project_name": g.project_info.name,
        "model": model_name,
    }

    g.custom_config_path = os.path.join(g.CONFIG_PATHS_DIR, "custom.yml")
    with open(g.custom_config_path, "w") as f:
        yaml.dump(custom_config, f)


def save_config(cfg):
    if "__include__" in cfg.yaml_cfg:
        cfg.yaml_cfg.pop("__include__")

    output_path = os.path.join(cfg.output_dir, "config.yml")

    with open(output_path, "w") as f:
        yaml.dump(cfg.yaml_cfg, f)


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
