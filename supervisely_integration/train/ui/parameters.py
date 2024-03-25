import os
import shutil
from datetime import datetime
from typing import List

import supervisely as sly
import yaml
from pycocotools.coco import COCO
from supervisely.app.widgets import Button, Card, Container, Editor, RadioTabs

import rtdetr_pytorch.train as train_cli
import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.output as output

general_tab = Container()
checkpoints_tab = Container()
optimization_tab = Container()

parameters_editor = Editor(language_mode="yaml", height_lines=100)
advanced_tab = Container([parameters_editor])

run_button = Button("Run training")
stop_button = Button("Stop training", button_type="danger")
stop_button.hide()


parameters_tabs = RadioTabs(
    ["General", "Checkpoints", "Optimization", "Advanced"],
    contents=[
        general_tab,
        checkpoints_tab,
        optimization_tab,
        Container([general_tab, checkpoints_tab, optimization_tab, advanced_tab]),
    ],
)

card = Card(
    title="5️⃣ Training hyperparameters",
    description="Specify training hyperparameters using one of the methods.",
    collapsable=True,
    content=Container(
        [parameters_tabs, run_button],
    ),
    content_top_right=stop_button,
)
card.lock()
card.collapse()


@run_button.click
def run_training():
    output.card.unlock()
    output.card.uncollapse()

    download_project()
    create_trainval()

    prepare_config()
    cfg = train()
    save_config(cfg)
    out_path = upload_model(cfg.output_dir)
    print(out_path)


@stop_button.click
def stop_training():
    # TODO: Implement the stop process
    pass


def prepare_config():
    custom_config_text = parameters_editor.get_value()
    model_name = g.train_mode.pretrained[0]
    arch = model_name.split("_coco")[0]
    config_name = f"{arch}_6x_coco"
    sly.logger.info(f"Model name: {model_name}, arch: {arch}, config_name: {config_name}")

    custom_config = yaml.safe_load(custom_config_text) or {}
    custom_config["__include__"] = [f"{config_name}.yml"]
    custom_config["remap_mscoco_category"] = False
    custom_config["num_classes"] = len(g.selected_classes)

    custom_config["train_dataloader"] = {
        "dataset": {
            "img_folder": f"{g.train_dataset_path}/img",
            "ann_file": f"{g.train_dataset_path}/coco_anno.json",
        }
    }
    custom_config["val_dataloader"] = {
        "dataset": {
            "img_folder": f"{g.val_dataset_path}/img",
            "ann_file": f"{g.val_dataset_path}/coco_anno.json",
        }
    }
    selected_classes = [obj_class.name for obj_class in g.selected_classes]
    custom_config["sly_metadata"] = {
        "classes": selected_classes,
        "project_id": g.selected_project_id,
        "project_name": g.selected_project_info.name,
        "model": model_name,
    }

    g.custom_config_path = os.path.join(g.CONFIG_PATHS_DIR, "custom_config.yml")
    with open(g.custom_config_path, "w") as f:
        yaml.dump(custom_config, f)


def train():
    model = g.train_mode.pretrained[0]
    finetune = g.train_mode.finetune
    cfg = train_cli.train(model, finetune, g.custom_config_path)
    return cfg


def save_config(cfg):
    if "__include__" in cfg.yaml_cfg:
        cfg.yaml_cfg.pop("__include__")

    output_path = os.path.join(g.OUTPUT_DIR, "config.yml")

    with open(output_path, "w") as f:
        yaml.dump(cfg.yaml_cfg, f)


def upload_model(output_dir):
    model_name = g.train_mode.pretrained[0]
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    team_files_dir = (
        f"/RT-DETR/{g.selected_project_info.name}_{g.selected_project_id}/{timestamp}_{model_name}"
    )
    local_dir = f"{output_dir}/upload"
    sly.fs.mkdir(local_dir)

    checkpoints = [f for f in os.listdir(output_dir) if f.endswith(".pth")]
    latest_checkpoint = sorted(checkpoints)[-1]
    shutil.move(f"{output_dir}/{latest_checkpoint}", f"{local_dir}/{latest_checkpoint}")
    shutil.move(f"{output_dir}/log.txt", f"{local_dir}/log.txt")
    shutil.move("output/config.yml", f"{local_dir}/config.yml")

    out_path = g.api.file.upload_directory(
        sly.env.team_id(),
        local_dir,
        team_files_dir,
    )
    return out_path


def download_project():
    g.project_dir = os.path.join(g.DOWNLOAD_DIR, g.selected_project_info.name)
    sly.logger.info(f"Downloading project to {g.project_dir}...")
    sly.Project.download(g.api, g.selected_project_info.id, g.project_dir)
    sly.logger.info(f"Project downloaded to {g.project_dir}.")
    g.project = sly.Project(g.project_dir, sly.OpenMode.READ)
    sly.logger.info(f"Project loaded from {g.project_dir}.")


def create_trainval():
    train_items, val_items = g.splits
    sly.logger.debug(f"Creating trainval datasets from splits: {g.splits}...")
    train_items: List[sly.project.project.ItemInfo]
    val_items: List[sly.project.project.ItemInfo]

    converted_project_dir = os.path.join(g.CONVERTED_DIR, g.selected_project_info.name)
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
        selected_classes = [obj_class.name for obj_class in g.selected_classes]

        coco_anno = get_coco_annotations(dataset_fs, g.converted_project.meta, selected_classes)
        coco_anno_path = os.path.join(dataset_fs.directory, "coco_anno.json")
        sly.json.dump_json_file(coco_anno, coco_anno_path)

    sly.logger.info("COCO annotations created")


def prepare_dataset(dataset_path: str, items: List[sly.project.project.ItemInfo]):
    sly.logger.debug(f"Preparing dataset in {dataset_path}...")
    img_dir = os.path.join(dataset_path, "img")
    ann_dir = os.path.join(dataset_path, "ann")
    sly.fs.mkdir(img_dir)
    sly.fs.mkdir(ann_dir)
    for item in items:
        src_img_path = os.path.join(g.project_dir, fix_widget_path(item.img_path))
        src_ann_path = os.path.join(g.project_dir, fix_widget_path(item.ann_path))
        dst_img_path = os.path.join(img_dir, item.name)
        dst_ann_path = os.path.join(ann_dir, f"{item.name}.json")
        sly.fs.copy_file(src_img_path, dst_img_path)
        sly.fs.copy_file(src_ann_path, dst_ann_path)

    sly.logger.info(f"Dataset prepared in {dataset_path}")


def fix_widget_path(bugged_path: str) -> str:
    """Fixes the broken ItemInfo paths from TrainValSplits widget.
    Removes the first two folders from the path.

    Bugged path: app_data/1IkWRgJG62f1ZuZ/ds0/ann/pexels_2329440.jpeg.json
    Corrected path: ds0/ann/pexels_2329440.jpeg.json

    :param bugged_path: Path to fix
    :type bugged_path: str
    :return: Fixed path
    :rtype: str
    """
    return "/".join(bugged_path.split("/")[2:])


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
