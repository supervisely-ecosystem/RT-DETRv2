import os
import shutil
from datetime import datetime

import supervisely as sly
import yaml
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
    prepare_config()
    cfg = train()
    save_config(cfg)
    out_path = upload_model(cfg.output_dir)


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
