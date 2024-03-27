import os
import shutil
from datetime import datetime
from typing import Any, Dict, List

import supervisely as sly
import yaml
from pycocotools.coco import COCO
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    Editor,
    Field,
    Input,
    InputNumber,
    RadioTabs,
    ReloadableArea,
    Select,
    BindedInputNumber,
)

import rtdetr_pytorch.train as train_cli
import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.output as output
import supervisely_integration.train.utils as utils

# region advanced widgets
advanced_mode_checkbox = Checkbox("Advanced mode")
advanced_mode_field = Field(
    advanced_mode_checkbox,
    title="Advanced mode",
    description="Enable advanced mode to specify custom training parameters manually.",
)
with open(g.default_config_path, "r") as f:
    default_config = f.read()
    sly.logger.debug(f"Loaded default config from {g.default_config_path}")
advanced_mode_editor = Editor(default_config, language_mode="yaml", height_lines=100)
advanced_mode_editor.hide()
# endregion

# region general widgets
number_of_epochs_input = InputNumber(value=20, min=1)
number_of_epochs_field = Field(
    number_of_epochs_input,
    title="Number of epochs",
    description="The number of epochs to train the model for",
)
input_size_input = BindedInputNumber(640, 640)
input_size_field = Field(
    input_size_input,
    title="Input size",
    description="Images will be resized to this size.",
)

train_batch_size_input = InputNumber(value=4, min=1)
train_batch_size_field = Field(
    train_batch_size_input,
    title="Train batch size",
    description="The number of images in a batch during training",
)

val_batch_size_input = InputNumber(value=8, min=1)
val_batch_size_field = Field(
    val_batch_size_input,
    title="Validation batch size",
    description="The number of images in a batch during validation",
)

validation_interval_input = InputNumber(value=1, min=1)
validation_interval_field = Field(
    validation_interval_input,
    title="Validation interval",
    description="The number of epochs between each validation run",
)

checkpoints_interval_input = InputNumber(value=1, min=1)
checkpoints_interval_field = Field(
    checkpoints_interval_input,
    title="Checkpoint interval",
    description="The number of epochs between each checkpoint save",
)

general_tab = Container(
    [
        number_of_epochs_field,
        input_size_field,
        train_batch_size_field,
        val_batch_size_field,
        validation_interval_field,
        checkpoints_interval_field,
    ]
)
# endregion

# region checkpoints widgets
# checkpoints_interval_input = InputNumber(value=1, min=1)
# checkpoints_interval_field = Field(
#     checkpoints_interval_input,
#     title="Checkpoints interval",
#     description="The number of epochs between each checkpoint save",
# )

# save_last_checkpoint_checkbox = Checkbox("Save last checkpoint")
# save_best_checkpoint_checkbox = Checkbox("Save best checkpoint")

# save_checkpoint_field = Field(
#     Container([save_last_checkpoint_checkbox, save_best_checkpoint_checkbox]),
#     title="Save checkpoints",
#     description="Choose which checkpoints to save",
# )

# checkpoints_tab = Container([checkpoints_interval_field, save_checkpoint_field])

# endregion

# region optimizer widgets
optimizer_select = Select([Select.Item(opt) for opt in g.OPTIMIZERS])
optimizer_field = Field(
    optimizer_select,
    title="Select optimizer",
    description="Choose the optimizer to use for training",
)
learning_rate_input = InputNumber(value=0.0002)
learning_rate_field = Field(
    learning_rate_input,
    title="Learning rate",
    description="The learning rate to use for the optimizer",
)
wight_decay_input = InputNumber(value=0.0001)
wight_decay_field = Field(
    wight_decay_input,
    title="Weight decay",
    description="The amount of L2 regularization to apply to the weights",
)
momentum_input = InputNumber(value=0.9)
momentum_field = Field(
    momentum_input,
    title="Momentum",
    description="The amount of momentum to apply to the weights",
)
momentum_field.hide()
beta1_input = InputNumber(value=0.9)
beta1_field = Field(
    beta1_input,
    title="Beta 1",
    description="The exponential decay rate for the first moment estimates",
)
beta2_input = InputNumber(value=0.999)
beta2_field = Field(
    beta2_input,
    title="Beta 2",
    description="The exponential decay rate for the second moment estimates",
)

clip_gradient_norm_checkbox = Checkbox("Clip gradient norm")
clip_gradient_norm_input = InputNumber(value=0.1)
clip_gradient_norm_field = Field(
    Container([clip_gradient_norm_checkbox, clip_gradient_norm_input]),
    title="Clip gradient norm",
    description="Select the highest gradient norm to clip the gradients",
)


optimization_tab = Container(
    [
        optimizer_field,
        learning_rate_field,
        wight_decay_field,
        momentum_field,
        beta1_field,
        beta2_field,
        clip_gradient_norm_field,
    ]
)

# endregion

# region scheduler widgets
scheduler_select = Select([Select.Item(sch) for sch in g.SCHEDULERS])

scheduler_widgets_container = Container()
scheduler_parameters_area = ReloadableArea(scheduler_widgets_container)

enable_warmup_checkbox = Checkbox("Enable warmup", True)
warmup_iterations_input = InputNumber(value=2)
warmup_iterations_field = Field(
    warmup_iterations_input,
    title="Warmup iterations",
    description="The number of iterations to warm up the learning rate",
)
warmup_ratio_input = InputNumber(value=0.001)
warmup_ratio_field = Field(
    warmup_ratio_input,
    title="Warmup ratio",
    description="The ratio of the initial learning rate to use for warmup",
)
warmup_container = Container([warmup_iterations_field, warmup_ratio_field])

learning_rate_scheduler_tab = Container(
    [
        scheduler_select,
        scheduler_parameters_area,
        enable_warmup_checkbox,
        warmup_container,
    ]
)

# endregion

run_button = Button("Run training")
stop_button = Button("Stop training", button_type="danger")
stop_button.hide()


parameters_tabs = RadioTabs(
    ["General", "Optimizer", "Learning rate scheduler"],
    contents=[
        general_tab,
        # checkpoints_tab,
        optimization_tab,
        learning_rate_scheduler_tab,
    ],
)

card = Card(
    title="5️⃣ Training hyperparameters",
    description="Specify training hyperparameters using one of the methods.",
    collapsable=True,
    content=Container(
        [advanced_mode_field, advanced_mode_editor, parameters_tabs, run_button],
    ),
    content_top_right=stop_button,
)
card.lock()


@advanced_mode_checkbox.value_changed
def advanced_mode_changed(is_checked: bool):
    if is_checked:
        advanced_mode_editor.show()
        parameters_tabs.hide()
    else:
        advanced_mode_editor.hide()
        parameters_tabs.show()


@optimizer_select.value_changed
def optimizer_changed(optimizer: str):
    if optimizer == "Adam":
        beta1_field.show()
        beta2_field.show()
        momentum_field.hide()
    elif optimizer == "AdamW":
        beta1_field.hide()
        beta2_field.hide()
        momentum_field.hide()
    elif optimizer == "SGD":
        beta1_field.hide()
        beta2_field.hide()
        momentum_field.show()


@enable_warmup_checkbox.value_changed
def warmup_changed(is_checked: bool):
    if is_checked:
        warmup_container.show()
    else:
        warmup_container.hide()


@scheduler_select.value_changed
def scheduler_changed(scheduler: str):
    # StepLR: by_epoch_checkbox, LR scheduler step (input number), gamma (input number)
    # MultiStepLR: by_epoch_checkbox, LR scheduler steps (input), gamma (input number)
    # ExponentialLR: by_epoch_checkbox, gamma (input number)
    # ReduceLROnPlateauLR: by_epoch_checkbox, factor (input number), patience (input number)
    # CosineAnnealingLR: by_epoch_checkbox, T_max (input number), min lr checkbox, min lr (input number), min lr ratio (input number)
    # CosineRestartLR: by_epoch_checkbox, periods (input), restarts (input), min lr checkbox, min lr (input number), min lr ratio (input number)

    scheduler_widgets_container._widgets.clear()
    scheduler_parameters_area.reload()
    g.widgets = None

    if not scheduler != "Without scheduler":
        by_epoch = Checkbox("By epoch", True)
        by_epoch_field = Field(by_epoch, title="By epoch", description="Use epoch-based scheduler")
        widgets = {
            "by_epoch": by_epoch,
            "by_epoch_field": by_epoch_field,
        }
    else:
        widgets = {}

    if scheduler == "StepLR":
        step = InputNumber(value=1)
        step_field = Field(
            step,
            title="Step",
            description="Period of learning rate decay",
        )
        widgets["step"] = step
        widgets["step_field"] = step_field
    elif scheduler == "MultiStepLR":
        # Decays the learning rate of each parameter group by gamma once the
        # number of epoch reaches one of the milestones
        steps = Input("15,18")
        steps_field = Field(
            steps,
            title="Milestones",
            description="List of epoch indices. Must be increasing.",
        )
        gamma = InputNumber(value=0.1)
        gamma_field = Field(
            gamma,
            title="Gamma",
            description="Multiplicative factor of learning rate decay",
        )
        widgets["steps"] = steps
        widgets["steps_field"] = steps_field
        widgets["gamma"] = gamma
        widgets["gamma_field"] = gamma_field
    elif scheduler == "ExponentialLR":
        gamma = InputNumber(value=0.1)
        gamma_field = Field(
            gamma,
            title="Gamma",
            description="Multiplicative factor of learning rate decay",
        )
        widgets["gamma"] = gamma
        widgets["gamma_field"] = gamma_field
    elif scheduler == "ReduceLROnPlateauLR":
        factor = InputNumber(value=0.1)
        factor_field = Field(
            factor,
            title="Factor",
            description="Factor by which the learning rate will be reduced",
        )
        patience = InputNumber(value=10)
        patience_field = Field(
            patience,
            title="Patience",
            description="Number of epochs with no improvement after which learning rate will be reduced",
        )
        widgets["factor"] = factor
        widgets["factor_field"] = factor_field
        widgets["patience"] = patience
        widgets["patience_field"] = patience_field
    elif scheduler == "CosineAnnealingLR":
        # Set the learning rate of each parameter group using a cosine annealing
        # schedule, where :math:`\eta_{max}` is set to the initial lr and
        # :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
        t_max = InputNumber(value=10)
        t_max_field = Field(
            t_max,
            title="T max",
            description="Maximum number of iterations",
        )

        widgets["t_max"] = t_max
        widgets["t_max_field"] = t_max_field

        # min_lr_checkbox = Checkbox("Min LR", True)

        # @min_lr_checkbox.value_changed
        # def min_lr_changed(is_checked: bool):
        #     if is_checked:
        #         widgets["min_lr_value"].enable()
        #         widgets["min_lr_ratio"].disable()
        #     else:
        #         widgets["min_lr_value"].disable()
        #         widgets["min_lr_ratio"].enable()

        # min_lr_field = Field(
        #     min_lr_checkbox,
        #     title="Min LR",
        #     description="Use minimum learning rate",
        # )
        # widgets["min_lr"] = min_lr_checkbox
        # widgets["min_lr_field"] = min_lr_field

        min_lr_value = InputNumber(value=0.)
        min_lr_value_field = Field(
            min_lr_value,
            title="Min LR value",
            description="Minimum learning rate",
        )
        widgets["min_lr_value"] = min_lr_value
        widgets["min_lr_value_field"] = min_lr_value_field

        # min_lr_ratio = InputNumber(value=0.1)
        # min_lr_ratio_field = Field(
        #     min_lr_ratio,
        #     title="Min LR ratio",
        #     description="Minimum learning rate ratio",
        # )
        # widgets["min_lr_ratio"] = min_lr_ratio
        # widgets["min_lr_ratio_field"] = min_lr_ratio_field

        # widgets["min_lr_ratio"].disable()

    elif scheduler == "CosineRestartLR":
        periods = Input("10,20,30")
        periods_field = Field(
            periods,
            title="Periods",
            description="Periods for restarts",
        )
        restarts = Input("2,3,4")
        restarts_field = Field(
            restarts,
            title="Restarts",
            description="Number of restarts",
        )
        widgets["periods"] = periods
        widgets["periods_field"] = periods_field
        widgets["restarts"] = restarts
        widgets["restarts_field"] = restarts_field

        min_lr_checkbox = Checkbox("Min LR", True)

        @min_lr_checkbox.value_changed
        def min_lr_changed(is_checked: bool):
            if is_checked:
                widgets["min_lr_value"].enable()
                widgets["min_lr_ratio"].disable()
            else:
                widgets["min_lr_value"].disable()
                widgets["min_lr_ratio"].enable()

        min_lr_field = Field(
            min_lr_checkbox,
            title="Min LR",
            description="Use minimum learning rate",
        )

        widgets["min_lr"] = min_lr_checkbox
        widgets["min_lr_field"] = min_lr_field

        min_lr_value = InputNumber(value=0.001)
        min_lr_value_field = Field(
            min_lr_value,
            title="Min LR value",
            description="Minimum learning rate",
        )
        widgets["min_lr_value"] = min_lr_value
        widgets["min_lr_value_field"] = min_lr_value_field

        min_lr_ratio = InputNumber(value=0.1)
        min_lr_ratio_field = Field(
            min_lr_ratio,
            title="Min LR ratio",
            description="Minimum learning rate ratio",
        )
        widgets["min_lr_ratio"] = min_lr_ratio
        widgets["min_lr_ratio_field"] = min_lr_ratio_field

        widgets["min_lr_ratio"].disable()

    elif scheduler == "OneCycleLR":
        # Sets the learning rate of each parameter group according to the
        # 1cycle learning rate policy. The 1cycle policy anneals the learning
        # rate from an initial learning rate to some maximum learning rate and then
        # from that maximum learning rate to some minimum learning rate much lower
        # than the initial learning rate.
        max_lr = InputNumber(value=0.005)
        max_lr_field = Field(
            max_lr,
            title="Max LR",
            description="Maximum learning rate",
        )
        pct_start = InputNumber(value=0.3)
        pct_start_field = Field(
            pct_start,
            title="Pct start",
            description="The percentage of the cycle spent increasing the learning rate",
        )
        widgets["max_lr"] = max_lr
        widgets["max_lr_field"] = max_lr_field
        widgets["pct_start"] = pct_start
        widgets["pct_start_field"] = pct_start_field

    g.widgets = widgets

    scheduler_widgets_container._widgets.extend(
        [widget for key, widget in widgets.items() if key.endswith("_field")]
    )
    scheduler_parameters_area.reload()


@run_button.click
def run_training():
    output.card.unlock()

    download_project()
    create_trainval()

    custom_config = read_parameters()
    prepare_config(custom_config)
    cfg = train()
    save_config(cfg)
    out_path = upload_model(cfg.output_dir)
    print(out_path)


@stop_button.click
def stop_training():
    # TODO: Implement the stop process
    pass


def read_parameters():
    sly.logger.debug("Reading training parameters...")
    if advanced_mode_checkbox.is_checked():
        sly.logger.info("Advanced mode enabled, using custom config from the editor.")
        custom_config = advanced_mode_editor.get_value()
    else:
        sly.logger.info("Advanced mode disabled, reading parameters from the widgets.")
        with open(g.default_config_path, "r") as f:
            custom_config = f.read()
        custom_config = yaml.safe_load(custom_config)

        clip_max_norm = clip_gradient_norm_input.get_value() if clip_gradient_norm_checkbox.is_checked() else -1
        general_params = {
            "epoches": number_of_epochs_input.value,
            "val_step": validation_interval_input.value,
            "checkpoint_step": checkpoints_interval_input.value,
            "clip_max_norm": clip_max_norm,
        }
        optimizer_params = read_optimizer_parameters()
        scheduler_params = read_scheduler_parameters()

        sly.logger.debug(f"General parameters: {general_params}")
        sly.logger.debug(f"Optimizer parameters: {optimizer_params}")
        sly.logger.debug(f"Scheduler parameters: {scheduler_params}")

        custom_config.update(general_params)
        custom_config["optimizer"]["type"] = optimizer_params["optimizer"]
        custom_config["optimizer"]["lr"] = optimizer_params["learning_rate"]
        custom_config["optimizer"]["weight_decay"] = optimizer_params["weight_decay"]
        if optimizer_params.get("momentum"):
            custom_config["optimizer"]["momentum"] = optimizer_params["momentum"]
        else:
            custom_config["optimizer"]["betas"] = [optimizer_params["beta1"], optimizer_params["beta2"]]

        # Set input_size
        w,h = input_size_input.get_value()
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
        custom_config["train_dataloader"]["num_workers"] = utils.get_num_workers(train_batch_size_input.value)
        custom_config["val_dataloader"]["num_workers"] = utils.get_num_workers(val_batch_size_input.value)
        
        # LR scheduler
        # TODO: warmup
        custom_config["lr_scheduler"] = scheduler_params["scheduler"]
        if scheduler_params["scheduler"] == "MultiStepLR":
            custom_config["lr_scheduler"] = {
                "type": "MultiStepLR",
                "milestones": scheduler_params["steps"],
                "gamma": scheduler_params["gamma"],
            }
        elif scheduler_params["scheduler"] == "CosineAnnealingLR":
            custom_config["lr_scheduler"] = {
                "type": "CosineAnnealingLR",
                "T_max": scheduler_params["t_max"],
                "eta_min": scheduler_params["min_lr_value"],
            }
        elif scheduler_params["scheduler"] == "OneCycleLR":
            total_steps = general_params["epoches"] * (len(g.converted_project.datasets.get("train")) // train_batch_size_input.value)
            custom_config["lr_scheduler"] = {
                "type": "OneCycleLR",
                "max_lr": scheduler_params["max_lr"],
                "total_steps": total_steps,
                "pct_start": scheduler_params["pct_start"],
            }
        
        # TODO: set imgaug
        if False:
            ops = custom_config["train_dataloader"]["dataset"]["transforms"]["ops"]
            for i, op in enumerate(ops):
                if op["type"] == "Resize":
                    resize_idx = i
                    break
            imgaug_op = {"type": "ImgAug", "config_path": "imgaug.json"}
            custom_config["train_dataloader"]["dataset"]["transforms"]["ops"] = [imgaug_op] + ops[resize_idx:]

    return custom_config


def read_optimizer_parameters():
    optimizer = optimizer_select.get_value()

    parameters = {
        "optimizer": optimizer,
        "learning_rate": learning_rate_input.get_value(),
        "weight_decay": wight_decay_input.get_value(),
        "clip_gradient_norm": clip_gradient_norm_checkbox.is_checked(),
        "clip_gradient_norm_value": clip_gradient_norm_input.get_value(),
    }

    if optimizer in ["Adam", "AdamW"]:
        parameters.update(
            {
                "beta1": beta1_input.get_value(),
                "beta2": beta2_input.get_value(),
            }
        )
    elif optimizer == "SGD":
        parameters.update({"momentum": momentum_input.get_value()})

    return parameters


def read_scheduler_parameters():
    scheduler = scheduler_select.get_value()

    parameters = {
        "scheduler": scheduler,
        "enable_warmup": enable_warmup_checkbox.is_checked(),
        "warmup_iterations": warmup_iterations_input.get_value(),
    }

    if g.widgets is not None:
        for key, widget in g.widgets.items():
            if isinstance(widget, (InputNumber, Input)):
                parameters[key] = widget.get_value()
            elif isinstance(widget, Checkbox):
                parameters[key] = widget.is_checked()

    return parameters


def prepare_config(custom_config: Dict[str, Any]):
    model_name = g.train_mode.pretrained[0]
    arch = model_name.split("_coco")[0]
    config_name = f"{arch}_6x_coco"
    sly.logger.info(f"Model name: {model_name}, arch: {arch}, config_name: {config_name}")

    custom_config["__include__"] = [f"{config_name}.yml"]
    custom_config["remap_mscoco_category"] = False
    custom_config["num_classes"] = len(g.selected_classes)
    custom_config["train_dataloader"]["dataset"]["img_folder"] = f"{g.train_dataset_path}/img"
    custom_config["train_dataloader"]["dataset"]["ann_file"] = f"{g.train_dataset_path}/coco_anno.json"
    custom_config["val_dataloader"]["dataset"]["img_folder"] = f"{g.val_dataset_path}/img"
    custom_config["val_dataloader"]["dataset"]["ann_file"] = f"{g.val_dataset_path}/coco_anno.json"
    selected_classes = [obj_class.name for obj_class in g.selected_classes]
    custom_config["sly_metadata"] = {
        "classes": selected_classes,
        "project_id": g.selected_project_id,
        "project_name": g.selected_project_info.name,
        "model": model_name,
    }

    g.custom_config_path = os.path.join(g.CONFIG_PATHS_DIR, "custom.yml")
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
