import os
import shutil
import sys

sys.path.insert(0, "rtdetrv2_pytorch")
from multiprocessing import cpu_count

import yaml

import supervisely as sly
from rtdetrv2_pytorch.src.core import YAMLConfig
from rtdetrv2_pytorch.src.solver import DetSolver
from supervisely.nn.training.train_app import TrainApp
from supervisely.nn.utils import ModelSource
from supervisely_integration.train.serve import RTDETRv2Benchmark
from supervisely_integration.train.sly2coco import get_coco_annotations

base_path = "supervisely_integration/train"
train = TrainApp(
    "RT-DETRv2",
    f"supervisely_integration/models_v2.json",
    f"{base_path}/hyperparameters.yaml",
    f"{base_path}/app_options.yaml",
)

inference_settings = {"confidence_threshold": 0.4}
train.register_inference_class(RTDETRv2Benchmark, inference_settings)

# For debug
# app_state = {
#     "input": {"project_id": 42201},
#     "train_val_split": {"method": "random", "split": "train", "percent": 80},
#     "classes": ["dog", "horse", "cat", "squirrel", "sheep"],
#     "model": {"source": "Pretrained models", "model_name": "RT-DETRv2-S"},
#     "hyperparameters": "epoches: 2\nbatch_size: 16\neval_spatial_size: [640, 640]  # height, width\n\ncheckpoint_freq: 5\nsave_optimizer: false\nsave_ema: false\n\noptimizer:\n  type: AdamW\n  lr: 0.0001\n  betas: [0.9, 0.999]\n  weight_decay: 0.0001\n\nclip_max_norm: 0.1\n\nlr_scheduler:\n  type: MultiStepLR  # CosineAnnealingLR | OneCycleLR\n  milestones: [35, 45]  # epochs\n  gamma: 0.1\n\nlr_warmup_scheduler:\n  type: LinearWarmup\n  warmup_duration: 1000  # steps\n\nuse_ema: True \nema:\n  type: ModelEMA\n  decay: 0.9999\n  warmups: 2000\n\nuse_amp: True\n",
#     "options": {"model_benchmark": {"enable": True, "speed_test": True}, "cache_project": True},
# }
# train.gui.load_from_app_state(app_state)


@train.start
def start_training():
    train_ann_path, val_ann_path = convert_data()
    checkpoint = train.model_files["checkpoint"]
    custom_config_path = prepare_config()
    cfg = YAMLConfig(
        custom_config_path,
        tuning=checkpoint,
    )
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # dump resolved config
    model_config_path = f"{output_dir}/model_config.yml"
    with open(model_config_path, "w") as f:
        yaml.dump(cfg.yaml_cfg, f)
    # train
    tensorboard_logs = f"{output_dir}/summary"
    train.start_tensorboard(tensorboard_logs)
    solver = DetSolver(cfg)
    solver.fit()
    # gather experiment info
    experiment_info = {
        "task_type": sly.nn.TaskType.OBJECT_DETECTION,
        "model_name": train.model_name,
        "model_files": {"config": model_config_path},
        "checkpoints": output_dir,
        "best_checkpoint": "best.pth",
    }
    return experiment_info


def convert_data():
    project = train.sly_project
    meta = project.meta

    train_dataset: sly.Dataset = project.datasets.get("train")
    coco_anno = get_coco_annotations(train_dataset, meta, train.classes)
    train_ann_path = f"{train_dataset.directory}/coco_anno.json"
    sly.json.dump_json_file(coco_anno, train_ann_path, indent=None)

    val_dataset: sly.Dataset = project.datasets.get("val")
    coco_anno = get_coco_annotations(val_dataset, meta, train.classes)
    val_ann_path = f"{val_dataset.directory}/coco_anno.json"
    sly.json.dump_json_file(coco_anno, val_ann_path, indent=None)
    return train_ann_path, val_ann_path


def prepare_config():
    rtdetrv2_config_dir = "rtdetrv2_pytorch/configs/rtdetrv2"
    if train.model_source == ModelSource.CUSTOM:
        config_path = train.model_files["config"]
        config = os.path.basename(config_path)
        shutil.move(config_path, f"{rtdetrv2_config_dir}/{config}")
    else:
        config = train.model_files["config"]

    custom_config = train.hyperparameters
    custom_config["__include__"] = [config]
    custom_config["remap_mscoco_category"] = False
    custom_config["num_classes"] = train.num_classes
    custom_config["print_freq"] = 50

    custom_config.setdefault("train_dataloader", {}).setdefault("dataset", {})
    custom_config["train_dataloader"]["dataset"]["img_folder"] = f"{train.train_dataset_dir}/img"
    custom_config["train_dataloader"]["dataset"][
        "ann_file"
    ] = f"{train.train_dataset_dir}/coco_anno.json"

    custom_config.setdefault("val_dataloader", {}).setdefault("dataset", {})
    custom_config["val_dataloader"]["dataset"]["img_folder"] = f"{train.val_dataset_dir}/img"
    custom_config["val_dataloader"]["dataset"][
        "ann_file"
    ] = f"{train.val_dataset_dir}/coco_anno.json"

    if "batch_size" in custom_config:
        batch_size = custom_config["batch_size"]
        num_workers = min(batch_size, 8, cpu_count())
        custom_config["train_dataloader"]["total_batch_size"] = batch_size
        custom_config["val_dataloader"]["total_batch_size"] = batch_size * 2
        custom_config["train_dataloader"]["num_workers"] = num_workers
        custom_config["val_dataloader"]["num_workers"] = num_workers

    custom_config_path = f"{rtdetrv2_config_dir}/custom_config.yml"
    with open(custom_config_path, "w") as f:
        yaml.dump(custom_config, f)

    return custom_config_path
