import sys
sys.path.insert(0, "rtdetrv2_pytorch")
import yaml
import supervisely as sly
from supervisely.nn.training.train_app import TrainApp
# from supervisely_integration.train.serve import RTDETRModelMB
from supervisely_integration.train.sly2coco import get_coco_annotations
from supervisely_integration.train.utils import get_num_workers
from rtdetrv2_pytorch.src.core import YAMLConfig
from rtdetrv2_pytorch.src.solver import DetSolver


base_path = "supervisely_integration/train"

_models = sly.json.load_json_file(f"{base_path}/models_v2.json")

train = TrainApp(
    "RT-DETRv2",
    _models,
    f"{base_path}/hyperparameters.yaml",
    {},
    sly.app.get_synced_data_dir(),
)

# train.register_inference_class(RTDETRModelMB)


@train.start
def start_training():
    checkpoint = train.model_files["checkpoint"]
    train_ann_path, val_ann_path = convert_data()
    custom_config = prepare_config()
    custom_config_path = "rtdetrv2_pytorch/configs/rtdetrv2/custom_config.yml"
    with open(custom_config_path, 'w') as f:
        yaml.dump(custom_config, f)
    cfg = YAMLConfig(
        custom_config_path,
        tuning=checkpoint,
    )
    output_dir = cfg.output_dir
    tensorboard_logs = f"{output_dir}/summary"
    model_config_path = f"{output_dir}/model_config.yml"
    with open(model_config_path, 'w') as f:
        yaml.dump(cfg.yaml_cfg, f)    
    # train
    # train.start_tensorboard(tensorboard_logs)
    solver = DetSolver(cfg)
    solver.fit()
    best_ckpt = f"{output_dir}/best.pth"
    last_ckpt = f"{output_dir}/last.pth"
    model_name = train.model_name

    # Gather experiment info
    experiment_info = {
        "model_name": model_name,
        "model_files": {"config": model_config_path},
        "checkpoints": output_dir,
        "best_checkpoint": "best.pth",
    }

    return experiment_info


def convert_data():
    project = train.sly_project
    meta = project.meta

    train_dataset : sly.Dataset = project.datasets.get("train")
    coco_anno = get_coco_annotations(train_dataset, meta, train.classes)
    train_ann_path = f"{train_dataset.directory}/coco_anno.json"
    sly.json.dump_json_file(coco_anno, train_ann_path, indent=None)

    val_dataset : sly.Dataset = project.datasets.get("val")
    coco_anno = get_coco_annotations(val_dataset, meta, train.classes)
    val_ann_path = f"{val_dataset.directory}/coco_anno.json"
    sly.json.dump_json_file(coco_anno, val_ann_path, indent=None)
    return train_ann_path, val_ann_path


def prepare_config():
    custom_config = train.hyperparameters
    custom_config["__include__"] = [train.model_files["config"]]
    custom_config["remap_mscoco_category"] = False
    custom_config["num_classes"] = train.num_classes
    custom_config["print_freq"] = 50

    custom_config.setdefault("train_dataloader", {}).setdefault("dataset", {})
    custom_config["train_dataloader"]["dataset"]["img_folder"] = f"{train.train_dataset_dir}/img"
    custom_config["train_dataloader"]["dataset"]["ann_file"] = f"{train.train_dataset_dir}/coco_anno.json"

    custom_config.setdefault("val_dataloader", {}).setdefault("dataset", {})
    custom_config["val_dataloader"]["dataset"]["img_folder"] = f"{train.val_dataset_dir}/img"
    custom_config["val_dataloader"]["dataset"]["ann_file"] = f"{train.val_dataset_dir}/coco_anno.json"

    if "batch_size" in custom_config:
        custom_config["train_dataloader"]["total_batch_size"] = custom_config["batch_size"]
        custom_config["val_dataloader"]["total_batch_size"] = custom_config["batch_size"] * 2
        custom_config["train_dataloader"]["num_workers"] = get_num_workers(custom_config["batch_size"])
        custom_config["val_dataloader"]["num_workers"] = get_num_workers(custom_config["batch_size"])
        
    return custom_config
