import os
import yaml
from rtdetrv2_pytorch.src.core import YAMLConfig
from rtdetrv2_pytorch.src.solver import DetSolver
import supervisely as sly
from supervisely_integration.train.sly2coco import get_coco_annotations
from supervisely_integration.train.utils import get_num_workers


api = sly.Api()

train_dataset_id = 96440
val_dataset_id = 96439
train_dataset_name = api.dataset.get_info_by_id(train_dataset_id).name
val_dataset_name = api.dataset.get_info_by_id(val_dataset_id).name
project_id = api.dataset.get_info_by_id(train_dataset_id).project_id
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
selected_classes = [obj_class.name for obj_class in project_meta.obj_classes]

models_json = "supervisely_integration/train/models_v2.json"
models = sly.json.load_json_file(models_json)
project_dir = "./app_data/project"
user_hyperparametrs = ""
with open("supervisely_integration/train/hyperparameters.yaml") as f:
    user_hyperparametrs = f.read()
model = models[0]
config_path = model["meta"]["model_files"]["config"]
config = os.path.basename(config_path)
checkpoint = "app_data/models/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
custom_config_path = "rtdetrv2_pytorch/configs/rtdetrv2/custom_config.yaml"


def prepare_data():
    # download
    if not os.path.exists(project_dir):
        sly.download(api, project_id, project_dir, dataset_ids=[train_dataset_id, val_dataset_id])
    project = sly.read_project(project_dir)
    meta = project.meta

    train_dataset : sly.Dataset = project.datasets.get(train_dataset_name)
    coco_anno = get_coco_annotations(train_dataset, meta, selected_classes)
    sly.json.dump_json_file(coco_anno, f"{train_dataset.directory}/coco_anno.json", indent=None)

    val_dataset : sly.Dataset = project.datasets.get(val_dataset_name)
    coco_anno = get_coco_annotations(val_dataset, meta, selected_classes)
    sly.json.dump_json_file(coco_anno, f"{val_dataset.directory}/coco_anno.json", indent=None)


def prepare_config():
    custom_config = yaml.safe_load(user_hyperparametrs) or {}
    custom_config["__include__"] = [config]
    custom_config["remap_mscoco_category"] = False
    custom_config["num_classes"] = len(selected_classes)

    custom_config.setdefault("train_dataloader", {}).setdefault("dataset", {})
    custom_config["train_dataloader"]["dataset"]["img_folder"] = f"{project_dir}/{train_dataset_name}/img"
    custom_config["train_dataloader"]["dataset"]["ann_file"] = f"{project_dir}/{train_dataset_name}/coco_anno.json"

    custom_config.setdefault("val_dataloader", {}).setdefault("dataset", {})
    custom_config["val_dataloader"]["dataset"]["img_folder"] = f"{project_dir}/{val_dataset_name}/img"
    custom_config["val_dataloader"]["dataset"]["ann_file"] = f"{project_dir}/{val_dataset_name}/coco_anno.json"

    if "batch_size" in custom_config:
        custom_config["train_dataloader"]["total_batch_size"] = custom_config["batch_size"]
        custom_config["val_dataloader"]["total_batch_size"] = custom_config["batch_size"] * 2
        custom_config["train_dataloader"]["num_workers"] = get_num_workers(custom_config["batch_size"])
        custom_config["val_dataloader"]["num_workers"] = get_num_workers(custom_config["batch_size"])
        
    # save custom config
    with open(custom_config_path, 'w') as f:
        yaml.dump(custom_config, f)
    return custom_config_path


if __name__ == "__main__":
    prepare_data()
    config_path = prepare_config()
    cfg = YAMLConfig(
        config_path,
        tuning=checkpoint,
    )
    output_dir = cfg.output_dir
    cfg.print_freq = 50
    os.makedirs(output_dir, exist_ok=True)
    tensorboard_logs = f"{output_dir}/summary"
    model_config_path = f"{output_dir}/model_config.yml"
    # dump config
    import yaml
    with open(model_config_path, 'w') as f:
        yaml.dump(cfg.yaml_cfg, f)    
    # train
    solver = DetSolver(cfg)
    solver.fit()
    best_ckpt = f"{output_dir}/best.pth"
    last_ckpt = f"{output_dir}/last.pth"
    model_name = model["meta"]["model_name"]