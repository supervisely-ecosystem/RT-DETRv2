import os
import sys

import supervisely as sly
from dotenv import load_dotenv

if sly.is_development:
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

# region constants
cwd = os.getcwd()
sly.logger.debug(f"Current working directory: {cwd}")
rtdetr_pytorch_path = os.path.join(cwd, "rtdetr_pytorch")
sys.path.insert(0, rtdetr_pytorch_path)
sly.logger.debug("Added rtdetr_pytorch to the system path")
CONFIG_PATHS_DIR = os.path.join(rtdetr_pytorch_path, "configs", "rtdetr")
default_config_path = os.path.join(CONFIG_PATHS_DIR, "placeholder.yml")
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sly.logger.debug(f"Current directory: {CURRENT_DIR}")
TEMP_DIR = os.path.join(CURRENT_DIR, "temp")
DOWNLOAD_DIR = os.path.join(TEMP_DIR, "download")
CONVERTED_DIR = os.path.join(TEMP_DIR, "converted")
sly.fs.mkdir(DOWNLOAD_DIR, remove_content_if_exists=True)
sly.fs.mkdir(CONVERTED_DIR, remove_content_if_exists=True)
sly.logger.debug(f"Download dir: {DOWNLOAD_DIR}, converted dir: {CONVERTED_DIR}")
OUTPUT_DIR = os.path.join(TEMP_DIR, "output")
sly.fs.mkdir(OUTPUT_DIR, remove_content_if_exists=True)
sly.logger.debug(f"Output dir: {OUTPUT_DIR}")

MODEL_MODES = ["Pretrained models", "Custom weights"]
TABLE_COLUMNS = [
    "Name",
    "Method",
    "Dataset",
    "Inference time",
    "Training epochs",
    "Training memory",
    "Box AP",
]
PRETRAINED_MODELS = [
    [
        "rtdetr_r18vd_coco",
        "METHOD1",
        "DATASET1",
        "INFERENCETIME1",
        "TRAININGEPOCHS1",
        "TRAININGMEMORY1",
        "BOXAP1",
    ],
    [
        "rtdetr_r34vd_coco",
        "METHOD2",
        "DATASET2",
        "INFERENCETIME2",
        "TRAININGEPOCHS2",
        "TRAININGMEMORY2",
        "BOXAP2",
    ],
]
CHECKPOINTS = {
    "rtdetr_r18vd_coco": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth",
    "rtdetr_r34vd_coco": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth",
    "rtdetr_r50vd_m_coco": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth",
    "rtdetr_r50vd_coco": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth",
    "rtdetr_r101vd_coco": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth",
    "rtdetr_r18vd_coco_objects365": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth",
    "rtdetr_r50vd_coco_objects365": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth",
    "rtdetr_r101vd_coco_objects365": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth",
}

OPTIMIZERS = ["Adam", "AdamW", "SGD"]

# endregion
# region envvars
team_id = sly.env.team_id()
wotkspace_id = sly.env.workspace_id()
project_id = sly.env.project_id()

# endregion
api = sly.Api.from_env()

# region state
selected_project_id = None
selected_project_info = None
selected_project_meta = None
project_dir = None
project = None
converted_project = None
train_dataset_path = None
val_dataset_path = None
custom_config_path = None
train_mode = None
selected_classes = None
splits = None
# endregion
