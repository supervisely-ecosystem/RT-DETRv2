import os
import sys

import supervisely as sly
from dotenv import load_dotenv

from rtdetr_pytorch.model_list import _models

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
    "Dataset",
    "AP_Val",
    "Params(M)",
    "FRPS(T4)",
]
PRETRAINED_MODELS = [
    [value for key, value in model_info.items() if key != "meta"] for model_info in _models
]
OPTIMIZERS = ["Adam", "AdamW", "SGD"]
SCHEDULERS = [
    "Without scheduler",
    "StepLR",
    "MultiStepLR",
    "ExponentialLR",
    "ReduceLROnPlateauLR",
    "CosineAnnealingLR",
    "CosineRestartLR",
]

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
widgets = None
# endregion
