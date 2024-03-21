import os

import supervisely as sly
from dotenv import load_dotenv

if sly.is_development:
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

# region constants
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
        "NAME1",
        "METHOD1",
        "DATASET1",
        "INFERENCETIME1",
        "TRAININGEPOCHS1",
        "TRAININGMEMORY1",
        "BOXAP1",
    ],
    [
        "NAME2",
        "METHOD2",
        "DATASET2",
        "INFERENCETIME2",
        "TRAININGEPOCHS2",
        "TRAININGMEMORY2",
        "BOXAP2",
    ],
]

# endregion
# region envvars
team_id = sly.env.team_id()
wotkspace_id = sly.env.workspace_id()
project_id = sly.env.project_id()

# endregion
api = sly.Api.from_env()

# region state
selected_project_info = None
train_mode = None
# endregion
