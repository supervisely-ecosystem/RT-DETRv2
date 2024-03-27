from collections import namedtuple
from typing import Dict, List, Optional

import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    Field,
    FileViewer,
    RadioTable,
    RadioTabs,
)

import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.classes as classes

TrainMode = namedtuple("TrainMode", ["pretrained", "custom", "finetune"])


def get_file_tree(api: sly.Api, team_id: int, path: Optional[str] = "/") -> List[Dict]:
    files = api.file.list(team_id, path)
    tree_items = []
    for file in files:
        path = file["path"]
        tree_items.append({"path": path})
    return tree_items


select_custom_weights = FileViewer(get_file_tree(g.api, g.team_id), selection_type="file")
pretrained_models_table = RadioTable(columns=g.TABLE_COLUMNS, rows=g.PRETRAINED_MODELS)

finetune_checkbox = Checkbox("Fine-tune", True)
finetune_field = Field(
    finetune_checkbox,
    title="Enable fine-tuning",
    description="Fine-tuning allows you to continue training a model from a checkpoint.",
)

select_model_button = Button("Select model")
change_model_button = Button("Change model")
change_model_button.hide()

model_mode = RadioTabs(
    g.MODEL_MODES.copy(),
    contents=[
        Container([pretrained_models_table, finetune_field]),
        select_custom_weights,
    ],
)

card = Card(
    title="2️⃣ Select a model",
    description="Select a model to train",
    collapsable=True,
    content=Container([model_mode, select_model_button]),
    content_top_right=change_model_button,
    lock_message="Click on the Change model button to select another model",
)
card.lock()


@select_model_button.click
def model_selected():
    mode = model_mode.get_active_tab()
    if mode == g.MODEL_MODES[0]:
        pretrained: List[str] = pretrained_models_table.get_selected_row()
        custom = None
        sly.logger.debug(f"Selected mode: {mode}, selected pretrained model: {pretrained}")
    else:
        pretrained = None
        custom: List[str] = select_custom_weights.get_selected_items()
        # TODO: Add single-item mode to the widget and remove indexing
        custom = custom[0] if custom else None
        sly.logger.debug(f"Selected mode: {mode}, path to custom weights: {custom}")
    finetune = finetune_checkbox.is_checked()
    g.train_mode = TrainMode(pretrained, custom, finetune)

    card.lock()
    change_model_button.show()

    classes.fill_classes_selector()
    classes.card.unlock()


@change_model_button.click
def change_model():
    g.train_mode = None
    card.unlock()
    change_model_button.hide()

    classes.fill_classes_selector(clear=True)
    classes.card.lock()
