import supervisely as sly
from supervisely.app.widgets import Button, Card, Container, TrainValSplits

import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.augmentations as augmentations

trainval_splits = TrainValSplits(g.PROJECT_ID)
select_button = Button("Select splits")
change_button = Button("Change splits")
change_button.hide()

card = Card(
    title="Train / Validation splits",
    description="Select splits for training and validation",
    content=Container([trainval_splits, select_button]),
    content_top_right=change_button,
    lock_message="Click on the Change splits button to select other splits",
)
card.lock()


@select_button.click
def splits_selected():
    card.lock()
    change_button.show()
    # g.splits = trainval_splits.get_splits()
    augmentations.card.unlock()
    g.update_step(step=5)


@change_button.click
def change_splits():
    card.unlock()
    augmentations.card.lock()
    change_button.hide()

    g.update_step(back=True)


def dump_train_val_splits(project_dir):
    # splits._project_id = None
    trainval_splits._project_fs = sly.Project(project_dir, sly.OpenMode.READ)
    g.splits = trainval_splits.get_splits()
    train_split, val_split = g.splits
    app_dir = g.data_dir  # @TODO: change to sly.app.get_synced_data_dir()?
    sly.json.dump_json_file(train_split, f"{app_dir}/train_split.json")
    sly.json.dump_json_file(val_split, f"{app_dir}/val_split.json")
