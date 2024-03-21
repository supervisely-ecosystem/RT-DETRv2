from typing import Optional

import supervisely as sly
from supervisely.app.widgets import Button, Card, Container, ReloadableArea, TrainValSplits

import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.parameters as parameters

trainval_container = Container()
trainval_area = ReloadableArea(trainval_container)

select_button = Button("Select splits")
change_button = Button("Change splits")
change_button.hide()

card = Card(
    title="4️⃣ Select splits",
    description="Select splits for training and validation",
    collapsable=True,
    content=Container([trainval_area, select_button]),
    content_top_right=change_button,
    lock_message="Click on the Change splits button to select other splits",
)
card.lock()
card.collapse()


def init_splits(project_id: Optional[int] = None):
    if not project_id:
        trainval_container._widgets.clear()
    else:
        trainval_splits = TrainValSplits(project_id)
        trainval_container._widgets.append(trainval_splits)
    trainval_area.reload()


@select_button.click
def splits_selected():
    g.splits = trainval_container._widgets[0].get_splits()

    sly.logger.info(f"Selected splits: {g.splits}")

    card.lock()
    card.collapse()
    change_button.show()

    parameters.card.unlock()
    parameters.card.uncollapse()


@change_button.click
def change_splits():
    g.splits = None

    sly.logger.debug("Splits reset.")

    card.unlock()
    card.collapse()
    change_button.hide()

    parameters.card.lock()
    parameters.card.collapse()
