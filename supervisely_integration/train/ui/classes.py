from typing import Optional

import supervisely as sly
from supervisely.app.widgets import Button, Card, ClassesListSelector, Container, Text

import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.parameters as parameters
import supervisely_integration.train.ui.splits as splits

empty_notification = Text("Please, select at least one class.", status="warning")
train_classes_selector = ClassesListSelector(multiple=True, empty_notification=empty_notification)
train_classes_selector.hide()

select_classes_button = Button("Select classes")
change_classes_button = Button("Change classes")
change_classes_button.hide()

card = Card(
    title="Select classes",
    description="Select classes to train the model on",
    content=Container([train_classes_selector, select_classes_button]),
    content_top_right=change_classes_button,
    lock_message="Click on the Change classes button to select other classes",
)
card.lock()


def fill_classes_selector(clear: Optional[bool] = False):
    if not clear:
        train_classes_selector.set(g.project_meta.obj_classes)
        train_classes_selector.select_all()
        train_classes_selector.show()
    else:
        train_classes_selector.set([])
        train_classes_selector.hide()


@select_classes_button.click
def classes_selected():
    selected_classes = train_classes_selector.get_selected_classes()
    if not selected_classes:
        return

    g.selected_classes = selected_classes

    sly.logger.info(f"Selected classes: {[cls.name for cls in selected_classes]}")

    card.lock()
    change_classes_button.show()

    splits.card.unlock()
    parameters.card.unlock()

    g.update_step()


@change_classes_button.click
def change_classes():
    g.selected_classes = None

    card.unlock()
    change_classes_button.hide()

    splits.card.lock()
    parameters.card.lock()

    g.update_step(back=True)
