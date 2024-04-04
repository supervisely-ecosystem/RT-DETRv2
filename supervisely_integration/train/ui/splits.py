from supervisely.app.widgets import Button, Card, Container, TrainValSplits

import supervisely_integration.train.globals as g

trainval_splits = TrainValSplits(g.project_id)
select_button = Button("Select splits")
change_button = Button("Change splits")
change_button.hide()

card = Card(
    title="Select splits",
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

    g.update_step()


@change_button.click
def change_splits():
    card.unlock()
    change_button.hide()

    g.update_step(back=True)
