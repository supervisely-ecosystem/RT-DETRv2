from supervisely.app.widgets import Card, TrainValSplits

import supervisely_integration.train.globals as g

trainval_splits = TrainValSplits(g.project_id)


card = Card(
    title="4️⃣ Select splits",
    description="Select splits for training and validation",
    collapsable=True,
    content=trainval_splits,
    lock_message="Click on the Change splits button to select other splits",
)
card.lock()
