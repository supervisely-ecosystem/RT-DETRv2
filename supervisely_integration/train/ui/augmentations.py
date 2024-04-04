from supervisely.app.widgets import AugmentationsWithTabs, Card, Container

import supervisely_integration.train.globals as g

augmentations = AugmentationsWithTabs(g, templates=g.augs, task_type="detection")

card = Card(
    title="Augmentations",
    description="Select augmentations for training",
    content=augmentations,
)
