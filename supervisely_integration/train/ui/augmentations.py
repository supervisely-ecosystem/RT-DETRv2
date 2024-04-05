from supervisely.app.widgets import Card
from supervisely.app.widgets.augmentations.augmentations import AugmentationsWithTabsNew

import supervisely_integration.train.globals as g

augmentations = AugmentationsWithTabsNew(
    g.api, g.project_id, templates=g.augs, task_type="detection"
)

card = Card(
    title="Augmentations",
    description="Select augmentations for training",
    content=augmentations,
)
