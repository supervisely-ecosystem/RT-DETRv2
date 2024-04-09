from supervisely.app.widgets import Card
from supervisely.app.widgets.augmentations.augmentations import AugmentationsWithTabsNew, Editor

import supervisely_integration.train.globals as g

# TODO: Change to real custom augs template
custom_augs = "custom-augs-template: 100"

content = []
additional_tabs = [
    AugmentationsWithTabsNew.Tab(
        tab_title="Custom",
        content_title="Custom augmentations",
        content_description="Use Editor to write custom augmentations",
        content=content,
        tab_description="Custom augmentations",
    )
]


augmentations = AugmentationsWithTabsNew(
    g.api, g.project_id, templates=g.augs, task_type="detection", additional_tabs=additional_tabs
)
custom_editor: Editor = augmentations.get_editor("Custom")
custom_editor.set_text(custom_augs)

card = Card(
    title="Augmentations",
    description="Select augmentations for training",
    content=augmentations,
)
