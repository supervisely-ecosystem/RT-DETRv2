import supervisely as sly
from supervisely.app.widgets import Container

import supervisely_integration.train.ui.classes as classes
import supervisely_integration.train.ui.input as input
import supervisely_integration.train.ui.model as model
import supervisely_integration.train.ui.output as output
import supervisely_integration.train.ui.parameters as parameters
import supervisely_integration.train.ui.splits as splits

layout = Container(
    [input.card, model.card, classes.card, splits.card, parameters.card, output.card]
)

app = sly.Application(layout=layout)
