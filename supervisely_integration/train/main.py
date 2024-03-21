import supervisely as sly
from supervisely.app.widgets import Container

import supervisely_integration.train.ui.input as input
import supervisely_integration.train.ui.model as model

layout = Container([input.card, model.card])

app = sly.Application(layout=layout)
