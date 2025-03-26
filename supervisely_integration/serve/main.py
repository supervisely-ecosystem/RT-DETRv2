import os

from dotenv import load_dotenv

import supervisely as sly
from supervisely_integration.serve.rtdetrv2 import RTDETRv2

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

model = RTDETRv2(
    use_gui=True,
    use_serving_gui_template=True,
)
model.serve()
