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

# docker run \
#   --shm-size=1g \
#   --runtime=nvidia \
#   --cap-add NET_ADMIN \
#   --env-file ~/supervisely.env \
#   --env ENV=production \
#   -v "./RT-DETRv2:/app" \
#   -w /app \
#   supervisely/rt-detrv2:1.0.3 \
#   python3 -m uvicorn main:model.app --app-dir /app/RT-DETRv2/supervisely_integration/serve --host 0.0.0.0 --port 8000 --ws websockets -- \
#   --model "RT-DETRv2-S" --predict "data/test" --output-dataset "Predictions/ds1" --workspace_id "349"
