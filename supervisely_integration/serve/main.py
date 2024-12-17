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
#   --env SLY_APP_DATA_DIR=./app_data \
#   --env TASK_ID=55555 \
#   --env TEAM_ID=8 \
#   --env WORKSPACE_ID=349 \
#   --env LOCAL_DEPLOY=True \
#   -v ".:/app" \
#   -v "supervisely:/app/supervisely" \
#   -w /app \
#   supervisely/rt-detrv2:1.0.3 \
#   python3 /app/supervisely_integration/serve/main.py --model "RT-DETRv2-S"
#   python3 supervisely_integration/serve/main.py --model "RT-DETRv2-S"
