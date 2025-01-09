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

# Local development
# PYTHONPATH="${PWD}:${PYTHONPATH}" \
# python ./supervisely_integration/serve/main.py \
# --model ./my_experiments/2315_RT-DETRv2/checkpoints/best.pth

# Docker deployment
# docker run \
#   --shm-size=1g \
#   --runtime=nvidia \
#   --cap-add NET_ADMIN \
#   --env-file ~/supervisely.env \
#   --env ENV=production \
#   --env LOCAL_DEPLOY=True \
#   -v ".:/app" \
#   -w /app \
#   -p 8000:8000 \
#   supervisely/rt-detrv2-gpu-cloud:1.0.3 \
#   --model "RT-DETRv2-S"
