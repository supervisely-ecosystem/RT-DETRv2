import os

from dotenv import load_dotenv

import supervisely as sly

load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("local.env")

api: sly.Api = sly.Api.from_env()

task_id = 68843  # <---- Change this to your task_id
method = "deploy_from_api"


data = {"deploy_params": {}}
api.app.send_request(task_id, method, data)
