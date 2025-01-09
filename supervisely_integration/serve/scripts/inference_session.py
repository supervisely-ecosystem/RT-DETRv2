from os.path import expanduser

from dotenv import load_dotenv

import supervisely as sly
from supervisely.nn.inference import SessionJSON

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(expanduser("~/supervisely.env"))

api = sly.Api.from_env()

host = "0.0.0.0"
port = 8000
session_url = f"http://{host}:{port}"

session = SessionJSON(api, session_url=session_url)

image_id = 113848
session.inference_image_id(image_id, True)
