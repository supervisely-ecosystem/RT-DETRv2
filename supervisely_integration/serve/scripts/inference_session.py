import supervisely as sly
from supervisely.nn.inference import RuntimeType, SessionJSON

api = sly.Api.from_env()

host = "0.0.0.0"
port = 8000
session_url = f"http://{host}:{port}"

session = SessionJSON(api, session_url=session_url)

image_id = 112346
session.inference_image_id(image_id, True)
