# import os
# import sys
# from typing import Optional

# from dotenv import load_dotenv

# import supervisely as sly
# from rtdetr_pytorch.model_list import _models
# from supervisely.nn.artifacts.rtdetr import RTDETR

# if sly.is_development:
#     load_dotenv("local.env")
#     load_dotenv(os.path.expanduser("~/supervisely.env"))

# TASK_ID = sly.env.task_id()
# TEAM_ID = sly.env.team_id()
# WORKSPACE_ID = sly.env.workspace_id()
# PROJECT_ID = sly.env.project_id()

# rtdetr_artifacts = RTDETR(TEAM_ID)

# # endregion
# api = sly.Api.from_env()
# augs = []
# # region state
# team = api.team.get_info_by_id(TEAM_ID)
