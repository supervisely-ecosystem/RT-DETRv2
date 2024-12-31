# This module contains functions that are used to configure the input and output of the workflow for the current app.
from os.path import join

import supervisely as sly
from supervisely.io.fs import get_file_name_with_ext
from supervisely.nn.utils import ModelSource


def workflow_input(api: sly.Api, model_files: dict, model_info: dict, model_source: str):
    try:
        if model_source == ModelSource.PRETRAINED:
            checkpoint_url = model_info["meta"]["model_files"]["checkpoint"]
            checkpoint_name = model_info["meta"]["model_name"]
        elif model_source == ModelSource.CUSTOM:
            checkpoint_name = get_file_name_with_ext(model_files["checkpoint"])
            checkpoint_url = join(model_info["artifacts_dir"], "checkpoints", checkpoint_name)

        model_name = "RT-DETRv2"
        meta = sly.WorkflowMeta(node_settings=sly.WorkflowSettings(title=f"Serve {model_name}"))

        sly.logger.debug(
            f"Workflow Input: Checkpoint URL - {checkpoint_url}, Checkpoint Name - {checkpoint_name}"
        )
        if checkpoint_url and api.file.exists(sly.env.team_id(), checkpoint_url):
            api.app.workflow.add_input_file(checkpoint_url, model_weight=True, meta=meta)
        else:
            sly.logger.debug(
                f"Checkpoint {checkpoint_url} not found in Team Files. Cannot set workflow input"
            )
    except Exception as e:
        sly.logger.debug(f"Failed to add input to the workflow: {repr(e)}")


def workflow_output(self):
    raise NotImplementedError("add_output is not implemented in this workflow")
