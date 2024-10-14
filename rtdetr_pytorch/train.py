import os
from typing import Callable, Optional

import supervisely as sly
import torch
from checkpoints import checkpoints
from src.core import YAMLConfig
from src.misc.sly_logger import LOGS, Logs
from src.solver import DetSolver


def train(
    model: str,
    finetune: bool,
    config_path: str,
    progress: sly.app.widgets.Progress,
):

    if finetune:
        checkpoint_url = checkpoints[model]
        name = os.path.basename(checkpoint_url)
        checkpoint_path = f"models/{name}"
        if not os.path.exists(checkpoint_path):
            os.makedirs("models", exist_ok=True)
            torch.hub.download_url_to_file(checkpoint_url, checkpoint_path)
        tuning = checkpoint_path
    else:
        tuning = ""

    cfg = YAMLConfig(
        config_path,
        # resume='',
        tuning=tuning,
    )

    solver = DetSolver(cfg)
    solver.fit()
    # solver.fit(progress)

    return cfg


def setup_callbacks(
    iter_callback: Optional[Callable] = None, eval_callback: Optional[Callable] = None
):

    sly.logger.debug("Setting up callbacks...")

    def print_iter(logs: Logs):
        print("ITER | Iter IDX: ", logs.iter_idx)
        print("ITER | Loss, lrs, memory: ", logs.loss, logs.lrs, logs.cuda_memory)

    def print_eval(logs: Logs):
        # Charts: AP vs AR (maxDets=100), All APs, All ARs
        print("EVAL | Epoch: ", logs.epoch)
        print("EVAL | Metrics: ", logs.evaluation_metrics)

    if iter_callback is None:
        sly.logger.info("iter callback not provided, using default prints...")
        iter_callback = print_iter
    if eval_callback is None:
        sly.logger.info("eval callback not provided, using default prints...")
        eval_callback = print_eval

    LOGS.iter_callback = iter_callback
    LOGS.eval_callback = eval_callback
    sly.logger.debug("Callbacks set...")
