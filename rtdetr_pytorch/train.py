from typing import Callable, Optional

import torch
from src.core import YAMLConfig
from src.misc.sly_logger import LOGS, Logs
from src.solver import DetSolver

import supervisely as sly
from supervisely.app.widgets import Progress


def train(
    finetune: bool,
    config_path: str,
    progress_bar_epochs: Progress,
    progress_bar_iters: Progress,
):
    cfg = YAMLConfig(
        config_path,
        tuning=finetune,
    )
    solver = DetSolver(cfg)
    solver.fit(progress_bar_epochs, progress_bar_iters)

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
    sly.logger.debug("Callbacks set...")
