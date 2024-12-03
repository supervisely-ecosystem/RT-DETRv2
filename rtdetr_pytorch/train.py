from typing import Callable, Optional

import torch
from src.core import YAMLConfig
from src.misc.sly_logger import LOGS, Logs
from src.solver import DetSolver

import supervisely as sly
from supervisely.app.widgets import Progress
from supervisely.nn.training.train_app import TrainApp


def train(train: TrainApp, config_path: str):

    path_to_model = train.model_files["checkpoint"]
    cfg = YAMLConfig(
        config_path,
        tuning=path_to_model,
    )
    solver = DetSolver(cfg)
    best_checkpoint_path = solver.fit()

    return cfg, best_checkpoint_path


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
