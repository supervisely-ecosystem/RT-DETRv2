from typing import Dict

from supervisely.app.widgets import Card, Container, FolderThumbnail, LineChart, Progress

import rtdetr_pytorch.train as train_cli

# TODO: Fix import, now it's causing error
# from rtdetr_pytorch.src.misc.sly_logger import Logs

loss = LineChart("Loss", series=[{"name": "Loss", "data": []}])
learning_rate = LineChart(
    "Learning Rate",
    series=[
        {"name": "lr0", "data": []},
        {"name": "lr1", "data": []},
        {"name": "lr2", "data": []},
        {"name": "lr3", "data": []},
    ],
)
cuda_memory = LineChart("CUDA Memory", series=[{"name": "Memory", "data": []}])
iter_container = Container([loss, learning_rate, cuda_memory])

validation_metrics = LineChart(
    "Validation Metrics",
    series=[
        {"name": "AP@IoU=0.50:0.95|maxDets=100", "data": []},
        {"name": "AP@IoU=0.50|maxDets=100", "data": []},
        {"name": "AP@IoU=0.75|maxDets=100", "data": []},
        {"name": "AR@IoU=0.50:0.95|maxDets=1", "data": []},
        {"name": "AR@IoU=0.50:0.95|maxDets=10", "data": []},
        {"name": "AR@IoU=0.50:0.95|maxDets=100", "data": []},
    ],
)

train_progress = Progress(hide_on_finish=False)

output_folder = FolderThumbnail()
output_folder.hide()


def iter_callback(logs):
    iter_idx = logs.iter_idx
    loss.add_to_series("Loss", (iter_idx, logs.loss))
    add_lrs(iter_idx, logs.lrs)
    cuda_memory.add_to_series("Memory", (iter_idx, logs.cuda_memory))


def eval_callback(logs):
    add_metrics(logs.epoch, logs.evaluation_metrics)


def add_lrs(iter_idx: int, lrs: Dict[str, float]):
    for series_name, lr in lrs.items():
        learning_rate.add_to_series(series_name, (iter_idx, lr))


def add_metrics(epoch: int, metrics: Dict[str, float]):
    for series_name, metric in metrics.items():
        if series_name.startswith("per_class"):
            continue
        validation_metrics.add_to_series(series_name, (epoch, metric))


train_cli.setup_callbacks(iter_callback=iter_callback, eval_callback=eval_callback)

card = Card(
    title="Download the weights",
    description="Here you can download the weights of the trained model",
    content=Container([train_progress, output_folder, iter_container, validation_metrics]),
)
card.lock()
