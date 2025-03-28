from typing import List, Union
from dataclasses import dataclass
import supervisely as sly
import numpy as np


@dataclass
class Prediction:
    annotation: sly.Annotation
    source = "images/image001.png"  # path | url | team_files_url
    image: np.ndarray = None
    image_info: sly.ImageInfo = None
    # image_id = 121
    # project_id = 234
    # dataset_id = 555


class PredictionSession:
    def __iter__():
    def __len__() -> int:
    def is_done() -> bool:
    def next(timeout=None, block=True) -> Prediction:
    def stop():
    def status() -> dict:  # progress, message, error traceback, context (project_id, dataset_id ...)
        pass


def predict(
    images="image.png",  # local paths, directory, local project, np.array, PIL.Image, url, team_files_url
    video="video.mp4",
    model_params={"conf": 0.55},
    project_id=123,
    dataset_id=456,
    upload_mode=None,  # None, append, replace, create, iou_merge (only bbox/mask)
    recursive=True,
) -> Union[Prediction, List[Prediction]]:
    pass

def predict_detached() -> PredictionSession:
    pass

x = predict("image.png")
x.annotation