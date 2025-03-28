# Inference

## Predict signature

```python
def predict(
    images="image.png",  # local paths, directory, local project, np.array, PIL.Image, url, team_files_url
    video="video.mp4",
    model_params={"conf": 0.55},
    project_id=123,
    dataset_id=456,
    upload_mode=None,  # None, append, replace, create, iou_merge (only bbox/mask)
    recursive=True,
    classes=["person", "car"],  # list of classes to predict
) -> Prediction | List[Prediction]:
    pass

def predict_detached(
    images="image.png",  # local paths, directory, local project, np.array, PIL.Image, url, team_files_url
    video="video.mp4",
    model_params={"conf": 0.55},
    project_id=123,
    dataset_id=456,
    upload_mode=None,  # None, append, replace, create, iou_merge (only bbox/mask)
    recursive=True,
    classes=None,  # list of classes to predict
) -> PredictionSession:
    pass
```

## Prediction result

```python
class Prediction:
    annotation: sly.Annotation
    source = "images/image001.png"  # path | url | team_files_url
    image: np.ndarray = None
    image_info: sly.ImageInfo = None
    # image_id = 121
    # project_id = 234
    # dataset_id = 555
```

## Stream

```python
class PredictionSession:
    def __iter__():
    def __len__() -> int:
    def is_done() -> bool:
    def next(timeout=None, block=True) -> Prediction:  # 🔴 подумать над именем: fetch, pop, next, consume
    def stop():
    def status():  progress, message, error traceback, context (project_id, dataset_id ...)
    def pause() ???
    def resume() ???
```


## Example

```python
# Connect
model = api.nn.connect(task_id)

# Predict
prediction = model.predict(
    images="image.png",
    model_params={"conf": 0.55},
)

# Detached mode
session = model.predict_detached(
    project_id=123,
    model_params={"conf": 0.55},
)
for prediction in session:
    pass
```

## ModelAPI

```python
class ModelAPI:
    url: str
    task_id: int
    api: sly.Api = None

    def __init__(self, url=None, task_id=None, api=None):
        pass

    def predict(
        images="image.png",  # local paths, directory, local project, np.array, PIL.Image, url, team_files_url
        video="video.mp4",
        model_params={"conf": 0.55},
        project_id=123,
        dataset_id=456,
        upload_mode=None,  # None, append, replace, create, iou_merge (only bbox/mask)
        recursive=True,
        classes=None,  # list of classes to predict
    ) -> Prediction | List[Prediction]:
        pass

    def predict_detached(
        images="image.png",  # local paths, directory, local project, np.array, PIL.Image, url, team_files_url
        video="video.mp4",
        model_params={"conf": 0.55},
        project_id=123,
        dataset_id=456,
        upload_mode=None,  # None, append, replace, create, iou_merge (only bbox/mask)
        recursive=True,
        classes=None,  # list of classes to predict
    ) -> PredictionSession:
        pass

    def default_model_params(self) -> dict:
        pass

    def model_classes(self) -> List[str]:
        pass

    def model_meta(self) -> sly.ProjectMeta:
        pass

    def model_info(self) -> dict:
        pass

    def load_checkpoint(self) -> dict:
        pass

    def shutdown_model(self) -> None:
        pass

```


## Class Hierarchy

```python
class ModelAPI:
    pass

class ModelAPI3D(ModelAPI):
    def predict(
        pcd="pointcloud.pcd",
        ...
    )

class ModelAPITracking(ModelAPI):
    def predict(
        video="video.mp4",
        tracking_method="botsort",
    )
```