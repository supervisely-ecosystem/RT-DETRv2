# Inference

🔴 Add sliding window (sahi)

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
) -> List[Prediction]:
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
    source: Any = "images/image001.png"  # path | url | ... or None
    project_id = 123
    dataset_id = 456
    image_id = 12345
    def load_image(): np.ndarray = None
    def draw(): np.ndarray = None
```

## Inference Stream

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
        input=None,  # local paths, directory, local project, np.array, PIL.Image, url, team_files_url
        settings=None,
        project_id=None,
        dataset_id=None,
        image_ids=None,
        batch_size=None,
        img_size=None,  # int for square resizing or a (height, width) tuple. Default: using model's default input size
        classes=None,  # list of classes to predict (List[str])
        upload=None,  # None, append, replace, create, iou_merge (only bbox/mask)
        recursive=False,
    ) -> List[Prediction]:
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

    def get_default_settings(self) -> dict:
        pass

    def get_classes(self) -> List[str]:
        pass

    def get_meta(self) -> sly.ProjectMeta:
        pass

    def get_info(self) -> dict:
        pass

    def load(self, model=None, runtime=None, ...):  # remote and local
        pass

    def stop(self):
        pass

    def shutdown(self):
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

class PromptableModelAPI(ModelAPI):
    def predict(
        images="image.png",
        prompt="a cat",
        bbox=...
        mask=...
        points=...
    )
```