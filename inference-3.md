# Demo: Inference & Deployment

**Table of Contents (only for this readme):**

- [Deploy model using API](#deploy-model-using-api)
- [Using Model Outside of Supervisely Platform](#using-model-outside-of-supervisely-platform)
  - [Deploy model using Docker](#deploy-model-using-docker)
  - [Deploy model using Python](#deploy-model-using-python)
- [Predict using SDK](#predict-using-sdk)
- [Predict using docker](#predict-using-docker)
- [Predict using API](#predict-using-api)

## Deploy

If you don't have available servers, you can follow the instructions from this guide to connect your machine to Supervisely

### Deploy model manually in Supervisely Web Interface

### Deploy model using API

Initialize API client:

```python
import os
import supervisely as sly
from dotenv import load_dotenv

# Ensure you've set API_TOKEN and SERVER_ADDRESS environment variables.
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()
```

🔴
team_id - в контекст API объекта / ENV засунуть чтобы не передавать всегда?
team_id - можно ли понять активную тиму?
еще можно подумать, что если всего одна команда то мы сами найдем файл
и даже если несколько команд то мы тоже можем найти команду и файл сами
🔴

* With path to the model checkpoint:
    ```python
    model = api.nn.deploy.custom(
        team_id=777, # optional + auto + regex
        checkpoint="/experiments/path/to/checkpoint.pth" # поддердивать file path и file url
    )
    ```

* With train task id:
    ```python
    model = api.nn.deploy.custom(
        train_id=123,
        checkpoint= # если не указан, то берем best / last / name.pht / path - можно еще имя или путь
    )
    ```

* Or you can deploy pretrained model:
    ```python
    model = api.nn.deploy.pretrained(framework="RT-DETRv2", model_name="RT-DETRv2-S")
    ```

> For more information, see [Deploy & Predict with Supervisely SDK](https://docs.supervisely.com/neural-networks/overview-1/deploy_and_predict_with_supervisely_sdk).

### Using Model Outside of Supervisely Platform

To use the model outside of the Supervisely platform, you will first need to download the model files.
You can download the entire folder containing your model files and checkpoints, or download only the necessary files:

![Download checkpoint from Team Files](img/team_files_download_2.png)

For example, create a folder `models` on your machine and place there downloaded files. This folder will be used in the next steps.

#### Deploy model using Docker. 
    
Deploying in a Docker Container is a convenient way to run a model without needing to install dependencies on your machine. You can use our pre-built docker image with the model implementation.

```bash
docker run \
--runtime=nvidia \
-v "./models:/models" \
-v "./data:/input" \
-v "./predictions:/output" \
-p 8080:8000 \
supervisely/rt-detrv2-app:latest \
deploy \
--model "/f.pth" \
```

Here is the description of the parameters:
* -v "./models:/models" - mount the "models" directory as model directory inside the container.
* -v "./data:/input" - mount the "data" directory as input directory inside the container. When predicting, you will be able to specify the path to the image or directory you want to process, relative to the "data" directory.
* -v "./predictions:/output" - mount the "predictions" directory as output directory inside the container.
* -p 8080:8000 - expose the port 8000 of the container to the port 8080 of your machine. The model will be available at http://localhost:8080
* supervisely/rt-detrv2-app:latest - the name and the version of the docker image. You can specify older versions if needed.
* deploy - the command to deploy the model.

Optionally, you can add the following parameters to the command:
```bash
--device cuda
```

```python

model = api.nn.connect(url="http://localhost:8000")

```

#### Deploy model using Python
In this variant, you will deploy a model locally as an API Server with the help of Supervisely SDK. The server will be ready to process API request for inference. It allows you to predict with local images, folders, videos, or remote supervisely projects and datasets (if you provide your Supervisely API token).

```python

```

## Predict

### Predict using SDK

You can use the Supervisely SDK to predict images, videos, directories, datasets, and projects using the deployed model. See [Deploy](#deploy) section to learn how to deploy the model.

The class used to get predictions is `ModelApi`. This class gives you remote access to the model. You can get information about the model, predict and stop the model.

```python
class ModelApi:
    def predcit():
    def stop():
    def shutdown():
    def get_info():
    def healthcheck():
```

If the model was deployed using Supervisely SDK as shown in the [Deploy model using API](#deploy-model-using-api) section, you already have the model object. Otherwise, you can create it using the following code:

Initialize API client:
```python
import os
import supervisely as sly
from dotenv import load_dotenv

# Ensure you've set API_TOKEN and SERVER_ADDRESS environment variables.
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()
```

Select from the list of deployed models:
```python
framework = "RT-DETRv2" # Opitonal
models = api.nn.get_running_models(framework=framework)
model = models[0]
```

Connect to the model by id:
```python
model_id = 123
model = api.nn.connect(model_id)
```

Or connect to the model by url if you deployed it as an API server as shown in the [Using Model Outside of Supervisely Platform](#using-model-outside-of-supervisely-platform) section:
```python
url = "localhost:8080"
model = api.nn.connect(url)
```

The method that is used to predict is `predict`. It allows you to predict images, videos, directories, datasets, and projects. The method returns a list of objectes of class `Prediciton`. This method is optimized and allows you to predict efficiently even the largest datasets.
You can run the inferenece in detached mode, in that case the method returns an iterable class 🔴`InferenceSession` that yields predictions as they are being received from the model.
You can stop the prediction process at any time by calling the `stop` method of the `InferenceSession` object.

```python
class Prediction:
    source: Union[str, int]
    annotation: sly.Annotation
    image_id: Optional[int]
    image_name: Optional[str]
    dataset_id: Optional[int]
    frame_index: Optional[int]
```

```python
class InferenceSession:
    def done() -> bool:
    def get(timeout: Optional[int] = None) -> Prediction:
    def get_nowait() -> Union[Prediction, None]:
    def stop():
```

Get predictions
```pytgon
annotations = model.predict(source, params)
```

Get predictions in detached mode. In this simple example we will get predictions for 10 seconds and then stop the session.
```python
with model.predict(source, params, detached=True) as session:
  predictions = []
  timeout = 10
  t = time.monotonic()
  while time.monotonic() - t < timeout:
      if session.done():
          break
      # prediction = session.get(timeout=10) # blocking
      prediction = session.get_nowait() # non blocking call
      if prediction is not None:
          predictions.append(prediction)
  session.stop()
  print(f"Predicted {len(predictions)} images in {timeout} seconds")
```

#### Predict images

Single Image:
```python
# np.array
image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
# str path to image
image = "/a/b/c.jpg"
# str url to image file in team files
image = "https://a/b/c.jpg"
# int image id
image = 111

params = {"confidence_threshold": 0.5} # Optional
prediction: Prediction = model.predict(image=image, params=params)
```

Multiple Images:
```python
# np.array
images = [np.array(Image.open(image_path).convert("RGB")) for image_path in IMAGE_PATHS]
# str path to image
images = ["/a/b/c.jpg", "/a/b/d.jpg"]
# str url to image file in team files
images = ["https:supervisely.com/files/123/a/b/c.jpg", "https://supervisely.com/files/123/a/b/d.jpg"]
# int image id
images = [111, 222]

# Processing images one by one
params = {"confidence_threshold": 0.5} # Optional
for image in images:
    prediction: Prediction = model.predict(image=image, params=params)
    json.dump(prediction.annotation.to_json(), open(f"{image}.json", "w"))

# Processing images in batch
predictions: List[Prediction] = model.predict(image=images, params=params)
json.dump([prediction.annotation.to_json() for prediction in predictions], open("predictions.json", "w"))

🔴 # if predict yields annotations
# for image, annotation in zip(images, model.predict(image=images, params=params)):
#     api.annotation.upload(image, annotation) 

# Processing in detached mode
with model.predict(image=images, params=params, detached=True) as session:
    for prediction in session:
        api.annotation.upload(prediction.source, prediction.annotation)
        
for prediction in model.predict(image=images, params=params):
    json.dump(prediction.annotation.to_json(), open(f"{prediction.source}.json", "w"))
```

#### Predict video

```python
# str path to video
video = "/a/b/c.mp4"
# str url to video file in team files
video = "https://a/b/c.mp4"
# int video id
video = 111

params = {"confidence_threshold": 0.5} # Optional

predictions: List[Prediction] = model.predict(video=video, params=params)

# Processing video in detached mode
with model.predict(video=video, params=params, detached=True) as session:
    for prediction in session:
        json.dump(prediction.annotation.to_json(), open(f"{prediction.source}_{prediction.frame_index}.json", "w"))
```

After you got predictions for video frames, you can apply tracking algorithm. You can use the following code:
```python
from supervisely.nn.tracking import track
import boxmot

device = "cuda:0"
tracker = boxmot.BotSort(reid_weights=Path('osnet_x0_25_msmt17.pt'), device=device, half=False)
video_ann: VideoAnnotation = track(tracker, predictions, video)
```

#### Predict directory

```python
# str path to directory with images or videos
directory = "/a/b/c"
# str url to directory with images or videos in team files
directory = "https://a/b/c"

params = {"confidence_threshold": 0.5} # Optional
recursive = True # Optional. Default is False. If True, the model will predict images in subdirectories

predictions: List[Prediction] = model.predict(dir=dir, params=params, recursive=True)

# Processing in detached mode
with model.predict(dir=dir, params=params, recursive=True, detached=True) as session:
    for prediction in session:
        json.dump(prediction.annotation.to_json(), open(f"{prediction.source}.json", "w"))
```

#### Predict dataset

```python
# int dataset id
dataset = 123

params = {"confidence_threshold": 0.5} # Optional
upload = True # merge replace smart merge # Optional. Default is False. If True, annotations will be uploaded to the dataset
mode: Literal["append", "replace"] = "append" # Optional. Default is "append". If "replace", all annotations in the dataset will be replaced with the new ones.

predictions: List[Prediction] = model.predict(dataset=dataset, upload=upload, mode=mode, params=params)

# Processing in detached mode
with model.predict(dataset=dataset, upload=upload, mode=mode, params=params, detached=True) as session:
    for prediction in session:
        json.dump(prediction.annotation.to_json(), open(f"{prediction.image_name}.json", "w"))
```

simple example
for image in api.dataset():
  ann = mode1.predict(image)

model ansamble - low priority
for image in api.dataset():
  ann = mode1.predict(image)
  for bbox in ann:
    crop
    label = model2.predict(image)
  api.annotation.upload(image, ann, )

#### Predict project

```python
# str path to a project directory
project = "/a/b/c"
# int project id
project = 123

params = {"confidence_threshold": 0.5} # Optional
upload = True # Optional. Default is False. If True, annotations will be uploaded to the project
mode: Literal["append", "replace"] = "append" # Optional. Default is "append". If "replace", all annotations in the project will be replaced with the new ones.

predictions: List[Prediction] = model.predict(project=project, upload=upload, mode=mode, params=params)

# Processing in detached mode
with model.predict(project=project, upload=upload, mode=mode, params=params, detached=True) as session:
    for prediction in queue:
        json.dump(prediction.annotation.to_json(), open(f"{prediction.image_name}.json", "w"))

```

#### More examples

Model ensemble:
```python
with model1.predict(project=project, detached=True) as session1, model2.predict(project=project, detached=True) as session2:
    for prediction1, prediction2 in zip(session1, session2):
        consensus = process_predictions(prediction1, prediction2)
        api.annotation.upload(prediction1.image_id, consensus)
```

Refine predictions with another model:
```python
with detection_model.predict(project=project, detached=True) as session:
    for prediction in session:
        to_refine = []
        labels = prediction.annotation.labels
        for i, label in enumerate(labels):
            conf = get_confidence(label)
            if conf < 0.5:
                to_refine.append(i)
        if to_refine:
            image = api.image.download_np(prediction.image_id)
            crops = [get_crop(image, labels[i]) for i in to_refine]
            refined_predictions = classification_model.predict(crops)
            for i, refined_prediction in zip(to_refine, refined_predictions):
                labels[i] = refined_prediction.annotation.labels[0]
            prediction.annotation = prediction.annotation.clone(labels=labels)
        api.annotation.upload(prediction.image_id, prediction.annotation)
```

Segmentation model + classification model:
```python
with segmentation_model.predict(project=project, detached=True) as seg_session:
    for seg_prediction in seg_session:
        labels = seg_prediction.annotation.labels
        image = api.image.download_np(seg_prediction.image_id)
        crops = []
        for label in labels:
            crops.append(get_crop(image, label))
        classes = classifyer_model.predict(crops)
        labels = [update_class(label, class) for label, class in zip(labels, classes)]
        seg_prediction.annotation = seg_prediction.annotation.clone(labels=labels)
        api.annotation.upload(seg_prediction.image_id, seg_prediction.annotation)
```

### Predict using docker

You will need to provide the path to the input directory, output directory, and the path to the model checkpoint.

```bash
docker run \
  --runtime=nvidia \
  -v "./models:/models" \
  -v "./data:/input" \
  -v "./predictions:/output" \
  supervisely/rt-detrv2-app:latest \
  predict "image.jpg" \
  --model "/f.pth" \
```

Here is the description of the parameters:
* -v "./models:/models" - mount the "models" directory as model directory inside the container
* -v "./data:/input" - mount the "data" directory as input directory inside the container.
* -v "./predictions:/output" - mount the "predictions" directory as output directory inside the container
* supervisely/rt-detrv2-app:latest - the name and the version of the docker image. You can specify older versions if needed.
* predict "image.jpg" - the command to run the model. You can specify the path to the image or directory you want to process, relative to the "data" directory.
* --model "/f.pth" - the path to the model checkpoint, relative to the "models" directory.

Optionally, you can add the following parameters to the command:
```bash
  --device cuda \
  --settings confidence_threshold=0.5
```

### Predict using API

If you deployed the model as described in [Using Model Outside of Supervisely Platform](#using-model-outside-of-supervisely-platform) section, you can predict via the API:

Curl Command
```bash
curl -X POST \
-d '{"image": "image.jpg"}' \
http://localhost:8080/predict
```

Python Code
```python
import requests
image = np.array(Image.open(IMAGE_PATH).convert("RGB")).tolist()
response = requests.post(
  "http://localhost:8080/predict",
  json={"image": image}
)
```
