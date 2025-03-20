## Your experiment info

Оглавление
Short usecase explanation
tqdm

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

team_id - в контекст API объекта / ENV засунуть чтобы не передавать всегда?
team_id - можно ли понять активную тиму?
еще можно подумать, что если всего одна команда то мы сами найдем файл
и даже если несколько команд то мы тоже можем найти команду и файл сами

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

If you deployed model using Supervisely SDK as shown in the [Deploy model using API](#deploy-model-using-api) section, you already have the model object. Otherwise, you can create it using the following code:

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

#### Predict images

```python
class Prediction:
    source: Union[str, int]
    annotation: sly.Annotation
    frame_index: Optional[int] # only for video frames
```

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

inference_settings-rename => 
inference_settings = {"confidence_threshold": 0.5} # Optional
annotation: Prediction = model.predict(image=image, settings=inference_settings)
model.predict(image=image, params=params)????
model.predict(image=image, opts=opts)????
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

inference_settings = {"confidence_threshold": 0.5} # Optional
predictions: List[Prediction] = model.predict(image=images, settings=inference_settings)
```
#### Predict video

detach (video / project / dataset)
+stop
whle true sleep

```python
# str path to video
video = "/a/b/c.mp4"
# str url to video file in team files
video = "https://a/b/c.mp4"
# int video id
video = 111

inference_settings = {"confidence_threshold": 0.5} # Optional
predictions: List[Prediction] = model.predict(video=video, settings=inference_settings)
```

After you got predictions for video frames, you can apply tracking algorithm. You can use the following code:
```python
from supervisely.nn.tracking import track
import boxmot

device = "cuda:0
tracker = boxmot.BotSort(reid_weights=Path('osnet_x0_25_msmt17.pt'), device=device, half=False)
video_ann = track(tracker, predictions, video)
```

#### Predict directory

```python
# str path to directory with images or videos
directory = "/a/b/c"
# str url to directory with images or videos in team files
directory = "https://a/b/c"

inference_settings = {"confidence_threshold": 0.5} # Optional
output = "/a/b/c/predictions" # Optional. If not specified, annotations will be saved in the same directory
predictions: List[Prediction] = model.predict(dir=dir, output=output, settings=inference_settings)
```

#### Predict dataset

```python
# int dataset id
dataset = 123

inference_settings = {"confidence_threshold": 0.5} # Optional
inplace = True # merge replace smart merge # Optional. Default is False. If True, annotations will be uploaded to the dataset
predictions: List[Prediction] = model.predict(dataset=dataset, inplace=inplace, settings=inference_settings)
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


batched example

mega optimal example


detach corutine?? example

iterator???
inplace
detach?

#### Predict project

```python
# str path to a project directory
project = "/a/b/c"
# int project id
project = 123

inference_settings = {"confidence_threshold": 0.5} # Optional
inplace = True # Optional. Default is False. If True, annotations will be either uploaded to the project or saved in the same directory
predictions: List[Prediction] = model.predict(project=project, inplace=inplace, settings=inference_settings)
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

пример что 

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
