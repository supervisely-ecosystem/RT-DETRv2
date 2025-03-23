# Demo: Inference & Deployment

**Table of Contents (only for this readme):**

- [Predict in One Click](#predict-in-one-click)
- [Deploy Model in Supervisely Platform](#serve-model-in-supervisely-platform)
- [Inference via API](#inference-via-api)
- [Using Model Outside of Supervisely Platform](#using-model-outside-of-supervisely-platform)
  - [Get predictions in your code](#get-predictions-in-your-code)
  - [Deploy model as a server on your machine](#deploy-model-as-a-server-on-your-machine)
  - [Deploy in a Docker Container](#deploy-in-a-docker-container)
- [Using Your Model as a Standalone PyTorch Model](#using-your-model-as-a-standalone-pytorch-model)
- ❓ onnx
- ❓ tensorrt
- ❓ tqdm
- ❓ - есть ли у нас контекст по пользователю - команда из env / или просто активная команда и тд, чтобы упростить количество аргументов типа team_id / agent_id и тд
- ❓ - удобно ли нам будет это использовать в других приложениях?
- 🔴 как сделан api yolo?

> You can use your model in very different ways depending on your needs. For more information, please, refer to our full [Inference & Deployment](https://docs.supervisely.com/neural-networks/overview-1) documentation.

## Predict in One Click
🔴 на странице эксперимента

Soon: run model inference on your data from experiment page.  
You can apply your model in a single click. Select the input project and datasets, configure inference settings if needed, and run the model.

*(We add this selector right into the page)*

Спрятал пока картинку чтобы не мешала восприятию
[Selecting a project GIF (tmp)](https://developer.supervisely.com/~gitbook/image?url=https%3A%2F%2Fuser-images.githubusercontent.com%2F79905215%2F222367677-cdee343d-a841-4868-9106-10d3f44d9e76.gif&width=768&dpr=4&quality=100&sign=424c1477&sv=2)

> You can also get predictions with **Applying Apps**, such as [Apply NN to Images](https://ecosystem.supervisely.com/apps/nn-image-labeling/project-dataset) or [Apply NN to Video](https://ecosystem.supervisely.com/apps/apply-nn-to-videos-project). Read more in documentation [Apply Model in Platform](https://docs.supervisely.com/neural-networks/overview-1/supervisely-serving-apps#apply-model-in-platform).

## Deploy Model in Supervisely Platform

To deploy a model on the platform, we use [Supervisely Serving Apps](https://docs.supervisely.com/neural-networks/overview-1/supervisely-serving-apps). For your model use the {Serve RT-DETRv2} Serving App:

*(Clickable widget)*
[Serve RT-DETRv2 App](img/serving-app.png)

❓ - это осталось или что с этими коментами?

🔴 - agent_id переименовать? server_id?
🔴 - запуск агента и установка зависимостией - сделано коля
🔴 - checkpoint="supervisely.com/files/123/a/b/c.pth" + web? вместо team_id + path
🔴 - в checkpoint нужно защивать инфу experiemnt dict в котором все - версия апы, метаданные

### Deploy using Supervisely SDK
Alternatively, you can use [Supervisely SDK](https://github.com/supervisely/supervisely) to deploy the model on the platform:

```python
import os
import supervisely as sly
from dotenv import load_dotenv

# Ensure you've set API_TOKEN and SERVER_ADDRESS environment variables.
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()
```
🔴
Проверить что sly.Api в конструкторе не отправляет запросы - сергей делал там проверку на hhtp / https - возможно ее надо переносить on-demand
ModelAPI - подумать как в Session сделать api опциональным чтобы использовать при локальном коннекте вне платформу. в ModelAPI точно это не должно быть и делаться on-demand

❓ - может сделаем api необязательным - или вообще его уберем?
❓ - почему тут URL, можно ли узказать deploy_id
```Python
class ModelApi:
  def init(self, api, url):
    self._api = api
    self._session_url
    self.session = None
  @property
  def session(self):
    if self._session is None:
      self._session = Session(self._api, self._session_url)
    return self._session
  def predcit():
  def stop():
  def shutdown():
  def get_info():
  def healthcheck():
```
🔴

* With path to the model checkpoint:
    ```python
    model = api.nn.deploy.custom(
        checkpoint="/experiments/path/to/checkpoint.pth"
    )
    ```

* With train task id: (❓ best / last / конкретный? может лучше сделать api.nn.get_checkpoints(train_id=123)->d: dict-> model = api.nn.deploy.custom(d["xxx"]))
    ```python
    model = api.nn.deploy.custom(
        train_id=123,
    )
    ```

* Or you can deploy pretrained model (❓ - как получить список возмоных? и залистить их - типа хелпы):
    ```python
    model = api.nn.deploy.pretrained(framework="RT-DETRv2", model_name="RT-DETRv2-S")
    ```

> For more information, see [Deploy & Predict with Supervisely SDK](https://docs.supervisely.com/neural-networks/overview-1/deploy_and_predict_with_supervisely_sdk).


## Inference via API

Every Served Model on the platform is available via API. You can get predictions using our convenient inference `ModelApi` class in Supervisely SDK.

### 1. Connect to model

If you deployed a model as shown in [Deploy using Supervisely SDK](#deploy-using-supervisely-sdk) then you already have the model object. Otherwise, you can connect to the model using the following code snippet:

Initialize the API client:
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
team_id = 123 # Optional # 🔴 автоподстановка при рендере, опциональная, ее можно получить либо из енва либо зная юзера.
framework = "RT-DETRv2" # Opitonal
models = api.nn.get_running_models(framework=framework, team_id=team_id)
model = models[0]
```

Connect to the model by id:
🔴 - откуда берется model_id? может лучше назвать deploy_id - по сути это task_id?
```python
model_id = 123
model = api.nn.connect(model_id)
```

Or connect to the model by url if you deployed it as an API server as shown in the [Using Model Outside of Supervisely Platform](#using-model-outside-of-supervisely-platform) section:
```python
url = "localhost:8080"
model = api.nn.connect(url)
```

### 2. Predict

```python
class ModelApi:
    def predict():
    def stop():
    def shutdown():
    def get_info():
    def healthcheck():
```

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
model_id = 123 🔴
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

🔴 - не до конца уверен что этот класс нужен
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
🔴 - название тоже под вопросом
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
🔴 - а нельзя проще через tqdm и генератор?
🔴 - почему этот пример идет в самом верху перед остальными более простыми и базовыми методами?
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
🔴 - как понять какие у модели есть поддерживаемые параметры, может это не в первом примере делать?
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

🔴 по идее если мы предиктим на картинках, то нам PredictionDTO не нужен, он нужен возможно только в видео и dataset / project
# Processing images one by one
params = {"confidence_threshold": 0.5} # Optional
for image in images:
    prediction: Prediction = model.predict(image=image, params=params)
    json.dump(prediction.annotation.to_json(), open(f"{image}.json", "w"))

# Processing images in batch
predictions: List[Prediction] = model.predict(image=images, params=params)
json.dump([prediction.annotation.to_json() for prediction in predictions], open("predictions.json", "w"))

🔴 # if predict yields annotations - то есть это будет дефолтная реализация?
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
🔴 - tqdm возможне
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

### 3. Stop the model

```python
model.shutdown()
```
or
```python
api.nn.deploy.stop(model_id)
```

> For more information, see [Inference API Tutorial](https://docs.supervisely.com/neural-networks/inference-api).

## Using Model Outside of Supervisely Platform

To use the model outside of the Supervisely platform, you will first need to download the model files.
You can download the entire folder containing your model files and checkpoints, or download only the necessary files:

🔴🔴🔴 кнопка "download model.zip"

![Download checkpoint from Team Files](img/team_files_download_2.png)

For example, create a folder `models` on your machine, place there your downloaded models. You had to specify a path to your model later.


### Deploy in a Docker Container

Deploying in a Docker Container is a convenient way to run a model without needing to install dependencies on your machine. You can use our pre-built docker image with the model implementation.

#### Using Docker run command

Predcit with custom model:
```bash
docker run \
  --env-file ~/supervisely.env \  # 🔴 Опционально. Если нет, то предикт проекта не будет работать. Если использовать баш скрипт или команду supervisely deploy/predict, то можно искать автоматически в дефолтных местах
  --runtime=nvidia \ 🔴 - а это автоматом можно по чекпоинту понимать?
  -v "<user_input_dir>:/input" \
  -v "<user_models_dir>:/models" \
  -v "<user_output_dir>:/output" \
  supervisely/rt-detrv2:1.0.11 \
  predict "<user_image_name>" \
  --model "<model path inside of models dir>" \ # .pth file
  --device cuda \
  --settings confidence_threshold=0.5
```

Deploy custom model:
```bash
docker run \
  --runtime=nvidia \
  -v "<user_models_dir>:/models" \
  -p <host_port>:8000 \
  supervisely/rt-detrv2:1.0.11 \
  deploy \
  --model "<model path inside of models dir>" \ # .pth file
```
🔴
Можно при релизе билдить новый образ rt-detrv2:<app-version> с кодом репы, это будет проще и быстрее. В докерфайле прописать 
```Dockerfile
FROM supervisely/rt-detrv2-base:v1
COPY . /app
ENTRYPOINT ["sh", "-c", "PYTHONPATH=\"${PWD}:${PYTHONPATH}\" exec python3 supervisely_integration/serve/main.py \"$@\"", "sh"]
```
🔴

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

Then we can predict using the API:
Either using Supervisely SDK or curl:

Тут надо описать какие параметры принимает API и какие ответы возвращает

```python
import supervisely as sly
model = sly.api.nn.connect(url="localhost:<host_port>")
annotation = model.predict(image="image.jpg")
```
```bash
curl -X POST http://localhost:<host_port>/predict \
  -d '{"image": "image.jpg"}' > prediction.json
```

#### Using Supervisely CLI

```bash
pip install -U supervisely
```

Then you can either predict an image or deploy the model in a container and then predict

Predict with cutom model:
```bash
supervisely predict \
  --model "./models/392_RT-DETRv2/checkpoints/best.pth" \
  --image "./image.jpg" \
  --output "./predictions" \
  --device cuda \
  --settings confidence_threshold=0.5
```

Predict with pre-defined model:
```bash
supervisely predict \
  --model "RT-DETRv2-S" \
  --image "./image.jpg" \
  --output "./predictions" \
  --device cuda \
  --settings confidence_threshold=0.5
```

Deploy custom model
```bash
supervisely deploy \
  --model "./models/392_RT-DETRv2/checkpoints/best.pth" \
  --device cuda \
  --settings confidence_threshold=0.5
  -- \
  --port <host_port>:8000  # example arguments to docker run
  -d # run in detached mode
```

Deploy pre-defined model
```bash
supervisely deploy \
  --model "RT-DETRv2-S" \
  --device cuda \
  --settings confidence_threshold=0.5
  -- \
  --port <host_port>:8000  # example arguments to docker run
  -d # run in detached mode
```

Then we can predict:
```bash
supervisely predict \
  <host_port> \
  "./image.jpg" \
  --output "./predictions"
```

> See more information in the [Deploy in Docker Container](https://docs.supervisely.com/neural-networks/overview-1/deploy_and_predict_with_supervisely_sdk#deploy-in-docker-container) documentation.


### Get predictions in your code

This example shows how to load your checkpoint and get predictions in any of your code. You will need to clone our RT-DETRv2 repository and install dependencies. Then, you can load the model in your code and get predictions.

#### 1. Clone repository

Clone our [RT-DETRv2](https://github.com/supervisely-ecosystem/RT-DETRv2) fork with the model implementation.

```bash
git clone https://github.com/supervisely-ecosystem/RT-DETRv2
cd RT-DETRv2
```

#### 2. Set up environment

Install [requirements.txt](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/rtdetrv2_pytorch/requirements.txt) manually, or use our pre-built docker image ([DockerHub](https://hub.docker.com/r/supervisely/rt-detrv2/tags) | [Dockerfile](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/docker/Dockerfile)). Additionally, you need to install Supervisely SDK.

```bash
pip install -r rtdetrv2_pytorch/requirements.txt
pip install supervisely
```

#### 3. Download checkpoint

Download your checkpoint and model files from Team Files.

![Download checkpoint from Team Files](img/team_files_download_2.png)

#### 4. Predict

Create `main.py` file in the root of the repository and paste the following code. This code will load the model, predict the image, and save the results to `./prediction.jpg`.

```python
import numpy as np
from PIL import Image
import os
import supervisely as sly
from supervisely.nn import ModelSource, RuntimeType

# Be sure you are in the root of the RT-DETRv2 repository
from supervisely_integration.serve.rtdetrv2 import RTDETRv2

# Put your path to image here
IMAGE_PATH = "sample_image.jpg"
# Model config and weights and list of classes (downloaded from Team Files)
CHECKPOINT_PATH = "model/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth" # or None если есть дефолтные в образе
CONFIG_PATH = "model/rtdetrv2_r18vd_120e_coco.yml" # or None если есть дефолтные в образе
CLASSES = "model/classes.json" # < Модель может сама найти? Можно переопределять? Нужно ли если есть model_meta.json?
MODEL_META_PATH = "model/model_meta.json" # < or None если есть дефолтные в образе

# Load model
class RTDETRv2Serving(Inference):
class RTDETRv2(Prediction/ ... ): - потом можно иерархию наследования подшаманить чтобы упростить количество аршументов и отделить наши от базового predict

model = RTDETRv2() - 🔴 - это возможно за счет того что унас для моделей будут класссы inference
model.load_checkpoint(path="/a/b.pth", device="cuda")

🔴🔴🔴🔴 - если это только для моделей, обученный в SLY, то тогда можно все вживать в чекпоинт для упрощения - будем искать поля в файле и если что- фолбекаться по умолчанию к поиску файла в той же папке в фоне и потом если что писать ошибку что не можем найти confix_x.abc

model = RTDETRv2(checkpoint="", device="cuda") # device можно определять самим если None

# Load image
img = np.array(Image.open(IMAGE_PATH).convert("RGB"))
img = "/a/b/c.jpg"
img = "https://a/b/c.jpg"
img = 777
img = [777, ]

# Predict
🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴 -- model.inference -> model.predict - чтобы было одинаково и единообразно
Это придется переделывать все существующие модели (и новые и старые).
prediction = model.inference(img, settings={"confidence_threshold": 0.4})

prediction: Class {image, ann, image_id} # Почему не просто аннотация

prediction.draw(save="prediction.jpg")
```

If you need to run the code in your project and not in the root of the repository, you can add the path to the repository into `PYTHONPATH`, or by the following lines at the beginning of the script:

```python
import sys
sys.path.append("/path/to/RT-DETRv2")
```

### Deploy model as a server on your machine

In this variant, you will deploy a model locally as an API Server with the help of Supervisely SDK. The server will be ready to process API request for inference. It allows you to predict with local images, folders, videos, or remote supervisely projects and datasets (if you provide your Supervisely API token).

#### 1. Clone repository

Clone our [RT-DETRv2](https://github.com/supervisely-ecosystem/RT-DETRv2) fork with the model implementation.

```bash
git clone https://github.com/supervisely-ecosystem/RT-DETRv2
cd RT-DETRv2
```

#### 2. Set up environment

Install [requirements](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/rtdetrv2_pytorch/requirements.txt) manually, or use our pre-built docker image ([DockerHub](https://hub.docker.com/r/supervisely/rt-detrv2/tags) | [Dockerfile](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/docker/Dockerfile)).

```bash
pip install -r rtdetrv2_pytorch/requirements.txt
pip install supervisely
```

#### 3. Download checkpoint

Download your checkpoint, model files and `experiment_info.json` from Team Files or the whole artifacts directory.

![Download checkpoint from Team Files](img/team_files_download_2.png)

Place downloaded files in the folder within app repo. For example, create `models` folder inside root directory of the repository and place all files there.

Your repo should look like this:

```plaintext
📦app-repo-root
 ┣ 📂models
 ┃ ┗ 📂392_RT-DETRv2
 ┃   ┣ 📂checkpoints
 ┃   ┃ ┗ 🔥best.pth
 ┃   ┣ 📜experiment_info.json
 ┃   ┣ 📜model_config.yml
 ┃   ┗ 📜model_meta.json
 ┗ ... other app repository files
```

#### 4. Deploy

To deploy, run our `main.py` script to start the server. You need to pass the path to your checkpoint file. Like in the previous section, add the path to the {RT-DETRv2} repository into `PYTHONPATH`.

```bash
PYTHONPATH="${PWD}:${PYTHONPATH}" \
python ./supervisely_integration/serve/main.py deploy \
--model "./models/392_RT-DETRv2/checkpoints/best.pth"
```

This command will start the server on [http://0.0.0.0:8000](http://0.0.0.0:8000) and will be ready to accept API requests for inference.


#### 5. Predict

##### Predict via API

After the model is deployed, use Supervisely [Inference Session API](https://developer.supervisely.com/app-development/neural-network-integration/inference-api-tutorial) with setting server address to [http://0.0.0.0:8000](http://0.0.0.0:8000).

```python
import os
from dotenv import load_dotenv
import supervisely as sly

load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()

# Create Inference Session
session = sly.nn.inference.Session(api, session_url="http://0.0.0.0:8000")

# local image
🔴🔴🔴 то же самое что и выше -- model.predict(image="image.jpg")
prediction = session.inference_image_path("image_01.jpg")

# batch of images
predictions = session.inference_image_paths(["image_01.jpg", "image_02.jpg"])
```

> For the full list of arguments, see the documentation [Deploy Model as a Server](https://docs.supervisely.com/neural-networks/overview-1/deploy_and_predict_with_supervisely_sdk#id-4.-deploy).


---

## Using Your Model as a Standalone PyTorch Model

Models trained in Supervisely can be used as a standalone PyTorch model (or ONNX / TensorRT) outside of the platform. This method completely decouple you from both Supervisely Platform and Supervisely SDK, and you will develop your own code for inference and deployment.

1. **Download** your checkpoint and model files from Team Files.

2. **Clone** our [RT-DETRv2](https://github.com/supervisely-ecosystem/RT-DETRv2) fork with the model implementation. Alternatively, you can use the original [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR/tree/0b6972de10bc968045aba776ec1a60efea476165) repository, but you may face some unexpected issues if the authors have updated the code.

```bash
git clone https://github.com/supervisely-ecosystem/RT-DETRv2
cd RT-DETRv2
```

3. **Set up environment:** Install [requirements](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/rtdetrv2_pytorch/requirements.txt) manually, or use our pre-built docker image ([DockerHub](https://hub.docker.com/r/supervisely/rt-detrv2/tags) | [Dockerfile](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/docker/Dockerfile)).

```bash
pip install -r rtdetrv2_pytorch/requirements.txt
```

4. **Run inference:** Refer to our example scripts of how to load RT-DETRv2 and get predictions:

- [demo_pytorch.py](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_pytorch.py)
- [demo_onnx.py](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_onnx.py)
- [demo_tensorrt.py](https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_tensorrt.py)

**demo_pytorch.py** is a simple example of how to load a PyTorch checkpoint and get predictions. You can use it as a starting point for your own code:

```python
import json
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
from rtdetrv2_pytorch.src.core import YAMLConfig


device = "cuda" if torch.cuda.is_available() else "cpu"

# put your files here
checkpoint_path = "model/best.pth"
config_path = "model/model_config.yml"
model_meta_path = "model/model_meta.json"
image_path = "img/coco_sample.jpg"


def draw(images, labels, boxes, scores, classes, thrh = 0.5):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]
        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{classes[lab[j].item()]} {round(scrs[j].item(),2)}", fill='blue', )


if __name__ == "__main__":

    # load class names
    with open(model_meta_path, "r") as f:
        model_meta = json.load(f)
    classes = [c["title"] for c in model_meta["classes"]]

    # load model
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model = cfg.model
    model.load_state_dict(state)
    model.deploy().to(device)
    postprocessor = cfg.postprocessor.deploy().to(device)
    h, w = 640, 640
    transforms = T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
    ])

    # prepare image
    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)
    im_data = transforms(im_pil)[None].to(device)

    # inference
    output = model(im_data)
    labels, boxes, scores = postprocessor(output, orig_size)

    # save result
    draw([im_pil], labels, boxes, scores, classes)
    im_pil.save("result.jpg")
```

## Evaluate on Your Data

You can run [Model Evaluation Benchmark](https://docs.supervisely.com/neural-networks/model-evaluation-benchmark) on your data to get metrics and visualize results.

🔴🔴🔴
- выбор проекта/датасета
- может это в модалке сделать? Не хочется запускать эпу, ждать, чтобы только выбрать датасет.

code snippet:

```python
import os
import supervisely as sly
from dotenv import load_dotenv

from supervisely.nn.benchmark import ModelBenchmark  🔴🔴🔴 пока есть только бенчмарки под каждый task_type


if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

api = sly.Api()

team_id = 8
gt_project_id = 73
model_id = 1234

# 1. Initialize benchmark
bench = ModelBenchmark(api, gt_project_id, output_dir=sly.app.get_data_dir())

# 2. Run evaluation
bench.run_evaluation(model_id)  🔴🔴🔴 need implement
return: {pred_project_id, inference_info}

# 3. Generate charts and dashboards
json_metrics = bench.generate_report()  🔴🔴🔴 need implement
```
