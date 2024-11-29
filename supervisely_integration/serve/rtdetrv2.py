import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from dotenv import load_dotenv
from PIL import Image

import supervisely as sly
from rtdetrv2_pytorch.src.core import YAMLConfig

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

root_dir = Path(__file__).parent.parent
app_options_path = os.path.join(root_dir, "serve", "app_options.yaml")
models_path = os.path.join(root_dir, "serve", "models_v2.json")


class RTDETRv2(sly.nn.inference.ObjectDetection):
    FRAMEWORK_NAME = "RT-DETRv2"
    MODELS = models_path
    APP_OPTIONS = app_options_path

    def load_model(
        self, model_source: str, model_files: dict, model_info: dict, device: str, runtime: str
    ):
        config_path = model_files["config"]
        checkpoint_path = model_files["checkpoint"]

        # del "__include__" and rewrite the config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "__include__" in config:
            config.pop("__include__")
            with open(config_path, "w") as f:
                yaml.dump(config, f)

        cfg = YAMLConfig(config_path, resume=checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        model = cfg.model
        model.load_state_dict(state)
        model.deploy().to(device)
        cfg.postprocessor.deploy().to(device)
        self.transforms = T.Compose(
            [
                T.Resize((640, 640)),
                T.ToTensor(),
            ]
        )
        self.cfg = cfg
        self.model = model
        self.postprocessor = cfg.postprocessor
        self.device = device

    def predict_batch(self, images: List[np.ndarray]):
        images_pil = [Image.fromarray(img) for img in images]
        orig_sizes = torch.tensor([im.size for im in images_pil]).to(self.device)
        im_data = torch.stack([self.transforms(im) for im in images_pil]).to(self.device)
        outputs = self.model(im_data)
        outputs = self.postprocessor(outputs, orig_sizes)
        labels, boxes, scores = outputs
        return labels, boxes, scores


source_path = __file__
settings_path = os.path.join(os.path.dirname(source_path), "inference_settings.yaml")
model = RTDETRv2(
    use_gui=True,
    custom_inference_settings=settings_path,
    use_serving_gui_template=True,
)
model.serve()
# Load model

# local
# model_files = {
#     "config": "app_data/work_dir/model/model_config.yml",
#     "checkpoint": "app_data/work_dir/model/checkpoint0005.pth",
# }
# # remote
# model_info = {
#     "Model": "RT-DETRv2-S",
#     "dataset": "COCO",
#     "AP_val": 48.1,
#     "Params(M)": 20,
#     "FPS(T4)": 217,
#     "meta": {
#         "task_type": "object detection",
#         "model_name": "RT-DETRv2-S",
#         "model_files": {
#             "checkpoint": "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
#             "config": "rtdetrv2_r18vd_120e_coco.yml",
#         },
#     },
# }
# model_files = {
#     "checkpoint": "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
#     "config": "rtdetrv2_r18vd_120e_coco.yml",
# }
# runtime = "PyTorch"

# device = "cuda" if torch.cuda.is_available() else "cpu"
# # rtdetrv2.load_model(model_files, model_info, device, runtime)
# # Predict
# images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(2)]
# labels, boxes, scores = rtdetrv2.predict_batch(images)
# print(labels, boxes, scores)
