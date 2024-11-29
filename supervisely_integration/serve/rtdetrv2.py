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
from rtdetrv2_pytorch.src.data.dataset.coco_dataset import mscoco_category2name
from supervisely.nn.utils import ModelSource

SERVE_PATH = "supervisely_integration/serve"
CONFIG_DIR = "rtdetrv2_pytorch/configs/rtdetrv2"


class RTDETRv2(sly.nn.inference.ObjectDetection):
    FRAMEWORK_NAME = "RT-DETRv2"
    MODELS = "supervisely_integration/models_v2.json"
    APP_OPTIONS = f"{SERVE_PATH}/app_options.yaml"
    INFERENCE_SETTINGS = f"{SERVE_PATH}/inference_settings.json"
    # TODO: may be do it auto?

    def load_model(
        self, model_source: str, model_files: dict, model_info: dict, device: str, runtime: str
    ):
        config_path = f'{CONFIG_DIR}/{model_files["config"]}'
        checkpoint_path = model_files["checkpoint"]
        if model_source == ModelSource.CUSTOM:
            self._remove_include(config_path)
        else:
            self.classes =list(mscoco_category2name.values())
        cfg = YAMLConfig(config_path, resume=checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        model = cfg.model
        model.load_state_dict(state)
        model.deploy().to(device)
        cfg.postprocessor.deploy().to(device)
        h, w = 640, 640
        self.transforms = T.Compose([
            T.Resize((h, w)),
            T.ToTensor(),
        ])
        self.cfg = cfg
        self.model = model
        self.postprocessor = cfg.postprocessor
        self.device = device

    @torch.no_grad()
    def predict_batch(self, images: List[np.ndarray], settings: dict = None):
        images_pil = [Image.fromarray(img) for img in images]
        orig_sizes = torch.tensor([im.size for im in images_pil]).to(self.device)
        im_data = torch.stack([self.transforms(im) for im in images_pil]).to(self.device)
        outputs = self.model(im_data)
        outputs = self.postprocessor(outputs, orig_sizes)
        labels, boxes, scores = outputs
        return labels, boxes, scores

    def _remove_include(self, config_path: str):
        # del "__include__" and rewrite the config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "__include__" in config:
            config.pop("__include__")
            with open(config_path, "w") as f:
                yaml.dump(config, f)
                