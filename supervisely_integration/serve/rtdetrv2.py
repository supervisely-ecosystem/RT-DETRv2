import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torchvision.transforms import ToTensor

import supervisely as sly
from rtdetrv2_pytorch.src.core import YAMLConfig
from rtdetrv2_pytorch.src.data.dataset.coco_dataset import mscoco_category2name
from supervisely.io.fs import get_file_name_with_ext
from supervisely.nn.inference import CheckpointInfo, ModelSource, RuntimeType, Timer
from supervisely.nn.prediction_dto import PredictionBBox

SERVE_PATH = "supervisely_integration/serve"
CONFIG_DIR = "rtdetrv2_pytorch/configs/rtdetrv2"


class RTDETRv2(sly.nn.inference.ObjectDetection):
    FRAMEWORK_NAME = "RT-DETRv2"
    MODELS = "supervisely_integration/models_v2.json"
    APP_OPTIONS = f"{SERVE_PATH}/app_options.yaml"
    INFERENCE_SETTINGS = f"{SERVE_PATH}/inference_settings.yaml"
    # TODO: may be do it auto?

    def load_model(
        self, model_files: dict, model_info: dict, model_source: str, device: str, runtime: str
    ):
        checkpoint_path = model_files["checkpoint"]
        if model_source == ModelSource.CUSTOM:
            config_path = model_files["config"]
            self._remove_include(config_path)
        else:
            config_path = f'{CONFIG_DIR}/{get_file_name_with_ext(model_files["config"])}'
            self.classes = list(mscoco_category2name.values())
            self.checkpoint_info = CheckpointInfo(
                checkpoint_name=os.path.basename(checkpoint_path),
                model_name=model_info["meta"]["model_name"],
                architecture=self.FRAMEWORK_NAME,
                checkpoint_url=model_info["meta"]["model_files"]["checkpoint"],
                model_source=model_source,
            )

        if runtime == RuntimeType.PYTORCH:
            self.cfg = YAMLConfig(config_path, resume=checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
            self.model = self.cfg.model
            self.model.load_state_dict(state)
            self.model.deploy().to(device)
            self.postprocessor = self.cfg.postprocessor.deploy().to(device)
            h, w = 640, 640
            self.transforms = T.Compose(
                [
                    T.Resize((h, w)),
                    T.ToTensor(),
                ]
            )
        elif runtime == RuntimeType.ONNXRUNTIME:
            # when runtime is ONNX and weights is .pth
            import onnxruntime
            from convert_onnx import convert_onnx

            self.img_size = [640, 640]
            if self.device == "cpu":
                providers = ["CPUExecutionProvider"]
            else:
                assert torch.cuda.is_available(), "CUDA is not available"
                providers = ["CUDAExecutionProvider"]
            onnx_model_path = convert_onnx(checkpoint_path, config_path)
            self.onnx_session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
        else:
            raise ValueError(f"Unknown runtime: {runtime}")

    def predict_benchmark(self, images_np: List[np.ndarray], settings: dict = None):
        if self.runtime == RuntimeType.PYTORCH:
            return self._predict_pytorch(images_np, settings)
        elif self.runtime == RuntimeType.ONNXRUNTIME:
            return self._predict_onnx(images_np, settings)

    @torch.no_grad()
    def _predict_pytorch(
        self, images_np: List[np.ndarray], settings: dict = None
    ) -> Tuple[List[List[PredictionBBox]], dict]:
        # 1. Preprocess
        with Timer() as preprocess_timer:
            imgs_pil = [Image.fromarray(img) for img in images_np]
            orig_target_sizes = torch.as_tensor([img.size for img in imgs_pil]).to(self.device)
            transformed_imgs = [self.transforms(img) for img in imgs_pil]
            samples = torch.stack(transformed_imgs).to(self.device)
        # 2. Inference
        with Timer() as inference_timer:
            outputs = self.model(samples)
        # 3. Postprocess
        with Timer() as postprocess_timer:
            labels, boxes, scores = self.postprocessor(outputs, orig_target_sizes)
            predictions = []
            for i, (labels, boxes, scores) in enumerate(zip(labels, boxes, scores)):
                classes = [self.classes[i] for i in labels.cpu().numpy()]
                boxes = boxes.cpu().numpy()
                scores = scores.cpu().numpy()
                conf_tresh = settings["confidence_threshold"]
                predictions.append(format_prediction(classes, boxes, scores, conf_tresh))
        benchmark = {
            "preprocess": preprocess_timer.get_time(),
            "inference": inference_timer.get_time(),
            "postprocess": postprocess_timer.get_time(),
        }
        return predictions, benchmark

    def _predict_onnx(
        self, images_np: List[np.ndarray], settings: dict
    ) -> Tuple[List[List[PredictionBBox]], dict]:
        # 1. Preprocess
        with Timer() as preprocess_timer:
            imgs = []
            orig_sizes = []
            for img_np in images_np:
                img = Image.fromarray(img_np)
                orig_sizes.append(list(img.size))
                img = img.resize(tuple(self.img_size))
                img = ToTensor()(img)[None].numpy()
                imgs.append(img)
            img_input = np.concatenate(imgs, axis=0)
            size_input = np.array(self.img_size * len(images_np), dtype=int).reshape(-1, 2)
        # 2. Inference
        with Timer() as inference_timer:
            labels, boxes, scores = self.onnx_session.run(
                output_names=None,
                input_feed={"images": img_input, "orig_target_sizes": size_input},
            )
        # 3. Postprocess
        with Timer() as postprocess_timer:
            predictions = []
            for i, (labels, boxes, scores) in enumerate(zip(labels, boxes, scores)):
                w, h = orig_sizes[i]
                boxes_orig = boxes / np.array(self.img_size * 2) * np.array([w, h, w, h])
                classes = [self.classes[label] for label in labels]
                conf_tresh = settings["confidence_threshold"]
                predictions.append(format_prediction(classes, boxes_orig, scores, conf_tresh))
        benchmark = {
            "preprocess": preprocess_timer.get_time(),
            "inference": inference_timer.get_time(),
            "postprocess": postprocess_timer.get_time(),
        }
        return predictions, benchmark

    def _remove_include(self, config_path: str):
        # del "__include__" and rewrite the config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "__include__" in config:
            config.pop("__include__")
            with open(config_path, "w") as f:
                yaml.dump(config, f)


def format_prediction(
    classes: list, boxes: np.ndarray, scores: list, conf_tresh: float
) -> List[PredictionBBox]:
    predictions = []
    for class_name, bbox_xyxy, score in zip(classes, boxes, scores):
        if score < conf_tresh:
            continue
        bbox_xyxy = np.round(bbox_xyxy).astype(int)
        bbox_xyxy = np.clip(bbox_xyxy, 0, None)
        bbox_yxyx = [bbox_xyxy[1], bbox_xyxy[0], bbox_xyxy[3], bbox_xyxy[2]]
        bbox_yxyx = list(map(int, bbox_yxyx))
        predictions.append(PredictionBBox(class_name, bbox_yxyx, float(score)))
    return predictions
