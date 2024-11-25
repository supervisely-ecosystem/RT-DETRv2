from typing import List
import torch
from PIL import Image
import numpy as np
import yaml
from rtdetrv2_pytorch.src.core import YAMLConfig
import torchvision.transforms as T


class RTDETRv2:
    def load_model(self, model_files: dict, device: str):
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
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        model = cfg.model
        model.load_state_dict(state)
        model.deploy().to(device)
        cfg.postprocessor.deploy().to(device)
        self.transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
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


rtdetrv2 = RTDETRv2()
# Load model
model_files = {
    "config": "app_data/work_dir/model/model_config.yml",
    "checkpoint": "app_data/work_dir/model/checkpoint0005.pth",
}
device = "cuda" if torch.cuda.is_available() else "cpu"
rtdetrv2.load_model(model_files, device)
# Predict
images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(2)]
labels, boxes, scores = rtdetrv2.predict_batch(images)
print(labels, boxes, scores)
