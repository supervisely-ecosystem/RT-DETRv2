import os
import sys
sys.path.insert(0, ".")
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rtdetrv2_pytorch.src.core import YAMLConfig
from rtdetrv2_pytorch.src.solver import DetSolver


if __name__ == "__main__":
    config_path = "rtdetrv2_pytorch/tools/ssl_config.yaml"
    checkpoint = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth"
    cfg = YAMLConfig(
        config_path,
        tuning=checkpoint,
    )
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # train
    solver = DetSolver(cfg)
    solver.fit()