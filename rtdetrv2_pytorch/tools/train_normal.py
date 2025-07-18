import os
import sys
sys.path.insert(0, ".")
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rtdetrv2_pytorch.src.core import YAMLConfig
from rtdetrv2_pytorch.src.solver import DetSolver
import yaml
from datetime import datetime


def save_config(cfg, output_dir):
    """Save the configuration to a YAML file."""
    output_config_path = os.path.join(output_dir, "config.yml")
    with open(output_config_path, "w") as f:
        yaml.dump(cfg.yaml_cfg, f)
    _remove_include(output_config_path)
    print(f"Configuration saved to {output_config_path}")

def _remove_include(config_path: str):
    # del "__include__" and rewrite the config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "__include__" in config:
        config.pop("__include__")
        with open(config_path, "w") as f:
            yaml.dump(config, f)


if __name__ == "__main__":
    config_path = "rtdetrv2_pytorch/tools/normal_config.yaml"
    checkpoint = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth"
    cfg = YAMLConfig(
        config_path,
        tuning=checkpoint,
    )

    cfg.output_dir  = cfg.output_dir + f"/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(cfg.output_dir, exist_ok=True)

    # dump resolved config
    save_config(cfg, cfg.output_dir)

    # train
    solver = DetSolver(cfg)
    solver.fit()
