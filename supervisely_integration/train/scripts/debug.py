from supervisely_integration.train.main import train
from supervisely.nn.utils import ModelSource, RuntimeType

# For debug
app_state = {
    "input": {"project_id": 43192},
    "train_val_split": {"method": "random", "split": "train", "percent": 80},
    "classes": ["apple"],
    "model": {"source": "Pretrained models", "model_name": "RT-DETRv2-S"},
    "hyperparameters": "epoches: 2\nbatch_size: 16\neval_spatial_size: [640, 640]  # height, width\n\ncheckpoint_freq: 5\nsave_optimizer: false\nsave_ema: false\n\noptimizer:\n  type: AdamW\n  lr: 0.0001\n  betas: [0.9, 0.999]\n  weight_decay: 0.0001\n\nclip_max_norm: 0.1\n\nlr_scheduler:\n  type: MultiStepLR  # CosineAnnealingLR | OneCycleLR\n  milestones: [35, 45]  # epochs\n  gamma: 0.1\n\nlr_warmup_scheduler:\n  type: LinearWarmup\n  warmup_duration: 1000  # steps\n\nuse_ema: True \nema:\n  type: ModelEMA\n  decay: 0.9999\n  warmups: 2000\n\nuse_amp: True\n",
    "options": {
        "model_benchmark": {"enable": True, "speed_test": True},
        "cache_project": True,
        "export": {RuntimeType.ONNXRUNTIME: True, RuntimeType.TENSORRT: True},
    },
}
train.gui.load_from_app_state(app_state)