from multiprocessing import cpu_count


def get_num_workers(batch_size: int):
    num_workers = min(batch_size, 8, cpu_count())
    return num_workers


# Debug
def load_from_config(train, hyperparameters_path: str):
    with open(hyperparameters_path, "r") as f:
        hyper_params = f.read()

    app_config = {
        "input": {"project_id": 43192},
        "train_val_splits": {
            "method": "random",  # "random", "tags", "datasets"
            # random
            "split": "train",  # "train", "val"
            "percent": 90,
            # tags
            # "train_tag": "cat",
            # "val_tag": "dog",
            # "untagged_action": "ignore",  # "train", "val", "ignore"
            # # datasets
            # "train_datasets": [101769, 101770],
            # "val_datasets": [101775, 101776],
        },
        "classes": ["apple"],
        "model": {
            # Pretrain
            "source": "Pretrained models",
            "model_name": "rtdetr_r50vd_coco_objects365",
            # Custom
            # "source": "Custom models",
            # "task_id": "debug-session",
            # "checkpoint": "checkpoint0011.pth",
        },
        "hyperparameters": hyper_params,
        "options": {
            "model_benchmark": {
                "enable": True,
                "speed_test": True,
            },
            "cache_project": True,
        },
    }

    train.gui.load_from_config(app_config)
