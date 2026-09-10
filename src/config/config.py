from dataclasses import dataclass

import yaml



@dataclass
class TrainingConfig:
    epochs: int
    seed: int

@dataclass
class DataConfig:
    data_path: str
    train_img_dir: str
    test_img_dir: str

@dataclass
class ModelConfig:
    learning_rate: float
    weight_decay: float
    batch_size: int
    img_height: int
    img_width: int
    num_classes: int
    model_name: str
    encoder_name: str
    encoder_weights: str
    in_channels: int
    num_workers: int
    trail: int

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result

def load_config(config_path, model_path) -> Config:
    with open(config_path, "r") as f:
        # raw = yaml.safe_load(f)
        base = load_yaml(config_path)
        model = load_yaml(model_path)

        print(f"Base config: {base}")
        print(f"Model config: {model}")

        raw = deep_merge(base, model)

        print(f"Merged config: {raw}")

    return Config(
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        data=DataConfig(**raw["data"]),
    )
