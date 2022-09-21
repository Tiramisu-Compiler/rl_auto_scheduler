from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

import yaml

USE_WANDB = False


@dataclass
class RayConfig:
    num_workers: int = 1
    training_iteration: int = 1000
    ray_num_cpus: int = 112
    checkpoint_freq: int = 5
    base_path: str = "/data/scratch/hbenyamina/github/rl_autoscheduler"
    name: str = "Training_multi_enhanced"
    log_directory: str = "ray_results"


@dataclass
class EnvironmentConfig:
    dataset_path: str = "../../Dataset_multi/"
    programs_file: str = "./multicomp.json"


@dataclass
class TiramisuConfig:
    tiramisu_path: str = "/data/scratch/hbenyamina/tiramisu_rl/"
    env_type: Literal["model", "cpu"] = "cpu"
    model_checkpoint: str = "/data/scratch/hbenyamina/model_published_nn_finale.pt"


@dataclass
class TrainingConfig:
    train_batch_size: int = 1024
    sgd_minibatch_size: int = 256
    lr: float = 1e-4
    num_sgd_iter: int = 4


@dataclass
class ModelConfig:
    layer_sizes: List[int] = field(default_factory=lambda: [600, 350, 200, 180])
    drops: List[float] = field(default_factory=lambda: [0.225, 0.225, 0.225, 0.225])


@dataclass
class RLAutoSchedulerConfig:
    ray: RayConfig
    environment: EnvironmentConfig
    tiramisu: TiramisuConfig
    training: TrainingConfig
    model: ModelConfig

    def __post_init__(self):
        if isinstance(self.ray, dict):
            self.ray = RayConfig(**self.ray)

    def __post_init__(self):
        if isinstance(self.environment, dict):
            self.environment = EnvironmentConfig(**self.environment)

    def __post_init__(self):
        if isinstance(self.tiramisu, dict):
            self.tiramisu = TiramisuConfig(**self.tiramisu)

    def __post_init__(self):
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)

    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)


def read_yaml_file(path):
    with open(path) as yaml_file:
        return yaml_file.read()


def parse_yaml_file(yaml_string: str) -> Dict[Any, Any]:
    return yaml.safe_load(yaml_string)


def dict_to_config(parsed_yaml: Dict[Any, Any]) -> RLAutoSchedulerConfig:
    ray = RayConfig(**parsed_yaml["ray"])
    environment = EnvironmentConfig(**parsed_yaml["environment"])
    tiramisu = TiramisuConfig(**parsed_yaml["tiramisu"])
    training = TrainingConfig(**parsed_yaml["training"])
    model = ModelConfig(**parsed_yaml["model"])
    return RLAutoSchedulerConfig(ray, environment, tiramisu, training, model)
