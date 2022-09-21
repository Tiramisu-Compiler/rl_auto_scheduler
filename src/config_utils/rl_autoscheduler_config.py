from dataclasses import dataclass, field
from typing import Literal, Dict, List

import torch
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


@dataclass
class EnvironmentConfig:
    dataset_path: str = "../../Dataset_multi/"
    programs_file: str = "./multicomp.json"


@dataclass
class TiramisuConfig:
    tiramisu_path: str = "/data/scratch/hbenyamina/tiramisu_rl/"
    env_type: Literal(["model", "cpu"]) = "cpu"
    model_checkpoint: str = "/data/scratch/hbenyamina/model_published_nn_finale.pt"


@dataclass
class RLAutoSchedulerConfig:
    ray_config: RayConfig
    environment_config: EnvironmentConfig
    tiramisu_config: TiramisuConfig

    def __post_init__(self):
        if isinstance(self.ray_config, dict):
            self.ray_config = RayConfig(**self.ray_config)

    def __post_init__(self):
        if isinstance(self.environment_config, dict):
            self.environment_config = EnvironmentConfig(
                **self.environment_config)

    def __post_init__(self):
        if isinstance(self.tiramisu_config, dict):
            self.tiramisu_config = TiramisuConfig(**self.tiramisu_config)
