import argparse
import os

import hydra
import ray
from hydra.core.config_store import ConfigStore
from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env

from config_utils.environment_variables import configure_env_variables
from config_utils.rl_autoscheduler_config import RLAutoSchedulerConfig
from rl_interface.environment import TiramisuScheduleEnvironment
from rl_interface.model import TiramisuModelMult
from utils.global_ray_variables import Actor, GlobalVarActor


@hydra.main(config_path="conf", config_name="config")
def main(config):
    configure_env_variables(config)
    local_dir = os.path.join(config.base_path,"ray_results")
    with ray.init(num_cpus=config.ray_num_cpus):
        progs_list_registery = GlobalVarActor.remote(
            config.programs_file, config.dataset_path, num_workers=config.num_workers
        )
        shared_variable_actor = Actor.remote(progs_list_registery)

        register_env(
            "Tiramisu_env_v1",
            lambda a: TiramisuScheduleEnvironment(
                config.programs_file,
                config.dataset_path,
                shared_variable_actor,
                config.model_checkpoint,
                env_type=config.env_type,
            ),
        )
        ModelCatalog.register_custom_model("tiramisu_model_v1", TiramisuModelMult)

        analysis = tune.run(
            "PPO",
            local_dir=local_dir,
            name=config.name,
            stop={"training_iteration": config.training_iteration},
            max_failures=0,
            checkpoint_freq=config.checkpoint_freq,
            verbose=0,
            config={
                "env": "Tiramisu_env_v1",
                "num_workers": config.num_workers,
                "batch_mode": "complete_episodes",
                "train_batch_size": 1024,
                "sgd_minibatch_size": 256,
                "lr": 1e-4,
                "num_sgd_iter": 4,
                "framework": "torch",
                "_disable_preprocessor_api": True,
                "model": {
                    "custom_model": "tiramisu_model_v1",
                    "custom_model_config": {
                        "layer_sizes": [600, 350, 200, 180],
                        "drops": [0.225, 0.225, 0.225, 0.225],
                    },
                },
            },
        )

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="experiment_config", node=RLAutoSchedulerConfig)
    main()


    
