import argparse
import os
import sys
sys.path.append("/data/scratch/hbenyamina/github/rl_autoscheduler/rl_interface")

# import hydra
import ray
# from hydra.core.config_store import ConfigStore
from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env

from utils.environment_variables import configure_env_variables
from utils.rl_autoscheduler_config import RLAutoSchedulerConfig, dict_to_config, parse_yaml_file, read_yaml_file
from rl_interface.environment import TiramisuScheduleEnvironment
from rl_interface.model import TiramisuModelMult
from utils.global_ray_variables import Actor, GlobalVarActor


# @hydra.main(config_path="config", config_name="config")
def main(config):
    print("Here")
    configure_env_variables(config)
    print("We are in", os.getcwd())
    print("There are the following files:",os.listdir())
    local_dir = os.path.join(config.ray.base_path,"ray_results")
    with ray.init(num_cpus=config.ray.ray_num_cpus):
        progs_list_registery = GlobalVarActor.remote(
            config.environment.programs_file, config.environment.dataset_path, num_workers=config.ray.num_workers
        )
        shared_variable_actor = Actor.remote(progs_list_registery)

        register_env(
            "Tiramisu_env_v1",
            lambda a: TiramisuScheduleEnvironment(
                config.environment.programs_file,
                config.environment.dataset_path,
                shared_variable_actor,
                config.tiramisu.model_checkpoint,
                env_type=config.tiramisu.env_type,
            ),
        )
        ModelCatalog.register_custom_model("tiramisu_model_v1", TiramisuModelMult)

        analysis = tune.run(
            "PPO",
            local_dir=local_dir,
            name=config.ray.name,
            stop={"training_iteration": config.ray.training_iteration},
            max_failures=0,
            checkpoint_freq=config.ray.checkpoint_freq,
            verbose=0,
            config={
                "env": "Tiramisu_env_v1",
                "num_workers": config.ray.num_workers,
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
    parsed_yaml_dict = parse_yaml_file(read_yaml_file("config.yaml"))
    config = dict_to_config(parsed_yaml_dict)
    main(config)


    
