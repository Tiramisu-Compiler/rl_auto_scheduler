import argparse
import json

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env

from config_utils.environment_variables import *
from rl_interface.environment import TiramisuScheduleEnvironment
from rl_interface.model import TiramisuModelMult
from utils.global_ray_variables import Actor, GlobalVarActor


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--training-iteration", default=1000, type=int)
    parser.add_argument("--ray-num-cpus", default=112, type=int)
    parser.add_argument("--checkpoint-freq", default=5, type=int)
    parser.add_argument("--env-type", default="model", choices=["cpu", "model"], type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    base_path = "/data/scratch/hbenyamina/github/rl_autoscheduler"
    dataset_path = "../../Dataset_multi/"
    programs_file = "./multicomp.json"
    model_checkpoint = "/data/scratch/hbenyamina/model_published_nn_finale.pt"
    local_dir = os.path.join(base_path,"ray_results")
    best_checkpoint = os.path.join(local_dir,"Reward_minus_Epsilon/PPO_Tiramisu_env_v1_00c6a_00000_0_2022-09-08_01-56-38/checkpoint_000068/checkpoint-68")
    name="Training_multi_enhanced"

    with ray.init(num_cpus=args.ray_num_cpus):
        progs_list_registery = GlobalVarActor.remote(
            programs_file, dataset_path, num_workers=args.num_workers
        )
        shared_variable_actor = Actor.remote(progs_list_registery)

        register_env(
            "Tiramisu_env_v1",
            lambda a: TiramisuScheduleEnvironment(
                programs_file,
                dataset_path,
                shared_variable_actor,
                model_checkpoint,
                env_type=args.env_type,
            ),
        )
        ModelCatalog.register_custom_model("tiramisu_model_v1", TiramisuModelMult)

        agent = ppo.PPOTrainer(
            env="Tiramisu_env_v1",
            config={
                "num_workers": args.num_workers,
                "batch_mode":"complete_episodes",
                "train_batch_size":1024,
                "sgd_minibatch_size": 256,
                "lr": 1e-4,
                "num_sgd_iter": 4,
                "explore": False,
                "framework":"torch",
                "_disable_preprocessor_api": True,
                "model": {
                    "custom_model": "tiramisu_model_v1",
                    "custom_model_config": {
                            "layer_sizes":[128, 1024, 1024, 128],
                            "drops":[0.225, 0.225, 0.225, 0.225]
                        }
                },
            },
        )

        agent.restore(best_checkpoint)

        env= TiramisuScheduleEnvironment(
                programs_file,
                dataset_path,
                shared_variable_actor,
                model_checkpoint,
                env_type=args.env_type,
            )

        
        results = []
        while True:
            result = dict()
            observation, done = env.reset(), False
            result["prog"] = env.prog.name
            while not done:
                try:
                    action = agent.compute_action(observation)
                    observation, reward, done, _ = env.step(action)
                except:
                    continue
            result["schedule_str"] = env.schedule_str
            result["speedup"] = env.speedup
            results.append(result)
            with open("results.json","w+") as file:
                file.write(json.dumps(results))
