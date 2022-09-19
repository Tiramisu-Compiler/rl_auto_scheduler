import ray, argparse
from rl_interface.model import TiramisuModelMult
from rl_interface.environment import SearchSpaceSparseEnhancedMult
from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env
from utils.global_ray_variables import GlobalVarActor, Actor
from config.environment_variables import *

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
    dataset_path = "../../Dataset_multi/"
    programs_file = "./multicomp.json"
    model_checkpoint = "/data/scratch/hbenyamina/model_published_nn_finale.pt"
    local_dir = "/data/scratch/hbenyamina/cost_model/rl_autoscheduler_multicomp/ray_results"
    name="Training_multi_enhanced"

    with ray.init(num_cpus=args.ray_num_cpus):
        progs_list_registery = GlobalVarActor.remote(
            programs_file, dataset_path, num_workers=args.num_workers
        )
        shared_variable_actor = Actor.remote(progs_list_registery)

        register_env(
            "Tiramisu_env_v1",
            lambda a: SearchSpaceSparseEnhancedMult(
                programs_file,
                dataset_path,
                shared_variable_actor,
                model_checkpoint,
                env_type=args.env_type,
            ),
        )
        ModelCatalog.register_custom_model("tiramisu_model_v1", TiramisuModelMult)

        analysis = tune.run(
            "PPO",
            local_dir=local_dir,
            name=name,
            stop={"training_iteration": args.training_iteration},
            max_failures=0,
            checkpoint_freq=args.checkpoint_freq,
            verbose=0,
            config={
                "env": "Tiramisu_env_v1",
                "num_workers": args.num_workers,
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
