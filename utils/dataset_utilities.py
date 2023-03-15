import bz2
import json
import os
import pickle
import random
import numpy as np
import ray

SAVING_FREQUENCY = 10000


class DataSetFormat:
    PICKLE = "PICKLE"
    JSON = "JSON"
    BZ2 = "BZ2"


@ray.remote
class DatasetAgent:
    def __init__(
        self,
        dataset_path,
        path_to_save_dataset,
        dataset_format,
        use_dataset=False,
        shuffle=False,
    ):
        self.dataset_path = dataset_path
        self.path_to_save_dataset = path_to_save_dataset
        self.dataset_format = dataset_format
        self.use_dataset = use_dataset
        self.shuffle = shuffle
        self.dataset = {}
        self.function_names = []
        self.nbr_updates = 0
        self.dataset_name = dataset_path.split("/")[-1].split(".")[0]

        if use_dataset:
            print(f"reading dataset from json at:{dataset_path}")
            match dataset_format:
                case DataSetFormat.PICKLE:
                    with open(dataset_path, "rb") as f:
                        self.dataset = pickle.load(f)
                        self.function_names = list(self.dataset.keys())
                case DataSetFormat.JSON:
                    with open(dataset_path, "rb") as f:
                        self.dataset = json.load(f)
                        self.function_names = list(self.dataset.keys())
                case DataSetFormat.BZ2:
                    with bz2.BZ2File(dataset_path, "rb") as f:
                        self.dataset = pickle.load(f)
                        self.function_names = list(self.dataset.keys())
                case _:
                    raise ValueError("Format specified not supported")
            print(f"[Done] reading dataset from json at:{dataset_path}")

        else:
            print(f"reading data from ls at: {dataset_path}")
            self.function_names = os.listdir(dataset_path)

        if self.shuffle:
            random.shuffle(self.function_names)

    def get_next_function(self):
        function_name = np.random.choice(self.function_names)
        if self.use_dataset:
            return function_name, self.dataset[function_name]
        else:
            return function_name, {
                "program_annotation": None,
                "schedules_legality_dict": {},
                "schedules_solver_results_dict": {},
            }

    def update_dataset(self, function_name, function_dict):
        self.dataset[function_name] = function_dict
        self.nbr_updates += 1
        print(f"# updates: {self.nbr_updates}")
        if self.nbr_updates % SAVING_FREQUENCY == 0:
            if self.nbr_updates % (2 * SAVING_FREQUENCY):
                self.save_dataset_to_disk(version=2)
            else:
                self.save_dataset_to_disk(version=1)

    def save_dataset_to_disk(self, version=1):
        print("[Start] Save the legality_annotations_dict to disk")

        updated_dataset_name = (
            f"{self.path_to_save_dataset}/{self.dataset_name}_updated_{version}"
        )
        match self.dataset_format:
            case DataSetFormat.PICKLE:
                with open(f"{updated_dataset_name}.pkl", "wb") as f:
                    pickle.dump(self.dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            case DataSetFormat.JSON:
                with open(f"{updated_dataset_name}.json", "w") as f:
                    json.dump(self.dataset, f)
            case DataSetFormat.BZ2:
                with bz2.BZ2File(f"{updated_dataset_name}.bz2.pkl", "wb") as f:
                    pickle.dump(self.dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            case _:
                raise ValueError("Format specified not supported")
        print("[Done] Save the legality_annotations_dict to disk")
        return True
