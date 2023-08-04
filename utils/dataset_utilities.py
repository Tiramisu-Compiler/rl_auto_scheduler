import bz2
import json
import os
import pickle
import random
from typing import Tuple
import numpy as np
import ray

# Frequency at which the dataset is saved to disk
SAVING_FREQUENCY = 10000


# Enum for the dataset format
class DataSetFormat:
    PICKLE = "PICKLE"
    JSON = "JSON"
    BZ2 = "BZ2"


@ray.remote
class DatasetAgent:
    """
    DatasetAgent is a class that is used to read the dataset and update it.
    It is used to read the dataset from disk and update it with the new functions.
    It is also used to save the dataset to disk.

    There are currently two modes of operation:
    1. use_dataset = True: In this mode, the dataset is read a pickle file on disk and the functions are returned from the dataset.
    2. use_dataset = False: In this mode, the dataset is not used and the functions are returned as placeholders. The list of functions is read from the disk using `ls`.

    :param dataset_path: path to the dataset
    :param path_to_save_dataset: path to save the dataset
    :param dataset_format: format of the dataset (PICKLE, JSON, BZ2)
    :param use_dataset: whether to use the dataset or not
    :param shuffle: whether to shuffle the dataset or not
    :param seed: seed for the random number generator
    """

    def __init__(
        self,
        dataset_path: str,
        path_to_save_dataset: str,
        dataset_format: DataSetFormat.BZ2 | DataSetFormat.JSON | DataSetFormat.PICKLE,
        use_dataset=False,
        shuffle=False,
        seed=None,
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
        self.current_function = 0
        self.dataset_size = 0
        self.seed = seed

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

        # Shuffle the dataset (can be used with random sampling turned off to get a random order)
        if self.shuffle:
            # Set the seed if specified (for reproducibility)
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self.function_names)

        self.dataset_size = len(self.function_names)

    def get_next_function(self, random=False) -> Tuple[str, dict]:
        # Choose a random function
        if random:
            function_name = np.random.choice(self.function_names)
        # Choose the next function sequentially
        else:
            function_name = self.function_names[
                self.current_function % self.dataset_size
            ]
            self.current_function += 1

        print(
            f"Selected function with index: {self.current_function}, name: {function_name}"
        )

        # If we are using the dataset, return the function from the dataset
        if self.use_dataset:
            return function_name, self.dataset[function_name]

        # If we are not using the dataset, return placeholders
        else:
            return function_name, {
                "program_annotation": None,
                "schedules_legality_dict": {},
                "schedules_solver_results_dict": {},
            }

    # Update the dataset with the new function
    def update_dataset(self, function_name: str, function_dict: dict) -> bool:
        """
        Update the dataset with the new function
        :param function_name: name of the function
        :param function_dict: dictionary containing the function information
        :return: True if the dataset was saved successfully
        """
        self.dataset[function_name] = function_dict
        self.nbr_updates += 1
        print(f"# updates: {self.nbr_updates}")
        if self.nbr_updates % SAVING_FREQUENCY == 0:
            if self.nbr_updates % (2 * SAVING_FREQUENCY):
                return self.save_dataset_to_disk(version=2)
            else:
                return self.save_dataset_to_disk(version=1)
        return False

    # Save the dataset to disk
    def save_dataset_to_disk(self, version=1) -> bool:
        """
        Save the dataset to disk
        :param version: version of the dataset to save (1 or 2)
        :return: True if the dataset was saved successfully
        """
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
