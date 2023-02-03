import bz2
import json
import logging
import os
import pickle
import random
import numpy as np
import ray.data


@ray.remote
class DatasetAgent:
    def __init__(self, dataset_path, use_dataset=False, shuffle=False):
        self.dataset_path = dataset_path
        self.use_dataset = use_dataset
        self.dataset = {}
        self.function_names = []
        self.shuffle = shuffle
        if use_dataset:
            logging.info(f"reading dataset from json at:{dataset_path}")
            with bz2.BZ2File(dataset_path, 'rb') as f:
                self.dataset = pickle.load(f)
                self.function_names = list(self.dataset.keys())
            logging.info(
                f"[Done] reading dataset from json at:{dataset_path}")

        else:
            os.getcwd()
            logging.info(f"reading data from ls at: {os.getcwd()}")
            self.function_names = os.listdir(dataset_path)

        if self.shuffle:
            random.shuffle(self.function_names)

    def get_next_function(self):
        function_name = np.random.choice(self.function_names)
        return function_name, self.dataset[function_name]

    def update_dataset(self, function_name, function_dict):
        self.dataset[function_name] = function_dict

    def save_dataset_to_disk(self, path, format):
        logging.info("[Start] Save the legality_annotations_dict to disk")

        if format == "pkl":
            with bz2.BZ2File(path, 'wb') as f:
                pickle.dump(self.dataset, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, "w") as f:
                json.dump(self.dataset, f)
        logging.info("[Done] Save the legality_annotations_dict to disk")
        return True
