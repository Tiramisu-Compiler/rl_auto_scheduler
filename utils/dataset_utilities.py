import json
import os
import random

import ray.data


class DatasetAgent:
    def __init__(self, dataset_path, shuffle=True):
        self.shuffle = shuffle
        if os.path.isfile(dataset_path):
            self.dataset = ray.data.read_json(dataset_path)
        self.function_names = self.dataset.keys()

        if self.shuffle:
            random.shuffle(self.function_names)

    def get_next_function(self):
        for function in self.function_names:
            yield function, self.dataset[function]
