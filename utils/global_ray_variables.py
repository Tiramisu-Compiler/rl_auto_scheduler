import ray
import json
import os


@ray.remote
class GlobalVarActor:
    def __init__(self, programs_file, dataset_path, num_workers=7):
        self.index = -1
        self.num_workers = num_workers
        self.progs_list = self.get_dataset(dataset_path)
        self.programs_file = programs_file
        self.progs_dict = dict()
        # if os.path.isfile(programs_file):
        #     with open(programs_file) as f:
        #         self.progs_dict = json.load(f)
        # else:
        #     self.progs_dict = dict()
        #     with open(programs_file,"w+") as f:
        #         f.write(json.dumps(self.progs_dict))

    def get_dataset(self, path):
        os.getcwd()
        print("***************************", os.getcwd())
        prog_list = os.listdir(path)
        return prog_list

    def set_progs_list(self, v):
        self.progs_list = v

    def get_progs_list(self, id):
        # with open(f'file_{id}.txt',"w+") as file:
        #     file.write(f"Using files which index % self.num_workers = {id % self.num_workers}, such as id = {id} and num_workers = {self.num_workers}")
        return [
            item
            for (index, item) in enumerate(self.progs_list)
            if (index % self.num_workers) == (id % self.num_workers)
        ]

    def update_progs_dict(self, v):
        self.progs_dict.update(v)
        return True

    def write_progs_dict(self):
        print("Saving progs_dict to disk")
        with open(self.programs_file, "w") as f:
            json.dump(self.progs_dict, f)
        return True

    def get_progs_dict(self):
        return self.progs_dict

    def increment(self):
        self.index += 1
        return self.index


@ray.remote
class Actor:
    def __init__(self, data_registry):
        self.data_registry = data_registry

    def get_progs_list(self, id):
        return ray.get(self.data_registry.get_progs_list.remote(id))

    def get_progs_dict(self):
        return ray.get(self.data_registry.get_progs_dict.remote())

    def write_progs_dict(self):
        return ray.get(self.data_registry.write_progs_dict.remote())

    def update_progs_dict(self, v):
        return ray.get(self.data_registry.update_progs_dict.remote(v))

    def increment(self):
        return ray.get(self.data_registry.increment.remote())
