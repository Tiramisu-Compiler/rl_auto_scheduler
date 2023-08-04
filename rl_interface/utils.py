import json
import os

from tiramisu_programs.schedule_utils import NumpyEncoder


class EnvironmentUtils:
    @classmethod
    def write_json_dataset(cls, filename, data):
        if not os.path.isdir("./Dataset/"):
            os.mkdir("./Dataset/")
        dataset_file = os.path.join("./Dataset/", filename)
        with open(dataset_file, "w+") as f:
            f.write(json.dumps(data, cls=NumpyEncoder))
        return True
