import os

tiramisu_path = "/scratch/hb2578/hbenyamina/cost_model/tiramisu/"  # Put the path to your tiramisu installation here
os.environ["TIRAMISU_ROOT"] = tiramisu_path

# The two environment variables below are set to 1 to avoid a Docker container error
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
os.environ["RAY_ALLOW_SLOW_STORAGE"] = "1"