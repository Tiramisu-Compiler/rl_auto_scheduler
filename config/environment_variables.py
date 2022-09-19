import os

tiramisu_path = "/data/scratch/hbenyamina/tiramisu_rl/"  # Put the path to your tiramisu installation here
os.environ["TIRAMISU_ROOT"] = tiramisu_path

# The two environment variables below are set to 1 to avoid a Docker container error
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
os.environ["RAY_ALLOW_SLOW_STORAGE"] = "1"
os.environ['TUNE_RESULT_DIR'] = "/data/scratch/hbenyamina/rl_auto_scheduler" #export TUNE_RESULT_DIR=/data/scratch/hbenyamina/rl_auto_scheduler