export WORKER_NUM=$((SLURM_JOB_NUM_NODES - 1))
export WORKER_PER_NODE=10
export CONDA_DIR=/data/scratch/hbenyamina/miniconda3
export CONDA_ENV=/data/scratch/hbenyamina/tiramisu-build-env
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_ALLOW_SLOW_STORAGE=1
export PORT=6379