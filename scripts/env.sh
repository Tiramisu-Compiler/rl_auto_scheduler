export WORKER_NUM=$((SLURM_JOB_NUM_NODES - 1))
export WORKER_PER_NODE=10
export CONDA_DIR=/data/scratch/hbenyamina/miniconda3
export CONDA_ENV=/data/scratch/hbenyamina/tiramisu-build-env