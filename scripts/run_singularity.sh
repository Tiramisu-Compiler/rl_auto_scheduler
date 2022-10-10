#!/bin/bash
#SBATCH --reservation c2
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task=28
#SBATCH -t 7-0:00:00
#SBATCH -o outputs/singularity.out
#SBATCH -e outputs/singularity.err


overlay_ext3=/scratch/hb2578/hbenyamina/cost_model/overlay-200000M-500K.ext3
singularity \
shell --overlay $overlay_ext3 \
/share/apps/admin/singularity-images/centos-8.2.2004.sif \
/bin/bash -c "source ~/.bashrc; \
     conda activate /scratch/hb2578/tiramisu-build-env;  \
     python train_ppo.py --num-workers 23 --single-node "