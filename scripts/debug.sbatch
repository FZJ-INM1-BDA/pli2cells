#!/usr/bin/env bash

#SBATCH --account=jinm11
#SBATCH --output=logs/slurm/out_%j.job
#SBATCH --error=logs/slurm/err_%j.job
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00


source $HOME/.bashrc
echo 'Starting script'

echo "Enabling environment..."
conda activate pli2cells

export CUDA_VISIBLE_DEVICES=0,1,2,3

# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i
# export MASTER_PORT=23519

# Debug flags
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1

# Run ime-ctl -i --block -K ${FILES} once before using cscratch
export HDF5_USE_FILE_LOCKING='FALSE'

srun -c 32 python scripts/train.py \
  debug=small_scale \
