#!/usr/bin/env bash

#SBATCH --account=jinm11
#SBATCH --output=logs/slurm/out_%j.job
#SBATCH --error=logs/slurm/err_%j.job
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=14:00:00


source $HOME/.bashrc
echo 'Starting script'

echo "Enabling environment..."
source environment/activate.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_DEBUG=INFO

# Run ime-ctl -i --block -K ${FILES} once before using cscratch
export HDF5_USE_FILE_LOCKING='FALSE'

echo $1

srun -c 32 python scripts/train.py \
  experiment=$1 \

