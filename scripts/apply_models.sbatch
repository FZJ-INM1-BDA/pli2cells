#!/usr/bin/env bash

#SBATCH --account=jinm11
#SBATCH --output=logs/slurm/out_%j.job
#SBATCH --error=logs/slurm/err_%j.job
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

source $HOME/.bashrc
echo 'Starting script'

echo "Enabling environment..."
source environment/activate.sh

# Define models to predict test data
MODEL=("models/wgan_no_reg_1/version_0" "models/wgan_no_reg_2/version_0" "models/wgan_no_reg_3/version_0" "models/wgan_with_reg_1/version_1" "models/wgan_with_reg_2/version_1" "models/wgan_with_reg_3/version_1")

for m in ${MODEL[*]}; do
  srun -c 32 python scripts/apply_model.py \
  --ckpt=$m/checkpoints/last.ckpt \
  --trans=datasets/vervet1818-stained/data/spline/transmittance/*h5 \
  --dir=datasets/vervet1818-stained/data/spline/direction/*h5 \
  --ret=datasets/vervet1818-stained/data/spline/retardation/*h5 \
  --out=$m/predictions \
  --num_workers=32 \
  --batch_size=8 \
  --patch_size=444 \
  --out_size=256 \
  --ram \
  --name="CresylStyle"
done
