#!/bin/bash
#SBATCH --job-name=dinov2_train
#SBATCH --output=logs/dinov2_train_%j.out
#SBATCH --error=logs/dinov2_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=8
#SBATCH --cpus-per-task=11
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --partition=debug

# Optional: Load modules or activate environment
source .venv/bin/activate


# Launch training using all GPUs
srun python train.py \
    --out /logs/thomas/dinov2/250603_dinov2_lt_in1k_50ep_dinov2_only_temp \
    --wandb_project dinov2 \
    --wandb_mode online
