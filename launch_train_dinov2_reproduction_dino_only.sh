#!/bin/bash
#SBATCH --job-name=dinov2_train
#SBATCH --output=/logs/thomas/dinov2/250603_dinov2_reproduction_dino_only/dinov2_train_%j.out
#SBATCH --error=/logs/thomas/dinov2/250603_dinov2_reproduction_dino_only/dinov2_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=8
#SBATCH --cpus-per-task=11
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --partition=debug

source .venv/bin/activate

srun python train_dinov2_reproduction.py \
    --out /logs/thomas/dinov2/250603_dinov2_reproduction_dino_only \
    --data /datasets/imagenet/train \
    --arch vits16 \
    --epochs 50 \
    --batch_size 2048 \
    --lr 0.002 \
    --resume \
    --precision bf16-mixed \
    --dino_loss_weight 1.0 \
    --ibot_loss_weight 0.0 \
    --koleo_loss_weight 0.0 \
    --momentum_end 1.0 \
    --momentum_start 0.992 \
    --weight_decay_end 0.4 \
    --centering softmax \
    --student_freeze_last_layer_epochs 2 \
    --warmup_teacher_temp_epochs 60 \
    --warmup_epochs 20 \
    --end_teacher_temp 0.07 \
    --start_teacher_temp 0.04 \
    --output_dim 65536 \
    --local_view_size 96,96 \
    --wandb_project dinov2 \
    --wandb_mode online \
    --wandb_dir /home/thomas/wandb \
    --wandb_key_path /home/thomas/.wandb_key
