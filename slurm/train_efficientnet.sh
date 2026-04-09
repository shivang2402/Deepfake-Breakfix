#!/bin/bash
#SBATCH --job-name=train-effnet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train-effnet-%j.log

mkdir -p logs checkpoints
module load cuda/12.3.0

echo "Training EfficientNet-B0..."
nvidia-smi
python train_efficientnet.py
