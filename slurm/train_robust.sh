#!/bin/bash
#SBATCH --job-name=train-robust
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train-robust-%j.log

mkdir -p logs checkpoints

module load cuda/12.3.0

echo "GPU info:"
nvidia-smi

echo "Training robust model..."
python train_robust.py
