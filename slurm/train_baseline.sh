#!/bin/bash
#SBATCH --job-name=train-baseline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train-%j.log

mkdir -p logs checkpoints

module load cuda/12.3.0

echo "GPU info:"
nvidia-smi

echo "Starting training..."
python train_baseline.py
