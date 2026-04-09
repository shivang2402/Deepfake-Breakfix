#!/bin/bash
#SBATCH --job-name=train-afsl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train-afsl-%j.log

mkdir -p logs checkpoints
module load cuda/12.3.0

echo "Training AFSL model..."
nvidia-smi
python train_afsl.py
