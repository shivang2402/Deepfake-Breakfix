#!/bin/bash
#SBATCH --job-name=run-attacks
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/attacks-%j.log

mkdir -p logs results/tables

module load cuda/12.3.0

echo "GPU info:"
nvidia-smi

echo "Running attacks..."
python run_attacks.py
