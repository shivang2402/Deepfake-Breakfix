#!/bin/bash
#SBATCH --job-name=adv-attacks
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/advanced-attacks-%j.log

mkdir -p logs results/tables
module load cuda/12.3.0

echo "Running advanced attacks (fine eps, C&W, EOT)..."
nvidia-smi
python run_advanced_attacks.py
