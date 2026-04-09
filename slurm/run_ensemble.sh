#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ensemble-%j.log

mkdir -p logs results/tables
module load cuda/12.3.0

echo "Running ensemble evaluation..."
nvidia-smi
python run_ensemble.py
