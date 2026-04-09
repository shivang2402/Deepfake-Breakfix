#!/bin/bash
#SBATCH --job-name=gradcam
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gradcam-%j.log

mkdir -p logs results/gradcam

module load cuda/12.3.0

echo "Generating Grad-CAM heatmaps..."
python run_gradcam.py
