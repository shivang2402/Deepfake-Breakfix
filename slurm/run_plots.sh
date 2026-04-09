#!/bin/bash
#SBATCH --job-name=plots
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/plots-%j.log

mkdir -p logs results/plots

module load cuda/12.3.0

echo "Generating plots..."
python run_plots.py
