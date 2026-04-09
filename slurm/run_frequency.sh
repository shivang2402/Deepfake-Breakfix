#!/bin/bash
#SBATCH --job-name=frequency
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/frequency-%j.log

mkdir -p logs results/plots
module load cuda/12.3.0

echo "Running frequency analysis..."
python run_frequency_analysis.py
