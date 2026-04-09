#!/bin/bash
# Submit all jobs with dependencies so they run in order overnight.
# Usage: bash slurm/run_all_overnight.sh

echo "Submitting overnight pipeline..."

# 1. Train robust model (waits for nothing — baseline attacks already running)
JOB1=$(sbatch --parsable slurm/train_robust.sh)
echo "Submitted train_robust: $JOB1"

# 2. Run attacks on robust model (waits for robust training)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/run_attacks_robust.sh)
echo "Submitted attacks_robust: $JOB2 (depends on $JOB1)"

# 3. Grad-CAM (waits for robust training — needs both checkpoints)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/run_gradcam.sh)
echo "Submitted gradcam: $JOB3 (depends on $JOB1)"

# 4. Plots (waits for both baseline and robust attacks to finish)
# baseline attacks job should already be done by now, but we wait for robust attacks
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/run_plots.sh)
echo "Submitted plots: $JOB4 (depends on $JOB2)"

echo ""
echo "Pipeline submitted! Check with: squeue -u \$USER"
echo "Jobs will run in order: train_robust -> attacks_robust + gradcam -> plots"
