#!/bin/bash
# Fix: retrain ResNet-18 (checkpoint was overwritten), retrain EfficientNet with fix,
# then rerun transferability and ensemble.

echo "Submitting fix pipeline..."

# 1. Retrain ResNet-18 baseline
JOB1=$(sbatch --parsable slurm/train_baseline.sh)
echo "Submitted train_baseline (resnet): $JOB1"

# 2. Retrain EfficientNet-B0 (with config fix)
JOB2=$(sbatch --parsable slurm/train_efficientnet.sh)
echo "Submitted train_efficientnet: $JOB2"

# 3. Transferability (needs both)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1:$JOB2 slurm/run_transferability.sh)
echo "Submitted transferability: $JOB3 (depends on $JOB1 and $JOB2)"

# 4. Ensemble (needs both)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB1:$JOB2 slurm/run_ensemble.sh)
echo "Submitted ensemble: $JOB4 (depends on $JOB1 and $JOB2)"

echo ""
echo "Check with: squeue -u \$USER"
