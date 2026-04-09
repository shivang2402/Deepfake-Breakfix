#!/bin/bash
# Submit all advanced extension jobs with dependencies.
# Usage: bash slurm/run_all_advanced.sh

echo "Submitting advanced pipeline..."

# 1. Train EfficientNet-B0 (no dependency)
JOB1=$(sbatch --parsable slurm/train_efficientnet.sh)
echo "Submitted train_efficientnet: $JOB1"

# 2. Frequency analysis (no dependency, no GPU needed)
JOB2=$(sbatch --parsable slurm/run_frequency.sh)
echo "Submitted frequency_analysis: $JOB2"

# 3. Advanced attacks on baseline (no dependency, needs baseline checkpoint)
JOB3=$(sbatch --parsable slurm/run_advanced_attacks.sh)
echo "Submitted advanced_attacks: $JOB3"

# 4. Train AFSL (no dependency)
JOB4=$(sbatch --parsable slurm/train_afsl.sh)
echo "Submitted train_afsl: $JOB4"

# 5. Transferability (needs both models trained)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/run_transferability.sh)
echo "Submitted transferability: $JOB5 (depends on $JOB1)"

# 6. Ensemble (needs both models trained)
JOB6=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/run_ensemble.sh)
echo "Submitted ensemble: $JOB6 (depends on $JOB1)"

# 7. Attacks on AFSL model (needs AFSL training done)
JOB7=$(sbatch --parsable --dependency=afterok:$JOB4 slurm/run_attacks_afsl.sh)
echo "Submitted attacks_afsl: $JOB7 (depends on $JOB4)"

echo ""
echo "Pipeline submitted! Independent jobs run in parallel."
echo "Check with: squeue -u \$USER"
echo ""
echo "Job order:"
echo "  Parallel: train_effnet, frequency, advanced_attacks, train_afsl"
echo "  After effnet: transferability, ensemble"
echo "  After afsl: attacks_afsl"
