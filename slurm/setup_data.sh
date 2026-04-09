#!/bin/bash
#SBATCH --job-name=download-data
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/download-%j.log

mkdir -p logs

module load cuda/12.3.0

# download dataset
python -c "
import kagglehub
path = kagglehub.dataset_download('xhlulu/140k-real-and-fake-faces')
print('Dataset path:', path)
"

# symlink into project
mkdir -p data
DATASET_PATH=$(python -c "import kagglehub; print(kagglehub.dataset_download('xhlulu/140k-real-and-fake-faces'))")
ln -sf "$DATASET_PATH/real_vs_fake/real-vs-fake" data/real-vs-fake

echo "Done. Checking:"
ls data/real-vs-fake/
