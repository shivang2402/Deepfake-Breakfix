# Grad-CAM Visualization

## What it does

Shows where the model is "looking" when making predictions. Helps us understand:
- What features does the model use on clean images?
- Where does it look when images are adversarially attacked?
- Does the robust model focus on different regions?

## How it works

Grad-CAM uses the gradients flowing into the last convolutional layer to produce a heatmap highlighting important regions. For ResNet-18, that's `layer4[-1]`.

## Files

- `src/visualization/gradcam.py` — Grad-CAM generation and grid saving
- `src/visualization/plots.py` — degradation curves and comparison charts
- `run_gradcam.py` — generates all heatmaps
- `run_plots.py` — generates all plots from results JSON files

## How to run

```bash
# generate heatmaps (needs model checkpoints)
python run_gradcam.py

# generate plots (needs results JSON files)
python run_plots.py

# on cluster
sbatch slurm/run_gradcam.sh
sbatch slurm/run_plots.sh
```

## Output

Heatmaps saved to `results/gradcam/`:
- `baseline_clean.png` — baseline model on clean images
- `baseline_adversarial.png` — baseline model on PGD attacked images
- `robust_clean.png` — robust model on clean images
- `robust_adversarial.png` — robust model on PGD attacked images

Plots saved to `results/plots/`:
- `degradation_<attack>.png` — accuracy curves per attack
- `comparison_<attack>.png` — baseline vs robust comparison
