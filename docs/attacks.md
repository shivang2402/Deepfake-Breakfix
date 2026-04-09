# Attacks

## What we're testing

The baseline model gets 99.59% accuracy on clean images. Now we throw stuff at it and see how bad it gets.

Two types of attacks:

### 1. Image Degradations

These are things that happen to images in the real world — social media compression, noise, etc.

- **JPEG Compression** — quality levels: 10, 20, 30, 40, 50, 60, 70, 80, 90
- **Gaussian Noise** — sigma: 0.01, 0.02, 0.05, 0.1, 0.15, 0.2
- **Gaussian Blur** — kernel sizes: 3, 5, 7, 9, 11
- **Downscale/Upscale** — factors: 2x, 4x, 8x (shrink then stretch back)

### 2. Adversarial Attacks

These are intentional attacks that add tiny, invisible perturbations to fool the model.

- **FGSM** (Fast Gradient Sign Method) — single-step attack, epsilon: 0.005 to 0.08
- **PGD** (Projected Gradient Descent) — multi-step attack (10 steps), same epsilon range

## Files

- `src/attacks/degradation.py` — image degradation functions
- `src/attacks/adversarial.py` — FGSM and PGD wrappers using torchattacks
- `src/attacks/evaluate_attacks.py` — runs attacks at each intensity and collects metrics
- `run_attacks.py` — main script, runs everything and saves results to JSON

## How to run

```bash
# locally
python run_attacks.py

# on cluster
sbatch slurm/run_attacks.sh
```

## Output

Results saved to `results/tables/attack_results.json` — accuracy, precision, recall, f1, and confusion matrix for every attack at every intensity.
