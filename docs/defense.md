# Adversarial Training (Defense)

## Idea

The baseline model breaks badly under adversarial attacks. So we retrain it with a mix of clean and adversarial examples during training. This forces the model to learn features that are harder to fool.

## How it works

During each training batch:
1. Take the batch and split it based on `mix_ratio` (default 50/50)
2. First half: generate PGD adversarial examples
3. Second half: keep as clean images
4. Train on the combined batch

We start from pretrained weights (not from baseline checkpoint) so the model learns fresh with adversarial data from the start.

## Files

- `src/defense/adversarial_training.py` — adversarial training loop
- `train_robust.py` — main script
- `run_attacks_robust.py` — run same attacks on the robust model

## Config

In `configs/default.yaml`:

```yaml
adversarial_training:
  mix_ratio: 0.5    # 50% adversarial examples per batch
  attack: "pgd"
  epsilon: 0.02
  epochs: 20
```

## How to run

```bash
# train robust model
python train_robust.py

# run attacks on robust model
python run_attacks_robust.py

# on cluster
sbatch slurm/train_robust.sh
# then after it finishes:
sbatch slurm/run_attacks_robust.sh
```

## What to expect

- Clean accuracy will probably drop a bit (maybe 1-3%)
- But accuracy under attacks should be much better, especially for FGSM/PGD
- Degradation attacks (JPEG, noise) might also improve since adversarial training acts as regularization
