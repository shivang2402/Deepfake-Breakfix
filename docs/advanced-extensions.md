# Advanced Extensions

## EfficientNet-B0 Baseline

Second architecture for comparison. Shows whether the vulnerability is model-specific or general.

```bash
python train_efficientnet.py
```

## Black-box Transferability

Generate adversarial examples on one model, test if they fool the other. If attacks transfer between ResNet-18 and EfficientNet-B0, the threat is worse than just white-box.

```bash
python run_transferability.py
```

Tests both directions (resnet->effnet, effnet->resnet) with PGD and FGSM at multiple epsilons.

## Fine-grained Epsilon Sweep

More epsilon values (0.001, 0.003, 0.008, 0.015, 0.03, 0.06) to find exactly where the model starts breaking.

## C&W Attack (Carlini & Wagner)

L2 attack that finds the smallest perturbation needed to flip the prediction. Stronger than PGD — tests the true minimum robustness of the model.

## EOT Attack (Expectation over Transformation)

Adversarial perturbations that are designed to survive JPEG compression and resizing. This is what would happen in practice — attacker posts a manipulated image on social media, platform compresses it, and the attack still works.

```bash
python run_advanced_attacks.py  # runs fine eps, C&W, and EOT
```

## Frequency Analysis

Compares the frequency spectrum (FFT) of real vs fake images. Fakes often have artifacts in high-frequency components that are invisible in pixel space. This tells us what the model might be relying on.

```bash
python run_frequency_analysis.py
```

## Ensemble Defense

Combine ResNet-18 and EfficientNet-B0 predictions by averaging softmax outputs. Harder to fool two different architectures at the same time. Tests both white-box (attack the ensemble directly) and black-box (attack one model, test on ensemble).

```bash
python run_ensemble.py
```

## AFSL Defense (Adversarial Feature Similarity Learning)

Based on Goswami et al. (2024). Trains the model so that clean and adversarial versions of the same image produce similar internal features. Should be stronger than basic adversarial training.

```bash
python train_afsl.py
python run_attacks_afsl.py
```

## Running Everything on Cluster

```bash
bash slurm/run_all_advanced.sh
```

This submits all jobs with proper dependencies. Independent jobs run in parallel:
- Parallel: train_effnet, frequency, advanced_attacks, train_afsl
- After effnet done: transferability, ensemble
- After afsl done: attacks on afsl model
