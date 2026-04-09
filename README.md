# Deepfake-Breakfix

Testing how easily deepfake detectors break under attacks, and whether adversarial training can fix them.

## What is this

Most deepfake detectors do well on clean images but fall apart when you throw real-world stuff at them — JPEG compression, noise, blur, resizing. On top of that, adversarial attacks like FGSM and PGD can fool detectors with changes you can't even see. This project tests how bad the damage is and tries to make the detector tougher using adversarial training.

## Project Structure

```
├── configs/default.yaml       # all hyperparams and attack sweep ranges
├── docs/                      # setup guide, data pipeline docs
├── results/
│   ├── plots/                 # degradation curves
│   ├── tables/                # metric tables
│   └── gradcam/               # heatmap outputs
├── src/
│   ├── data/                  # dataset + dataloader
│   ├── models/                # model definitions
│   ├── attacks/               # image degradation + adversarial attacks
│   ├── defense/               # adversarial training
│   ├── visualization/         # grad-cam, plots
│   └── utils/                 # config, metrics, helpers
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Check [docs/setup.md](docs/setup.md) for dataset download and environment setup.

## Training

```bash
python train_baseline.py
```

See [docs/baseline-model.md](docs/baseline-model.md) for model details.

## Dataset

[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) from Kaggle.

| Split | Fake | Real | Total |
|-------|------|------|-------|
| Train | 50k | 50k | 100k |
| Valid | 10k | 10k | 20k |
| Test  | 10k | 10k | 20k |

## Attacks

After training, we attack the model with:
- **Degradations**: JPEG compression, Gaussian noise, Gaussian blur, downscale/upscale
- **Adversarial**: FGSM and PGD with epsilon sweeps

```bash
python run_attacks.py
```

See [docs/attacks.md](docs/attacks.md) for details.

## Defense

Retrain the model with a mix of clean + PGD adversarial examples. See [docs/defense.md](docs/defense.md).

```bash
python train_robust.py
python run_attacks_robust.py
```

## Visualization

Grad-CAM heatmaps and comparison plots. See [docs/gradcam.md](docs/gradcam.md).

```bash
python run_gradcam.py
python run_plots.py
```

## Running on Cluster (SLURM)

Submit the full overnight pipeline:

```bash
bash slurm/run_all_overnight.sh
```

This chains: robust training -> attacks on robust model + grad-cam -> plots.

## Progress

- [x] Project setup and data pipeline
- [x] Baseline model — ResNet-18, 99.59% test accuracy
- [ ] Image degradation attacks
- [ ] Adversarial attacks (FGSM, PGD)
- [ ] Adversarial training
- [ ] Grad-CAM visualization
- [ ] Comparison plots
- [ ] Final results

## Tools

Python, PyTorch, OpenCV, torchattacks, pytorch-grad-cam

## Author

Shivang Patel — PRCV Final Project
