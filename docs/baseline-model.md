# Baseline Model

## What it does

Trains a pretrained CNN on the real vs fake face classification task. This is the model we will attack later.

## Models

Two options (set in `configs/default.yaml`):

- **ResNet-18** — lightweight, fast to train, good enough for binary classification
- **EfficientNet-B0** — more efficient architecture, might do slightly better

Both are loaded with ImageNet pretrained weights. The last classification layer is swapped out for 2 classes (fake/real).

## Files

- `src/models/classifier.py` — model loading, swaps the head
- `src/models/train.py` — training loop, validation, early stopping, saves best checkpoint
- `src/models/evaluate.py` — runs model on test set, computes metrics
- `src/utils/metrics.py` — accuracy, precision, recall, f1, confusion matrix
- `train_baseline.py` — main script to run everything

## How to train

```bash
python train_baseline.py
```

Picks up all settings from `configs/default.yaml`. To switch between ResNet-18 and EfficientNet-B0, change the `architecture` field in the config.

## Training details

- Optimizer: Adam
- Scheduler: Cosine annealing
- Early stopping: stops if val accuracy doesn't improve for 5 epochs
- Best model saved to `checkpoints/best_<arch>.pth`

## Metrics tracked

- Accuracy, Precision, Recall, F1
- Confusion matrix
