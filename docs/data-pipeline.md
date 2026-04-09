# Data Pipeline

## What it does

Loads the 140k Real and Fake Faces dataset and gives you batches ready for training/eval.

## Files

### `src/data/dataset.py`

**`DeepfakeDataset`** — custom PyTorch Dataset.

- Reads from `train/`, `valid/`, or `test/` split
- Each split has `fake/` (label 0) and `real/` (label 1) folders
- Returns `(image_tensor, label)`
- `get_class_counts()` gives you how many samples per class

**`get_transforms(image_size, split)`** — gives transforms based on split.

- Train: resize, random flip, rotation, color jitter, normalize (ImageNet stats)
- Valid/Test: just resize and normalize, no augmentation

### `src/data/dataloader.py`

**`create_dataloaders(data_dir, image_size, batch_size, num_workers)`**

- Makes DataLoaders for train, valid, test
- Train loader shuffles and drops last incomplete batch
- Uses pin_memory for faster GPU transfer

## How to use

```python
from src.data.dataloader import create_dataloaders

loaders = create_dataloaders("data/real-vs-fake", image_size=224, batch_size=32)

for images, labels in loaders["train"]:
    # images: (B, 3, 224, 224), labels: (B,)
    pass
```

## Stats

| Split | Fake | Real | Total |
|-------|------|------|-------|
| Train | 50k | 50k | 100k |
| Valid | 10k | 10k | 20k |
| Test  | 10k | 10k | 20k |

Perfectly balanced, no need for weighted sampling.
