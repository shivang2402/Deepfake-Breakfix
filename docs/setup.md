# Setup

## Requirements

- Python 3.9+
- PyTorch 2.0+
- GPU helps a lot for training (MPS on Mac, CUDA on cluster)

## Install

```bash
pip install -r requirements.txt
```

## Dataset

We are using [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) from Kaggle.

### Download

```python
import kagglehub
path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
print("Path to dataset files:", path)
```

### Link to project

After downloading, symlink it into the project:

```bash
mkdir -p data/
ln -s /path/to/real_vs_fake/real-vs-fake data/real-vs-fake
```

Folder structure should look like this:

```
data/real-vs-fake/
├── train/
│   ├── fake/   (50k images)
│   └── real/   (50k images)
├── valid/
│   ├── fake/   (10k images)
│   └── real/   (10k images)
└── test/
    ├── fake/   (10k images)
    └── real/   (10k images)
```

## Config

All the hyperparams and attack sweep values are in `configs/default.yaml`. Change batch size, learning rate, epsilon ranges etc. from there.
