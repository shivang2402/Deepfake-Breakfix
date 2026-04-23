# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Evaluate a model on a dataloader and print metrics

import torch
from tqdm import tqdm
from src.utils.metrics import compute_metrics


def evaluate_model(model, loader, device):
    """Run model on a dataloader and return all metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    metrics = compute_metrics(all_labels, all_preds)
    return metrics


def print_metrics(metrics):
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
