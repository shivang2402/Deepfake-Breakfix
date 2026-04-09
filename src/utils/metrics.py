import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(all_labels, all_preds):
    """Compute accuracy, precision, recall, f1 and confusion matrix."""
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary")
    rec = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    cm = confusion_matrix(all_labels, all_preds)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}
