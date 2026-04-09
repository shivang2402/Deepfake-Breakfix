"""Ensemble defense — combine predictions from multiple models."""
import torch
import torch.nn.functional as F


class EnsembleModel(torch.nn.Module):
    """Combine multiple models by averaging their softmax outputs."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        outputs = []
        for model in self.models:
            out = model(x)
            probs = F.softmax(out, dim=1)
            outputs.append(probs)
        # average probabilities
        avg = torch.stack(outputs).mean(dim=0)
        return avg
