# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Load pretrained ResNet 18 or EfficientNet B0 with a 2 way head

import torch.nn as nn
from torchvision import models


def get_model(arch="resnet18", num_classes=2, pretrained=True):
    """Load a pretrained model and swap the last layer for our task."""
    if arch == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model
