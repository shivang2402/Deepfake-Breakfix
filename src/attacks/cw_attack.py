# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Carlini and Wagner L2 attack wrapper

"""Carlini & Wagner L2 attack wrapper using torchattacks."""
import torchattacks


def get_cw(model, c=1.0, kappa=0, steps=50, lr=0.01):
    """C&W L2 attack. Finds minimal perturbation to flip prediction."""
    return torchattacks.CW(model, c=c, kappa=kappa, steps=steps, lr=lr)
