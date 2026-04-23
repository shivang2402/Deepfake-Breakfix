# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# FGSM and PGD attack wrappers using torchattacks

"""Adversarial attacks using torchattacks — FGSM and PGD."""
import torch
import torchattacks


def get_fgsm(model, eps=0.01):
    return torchattacks.FGSM(model, eps=eps)


def get_pgd(model, eps=0.01, alpha=0.005, steps=10):
    return torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)


def attack_batch(attack, images, labels):
    """Run attack on a batch, return adversarial images."""
    adv_images = attack(images, labels)
    return adv_images
