# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Generate Grad CAM heatmaps for baseline and robust models

# standard library
import os
import sys

# third party
import torch
import torchattacks

# local
from src.utils.config import load_config
from src.data.dataset import DeepfakeDataset, get_transforms
from src.models.classifier import get_model
from src.visualization.gradcam import save_gradcam_grid


# pick the best available device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# get the last convolutional layer for Grad CAM
def get_target_layer(model, arch):
    if arch == "resnet18":
        return model.layer4[-1]
    if arch == "efficientnet_b0":
        return model.features[-1]
    return None


# pick n samples, half fake half real
def pick_samples(dataset, n=4):
    fakes, reals = [], []
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label == 0 and len(fakes) < n // 2:
            fakes.append((img, label, "fake"))
        elif label == 1 and len(reals) < n // 2:
            reals.append((img, label, "real"))
        if len(fakes) >= n // 2 and len(reals) >= n // 2:
            break
    return fakes + reals


# generate PGD adversarial versions of a list of samples
def make_adv_samples(samples, model, device):
    attack = torchattacks.PGD(model, eps=0.02, alpha=0.005, steps=10)
    adv_samples = []
    for img, label, desc in samples:
        adv_img = attack(img.unsqueeze(0).to(device), torch.tensor([label]).to(device))
        adv_samples.append((adv_img.squeeze(0).cpu(), label, "%s + PGD" % desc))
    return adv_samples


# main function
def main(argv):
    # load config and pick device
    config = load_config("configs/default.yaml")
    device = get_device()
    print("Using device:", device)

    os.makedirs("results/gradcam", exist_ok=True)

    # pick a small sample set from the test split
    arch = config["model"]["architecture"]
    tf = get_transforms(config["data"]["image_size"], split="test")
    dataset = DeepfakeDataset("data/real-vs-fake", split="test", transform=tf)
    samples = pick_samples(dataset, n=4)

    # baseline model: clean images
    print("Grad-CAM: baseline model, clean images")
    baseline = get_model(arch=arch, num_classes=2)
    baseline.load_state_dict(torch.load(
        "checkpoints/best_%s.pth" % arch, map_location=device, weights_only=True
    ))
    baseline.to(device).eval()
    target_layer = get_target_layer(baseline, arch)
    save_gradcam_grid(
        baseline, target_layer, samples,
        "results/gradcam/baseline_clean.png", "Baseline Model, Clean Images"
    )

    # baseline model: PGD attacked images
    print("Grad-CAM: baseline model, adversarial images (PGD)")
    adv_samples = make_adv_samples(samples, baseline, device)
    save_gradcam_grid(
        baseline, target_layer, adv_samples,
        "results/gradcam/baseline_adversarial.png", "Baseline Model, PGD Attacked Images"
    )

    # robust model: same two views if checkpoint exists
    robust_path = "checkpoints/best_%s_robust.pth" % arch
    if os.path.exists(robust_path):
        print("Grad-CAM: robust model, clean images")
        robust = get_model(arch=arch, num_classes=2)
        robust.load_state_dict(torch.load(robust_path, map_location=device, weights_only=True))
        robust.to(device).eval()
        target_layer_r = get_target_layer(robust, arch)

        save_gradcam_grid(
            robust, target_layer_r, samples,
            "results/gradcam/robust_clean.png", "Robust Model, Clean Images"
        )

        print("Grad-CAM: robust model, adversarial images (PGD)")
        adv_samples_r = make_adv_samples(samples, robust, device)
        save_gradcam_grid(
            robust, target_layer_r, adv_samples_r,
            "results/gradcam/robust_adversarial.png", "Robust Model, PGD Attacked Images"
        )
    else:
        print("Robust checkpoint not found at %s, skipping robust Grad-CAM" % robust_path)

    print("Done!")
    return


if __name__ == "__main__":
    main(sys.argv)
