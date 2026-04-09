"""Generate Grad-CAM heatmaps for baseline and robust models."""
import os
import torch
from src.utils.config import load_config
from src.data.dataset import DeepfakeDataset, get_transforms
from src.models.classifier import get_model
from src.visualization.gradcam import save_gradcam_grid
from src.attacks.degradation import apply_degradation
import torchattacks


def get_target_layer(model, arch):
    """Get the last conv layer for Grad-CAM."""
    if arch == "resnet18":
        return model.layer4[-1]
    elif arch == "efficientnet_b0":
        return model.features[-1]


def pick_samples(dataset, n=4):
    """Pick n samples — half fake, half real."""
    fakes = []
    reals = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label == 0 and len(fakes) < n // 2:
            fakes.append((img, label, "fake"))
        elif label == 1 and len(reals) < n // 2:
            reals.append((img, label, "real"))
        if len(fakes) >= n // 2 and len(reals) >= n // 2:
            break
    return fakes + reals


def main():
    config = load_config("configs/default.yaml")
    os.makedirs("results/gradcam", exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    arch = config["model"]["architecture"]
    tf = get_transforms(config["data"]["image_size"], split="test")
    dataset = DeepfakeDataset("data/real-vs-fake", split="test", transform=tf)
    samples = pick_samples(dataset, n=4)

    # --- baseline model on clean images ---
    print("Grad-CAM: baseline model, clean images")
    baseline = get_model(arch=arch, num_classes=2)
    baseline.load_state_dict(torch.load(f"checkpoints/best_{arch}.pth", map_location=device, weights_only=True))
    baseline.to(device)
    baseline.eval()

    target_layer = get_target_layer(baseline, arch)
    save_gradcam_grid(baseline, target_layer, samples,
                      "results/gradcam/baseline_clean.png", "Baseline Model — Clean Images")

    # --- baseline model on adversarial images ---
    print("Grad-CAM: baseline model, adversarial images (PGD)")
    attack = torchattacks.PGD(baseline, eps=0.02, alpha=0.005, steps=10)
    adv_samples = []
    for img, label, desc in samples:
        adv_img = attack(img.unsqueeze(0).to(device), torch.tensor([label]).to(device))
        adv_samples.append((adv_img.squeeze(0).cpu(), label, f"{desc} + PGD"))

    save_gradcam_grid(baseline, target_layer, adv_samples,
                      "results/gradcam/baseline_adversarial.png", "Baseline Model — PGD Attacked Images")

    # --- robust model on clean images ---
    robust_path = f"checkpoints/best_{arch}_robust.pth"
    if os.path.exists(robust_path):
        print("Grad-CAM: robust model, clean images")
        robust = get_model(arch=arch, num_classes=2)
        robust.load_state_dict(torch.load(robust_path, map_location=device, weights_only=True))
        robust.to(device)
        robust.eval()

        target_layer_r = get_target_layer(robust, arch)
        save_gradcam_grid(robust, target_layer_r, samples,
                          "results/gradcam/robust_clean.png", "Robust Model — Clean Images")

        # --- robust model on adversarial images ---
        print("Grad-CAM: robust model, adversarial images (PGD)")
        attack_r = torchattacks.PGD(robust, eps=0.02, alpha=0.005, steps=10)
        adv_samples_r = []
        for img, label, desc in samples:
            adv_img = attack_r(img.unsqueeze(0).to(device), torch.tensor([label]).to(device))
            adv_samples_r.append((adv_img.squeeze(0).cpu(), label, f"{desc} + PGD"))

        save_gradcam_grid(robust, target_layer_r, adv_samples_r,
                          "results/gradcam/robust_adversarial.png", "Robust Model — PGD Attacked Images")
    else:
        print(f"Robust checkpoint not found at {robust_path}, skipping robust Grad-CAM")

    print("Done!")


if __name__ == "__main__":
    main()
