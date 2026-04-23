# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Generate additional figures for the IEEE report

# standard library
import io
import os
import sys

# third party
import numpy as np
import torch
import torchattacks
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# local
from src.utils.config import load_config
from src.data.dataset import DeepfakeDataset, get_transforms
from src.models.classifier import get_model


def unnormalize(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = t.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def sample_grid(dataset, n_per_class=4, save_path="results/report/samples.png"):
    """Grid of real and fake samples."""
    fakes, reals = [], []
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label == 0 and len(fakes) < n_per_class:
            fakes.append(img)
        elif label == 1 and len(reals) < n_per_class:
            reals.append(img)
        if len(fakes) >= n_per_class and len(reals) >= n_per_class:
            break

    fig, axes = plt.subplots(2, n_per_class, figsize=(n_per_class * 2.5, 5))
    for i in range(n_per_class):
        axes[0, i].imshow(unnormalize(fakes[i]))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Fake", fontsize=12, fontweight="bold")
        axes[1, i].imshow(unnormalize(reals[i]))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Real", fontsize=12, fontweight="bold")

    # add row labels
    fig.text(0.02, 0.75, "Fake", fontsize=13, fontweight="bold", rotation=90, va="center")
    fig.text(0.02, 0.28, "Real", fontsize=13, fontweight="bold", rotation=90, va="center")
    plt.suptitle("Dataset Samples: Real vs Fake Faces", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def attack_viz(model, dataset, device, save_path="results/report/attack_viz.png"):
    """Show original, perturbation (amplified), and adversarial side by side."""
    # pick one fake, one real
    samples = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        if len(samples) == 0 and label == 0:
            samples.append((img, label, "fake"))
        elif len(samples) == 1 and label == 1:
            samples.append((img, label, "real"))
        if len(samples) == 2:
            break

    attack = torchattacks.PGD(model, eps=0.02, alpha=0.005, steps=10)

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))

    for row, (img, label, name) in enumerate(samples):
        img_t = img.unsqueeze(0).to(device)
        label_t = torch.tensor([label]).to(device)
        adv = attack(img_t, label_t)

        # perturbation (amplified 10x for visibility)
        perturb = (adv - img_t) * 10
        perturb_vis = unnormalize(perturb.squeeze(0))
        perturb_vis = (perturb_vis - perturb_vis.min()) / (perturb_vis.max() - perturb_vis.min() + 1e-8)

        axes[row, 0].imshow(unnormalize(img))
        axes[row, 0].set_title(f"Original ({name})", fontsize=11)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(perturb_vis)
        axes[row, 1].set_title("Perturbation (x10)", fontsize=11)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(unnormalize(adv.squeeze(0)))
        axes[row, 2].set_title(f"Adversarial (eps=0.02)", fontsize=11)
        axes[row, 2].axis("off")

    plt.suptitle("PGD Attack Visualization", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def jpeg_progression(dataset, save_path="results/report/jpeg_progression.png"):
    """Show same face at different JPEG quality levels."""
    img, label = dataset[0]
    qualities = [10, 30, 50, 70, 90]

    fig, axes = plt.subplots(1, len(qualities) + 1, figsize=(15, 3))

    axes[0].imshow(unnormalize(img))
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    raw = (img * std + mean).clamp(0, 1)
    pil = transforms.ToPILImage()(raw)

    for i, q in enumerate(qualities):
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        compressed = Image.open(buf)
        axes[i + 1].imshow(np.array(compressed))
        axes[i + 1].set_title(f"Q = {q}", fontsize=11)
        axes[i + 1].axis("off")

    plt.suptitle("JPEG Compression Progression", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def epsilon_progression(model, dataset, device, save_path="results/report/epsilon_progression.png"):
    """Same face attacked at different epsilon values."""
    img, label = dataset[0]
    eps_values = [0.005, 0.01, 0.02, 0.04, 0.08]

    fig, axes = plt.subplots(1, len(eps_values) + 1, figsize=(15, 3))

    axes[0].imshow(unnormalize(img))
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    img_t = img.unsqueeze(0).to(device)
    label_t = torch.tensor([label]).to(device)

    for i, eps in enumerate(eps_values):
        attack = torchattacks.PGD(model, eps=eps, alpha=eps / 4, steps=10)
        adv = attack(img_t, label_t)
        axes[i + 1].imshow(unnormalize(adv.squeeze(0)))
        axes[i + 1].set_title(f"eps = {eps}", fontsize=11)
        axes[i + 1].axis("off")

    plt.suptitle("PGD Attack at Different Epsilon Values", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def defense_bar_chart(save_path="results/report/defense_bars.png"):
    """Bar chart comparing defenses across attack strengths."""
    attacks = ["Clean", "FGSM\neps=0.01", "PGD\neps=0.005", "PGD\neps=0.01", "PGD\neps=0.02"]
    baseline = [99.56, 45.83, 30.76, 8.47, 0.79]
    adv_train = [98.92, 49.83, 49.39, 46.57, 33.55]
    afsl = [88.30, 59.14, 46.88, 24.12, 8.19]
    ensemble_bb = [99.0, 60.0, 45.77, 27.87, 7.02]

    x = np.arange(len(attacks))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 1.5 * width, baseline, width, label="Baseline", color="#C62828")
    ax.bar(x - 0.5 * width, adv_train, width, label="Adversarial Training", color="#2E7D32")
    ax.bar(x + 0.5 * width, afsl, width, label="AFSL", color="#1565C0")
    ax.bar(x + 1.5 * width, ensemble_bb, width, label="Ensemble (black box)", color="#E65100")

    ax.set_xlabel("Attack Setting", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Defense Comparison Across Attack Strengths", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def transferability_heatmap(save_path="results/report/transferability_heatmap.png"):
    """Heatmap showing attack transfer between models."""
    eps_vals = [0.001, 0.003, 0.005, 0.01, 0.02, 0.04, 0.08]

    # from results/tables/transferability_results.json
    res_to_eff = [0.7242, 0.7141, 0.7066, 0.6884, 0.6437, 0.5141, 0.3199]  # target effnet
    eff_to_res = [0.6183, 0.6107, 0.6048, 0.6011, 0.5904, 0.5793, 0.5448]  # target resnet

    data = np.array([res_to_eff, eff_to_res]) * 100

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(eps_vals)))
    ax.set_xticklabels([f"{e}" for e in eps_vals])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ResNet -> EffNet", "EffNet -> ResNet"])
    ax.set_xlabel("PGD Epsilon", fontsize=11)
    ax.set_title("Black Box Transferability: Target Model Accuracy (%)", fontsize=13, fontweight="bold")

    # add text annotations
    for i in range(2):
        for j in range(len(eps_vals)):
            color = "white" if data[i, j] < 40 else "black"
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                    color=color, fontweight="bold", fontsize=9)

    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def confusion_matrices(save_path="results/report/confusion_matrices.png"):
    """Confusion matrix heatmaps for baseline, robust, AFSL."""
    matrices = {
        "Baseline ResNet-18": [[9968, 32], [50, 9950]],
        "Adversarial Training": [[9908, 92], [124, 9876]],
        "AFSL Defense": [[8774, 1226], [1114, 8886]],
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, (name, mat) in zip(axes, matrices.items()):
        mat = np.array(mat)
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=10000)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Fake", "Real"])
        ax.set_yticklabels(["Fake", "Real"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(2):
            for j in range(2):
                color = "white" if mat[i, j] > 5000 else "black"
                ax.text(j, i, f"{mat[i, j]}", ha="center", va="center",
                        color=color, fontweight="bold", fontsize=11)

    plt.suptitle("Confusion Matrices on Clean Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# pick the best available device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# main function
def main(argv):
    os.makedirs("results/report", exist_ok=True)
    config = load_config("configs/default.yaml")
    device = get_device()
    print("Device:", device)

    # load test dataset
    tf = get_transforms(224, split="test")
    dataset = DeepfakeDataset("data/real-vs-fake", split="test", transform=tf)

    # static plots that do not require a trained model
    sample_grid(dataset)
    jpeg_progression(dataset)
    defense_bar_chart()
    transferability_heatmap()
    confusion_matrices()

    # model based plots (require baseline checkpoint)
    arch = config["model"]["architecture"]
    ckpt = "checkpoints/best_%s.pth" % arch
    if os.path.exists(ckpt):
        model = get_model(arch=arch, num_classes=2)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device).eval()
        attack_viz(model, dataset, device)
        epsilon_progression(model, dataset, device)
    else:
        print("No checkpoint at %s, skipping attack visualizations" % ckpt)

    print("\nAll report images generated in results/report/")
    return


if __name__ == "__main__":
    main(sys.argv)
