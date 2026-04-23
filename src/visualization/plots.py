# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Degradation curves and baseline vs robust comparison plots

"""Plotting degradation curves and comparison charts."""
import json
import matplotlib.pyplot as plt
import os


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_degradation_curve(results, attack_name, xlabel, save_path):
    """Plot accuracy vs intensity for a single attack."""
    intensities = []
    accs = []
    for intensity, metrics in sorted(results.items(), key=lambda x: float(x[0])):
        intensities.append(float(intensity))
        accs.append(metrics["accuracy"])

    plt.figure(figsize=(8, 5))
    plt.plot(intensities, accs, "o-", linewidth=2, markersize=6)
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs {attack_name}")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison(baseline_results, robust_results, attack_name, xlabel, save_path):
    """Plot baseline vs robust model accuracy for the same attack."""
    b_intensities = []
    b_accs = []
    for intensity, metrics in sorted(baseline_results.items(), key=lambda x: float(x[0])):
        b_intensities.append(float(intensity))
        b_accs.append(metrics["accuracy"])

    r_intensities = []
    r_accs = []
    for intensity, metrics in sorted(robust_results.items(), key=lambda x: float(x[0])):
        r_intensities.append(float(intensity))
        r_accs.append(metrics["accuracy"])

    plt.figure(figsize=(8, 5))
    plt.plot(b_intensities, b_accs, "o-", linewidth=2, markersize=6, label="Baseline")
    plt.plot(r_intensities, r_accs, "s--", linewidth=2, markersize=6, label="Robust")
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.title(f"Baseline vs Robust: {attack_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_all_degradation_curves(results_path, output_dir="results/plots"):
    """Generate all degradation curve plots from a results JSON."""
    os.makedirs(output_dir, exist_ok=True)
    results = load_results(results_path)

    labels = {
        "jpeg": ("JPEG Compression", "JPEG Quality"),
        "gaussian_noise": ("Gaussian Noise", "Sigma"),
        "gaussian_blur": ("Gaussian Blur", "Kernel Size"),
        "downscale": ("Downscale/Upscale", "Scale Factor"),
        "fgsm": ("FGSM Attack", "Epsilon"),
        "pgd": ("PGD Attack", "Epsilon"),
    }

    for attack_name, attack_results in results.items():
        title, xlabel = labels.get(attack_name, (attack_name, "Intensity"))
        save_path = os.path.join(output_dir, f"degradation_{attack_name}.png")
        plot_degradation_curve(attack_results, title, xlabel, save_path)


def plot_all_comparisons(baseline_path, robust_path, output_dir="results/plots"):
    """Generate comparison plots for all attacks."""
    os.makedirs(output_dir, exist_ok=True)
    baseline = load_results(baseline_path)
    robust = load_results(robust_path)

    labels = {
        "jpeg": ("JPEG Compression", "JPEG Quality"),
        "gaussian_noise": ("Gaussian Noise", "Sigma"),
        "gaussian_blur": ("Gaussian Blur", "Kernel Size"),
        "downscale": ("Downscale/Upscale", "Scale Factor"),
        "fgsm": ("FGSM Attack", "Epsilon"),
        "pgd": ("PGD Attack", "Epsilon"),
    }

    for attack_name in baseline:
        if attack_name in robust:
            title, xlabel = labels.get(attack_name, (attack_name, "Intensity"))
            save_path = os.path.join(output_dir, f"comparison_{attack_name}.png")
            plot_comparison(baseline[attack_name], robust[attack_name], title, xlabel, save_path)
