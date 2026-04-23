# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# 2D FFT radial spectrum analysis of real vs fake images

"""Frequency-domain analysis of real vs fake images."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_dct_spectrum(img_tensor):
    """Compute average DCT magnitude spectrum of an image tensor.
    img_tensor: (C, H, W) normalized tensor
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img_tensor.cpu() * std + mean).clamp(0, 1)

    # convert to grayscale
    gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    gray = gray.numpy()

    # 2D FFT (using FFT as proxy for DCT — faster, similar info)
    spectrum = np.fft.fft2(gray)
    spectrum = np.fft.fftshift(spectrum)
    magnitude = np.log1p(np.abs(spectrum))
    return magnitude


def compute_azimuthal_average(spectrum):
    """Compute azimuthal (radial) average of a 2D spectrum."""
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2

    # distance from center for each pixel
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = min(cy, cx)
    radial_avg = np.zeros(max_r)
    for i in range(max_r):
        mask = r == i
        if mask.any():
            radial_avg[i] = spectrum[mask].mean()

    return radial_avg


def analyze_frequency(loader, device, n_samples=500):
    """Compute average frequency spectra for real and fake images."""
    fake_spectra = []
    real_spectra = []

    count = 0
    for imgs, labels in tqdm(loader, desc="Frequency analysis", leave=False):
        for img, label in zip(imgs, labels):
            spectrum = compute_dct_spectrum(img)
            radial = compute_azimuthal_average(spectrum)

            if label.item() == 0:
                fake_spectra.append(radial)
            else:
                real_spectra.append(radial)

            count += 1
            if count >= n_samples:
                break
        if count >= n_samples:
            break

    fake_avg = np.mean(fake_spectra, axis=0)
    real_avg = np.mean(real_spectra, axis=0)
    return fake_avg, real_avg


def plot_frequency_comparison(fake_avg, real_avg, save_path):
    """Plot radial frequency spectrum comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # radial average comparison
    axes[0].plot(fake_avg, label="Fake", alpha=0.8)
    axes[0].plot(real_avg, label="Real", alpha=0.8)
    axes[0].set_xlabel("Frequency (radial distance)")
    axes[0].set_ylabel("Log Magnitude")
    axes[0].set_title("Radial Frequency Spectrum")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # difference
    diff = fake_avg - real_avg
    axes[1].plot(diff, color="red", alpha=0.8)
    axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    axes[1].set_xlabel("Frequency (radial distance)")
    axes[1].set_ylabel("Difference (Fake - Real)")
    axes[1].set_title("Frequency Difference (Fake - Real)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
