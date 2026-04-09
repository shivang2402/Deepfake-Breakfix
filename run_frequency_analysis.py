"""Run frequency-domain analysis on the dataset."""
import os
import torch
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.attacks.frequency import analyze_frequency, plot_frequency_comparison


def main():
    config = load_config("configs/default.yaml")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    loaders = create_dataloaders(
        data_dir="data/real-vs-fake",
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    print("Analyzing frequency spectra...")
    fake_avg, real_avg = analyze_frequency(loaders["test"], device, n_samples=1000)

    os.makedirs("results/plots", exist_ok=True)
    plot_frequency_comparison(fake_avg, real_avg, "results/plots/frequency_analysis.png")
    print("Done!")


if __name__ == "__main__":
    main()
