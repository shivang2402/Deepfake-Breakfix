# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Compute and plot frequency spectrum comparison of real vs fake faces

# standard library
import os
import sys

# third party
import torch

# local
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.attacks.frequency import analyze_frequency, plot_frequency_comparison


# pick the best available device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# main function
def main(argv):
    # load config and pick device
    config = load_config("configs/default.yaml")
    device = get_device()

    # build test loader
    loaders = create_dataloaders(
        data_dir="data/real-vs-fake",
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # compute average FFT spectrum of real vs fake and save plot
    print("Analyzing frequency spectra...")
    fake_avg, real_avg = analyze_frequency(loaders["test"], device, n_samples=1000)

    os.makedirs("results/plots", exist_ok=True)
    plot_frequency_comparison(fake_avg, real_avg, "results/plots/frequency_analysis.png")
    print("Done!")

    return


if __name__ == "__main__":
    main(sys.argv)
