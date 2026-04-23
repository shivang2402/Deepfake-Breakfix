# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Train EfficientNet B0 baseline for architecture comparison

# standard library
import sys

# third party
import torch

# local
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.models.classifier import get_model
from src.models.train import train_model
from src.models.evaluate import evaluate_model, print_metrics


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
    print("Using device:", device)

    # build data loaders
    loaders = create_dataloaders(
        data_dir="data/real-vs-fake",
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # override arch to efficientnet b0 and train
    config["model"]["architecture"] = "efficientnet_b0"
    print("Training efficientnet_b0...")
    model = get_model(arch="efficientnet_b0", num_classes=config["model"]["num_classes"])
    model, best_acc = train_model(model, loaders["train"], loaders["valid"], config, device)
    print("\nBest validation accuracy: %.4f" % best_acc)

    # load best checkpoint and evaluate
    print("\nTest set results:")
    model.load_state_dict(torch.load(
        "checkpoints/best_efficientnet_b0.pth",
        map_location=device, weights_only=True
    ))
    model.to(device)
    metrics = evaluate_model(model, loaders["test"], device)
    print_metrics(metrics)

    return


if __name__ == "__main__":
    main(sys.argv)
