# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Train model with AFSL defense (Adversarial Feature Similarity Learning)

# standard library
import sys

# third party
import torch

# local
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.models.classifier import get_model
from src.models.evaluate import evaluate_model, print_metrics
from src.defense.afsl import train_afsl


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

    # build model and train with AFSL defense
    arch = config["model"]["architecture"]
    print("Training %s with AFSL..." % arch)
    model = get_model(arch=arch, num_classes=config["model"]["num_classes"])
    model, best_acc = train_afsl(model, arch, loaders["train"], loaders["valid"], config, device)
    print("\nBest validation accuracy: %.4f" % best_acc)

    # load best AFSL checkpoint and evaluate on clean test set
    print("\nClean test set results:")
    ckpt_path = "checkpoints/best_%s_afsl.pth" % arch
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    metrics = evaluate_model(model, loaders["test"], device)
    print_metrics(metrics)

    return


if __name__ == "__main__":
    main(sys.argv)
