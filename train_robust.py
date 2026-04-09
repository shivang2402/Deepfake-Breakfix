"""Train adversarially robust model."""
import torch
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.models.classifier import get_model
from src.models.evaluate import evaluate_model, print_metrics
from src.defense.adversarial_training import train_robust_model


def main():
    config = load_config("configs/default.yaml")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    loaders = create_dataloaders(
        data_dir="data/real-vs-fake",
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # start from pretrained, not from our baseline checkpoint
    # this way we train fresh with adversarial augmentation
    arch = config["model"]["architecture"]
    print(f"Training robust {arch}...")
    model = get_model(arch=arch, num_classes=config["model"]["num_classes"])

    model, best_acc = train_robust_model(model, loaders["train"], loaders["valid"], config, device)
    print(f"\nBest validation accuracy: {best_acc:.4f}")

    # evaluate on clean test set
    print("\nClean test set results:")
    model.load_state_dict(torch.load(f"checkpoints/best_{arch}_robust.pth", map_location=device, weights_only=True))
    model.to(device)
    metrics = evaluate_model(model, loaders["test"], device)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
