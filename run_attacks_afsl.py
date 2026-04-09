"""Run all attacks on the AFSL model and save results."""
import json
import os
import torch
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.models.classifier import get_model
from src.attacks.adversarial import get_fgsm, get_pgd
from src.attacks.evaluate_attacks import eval_degradation, eval_adversarial


def main():
    config = load_config("configs/default.yaml")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    loaders = create_dataloaders(
        data_dir="data/real-vs-fake",
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    test_loader = loaders["test"]

    arch = config["model"]["architecture"]
    model = get_model(arch=arch, num_classes=2)
    model.load_state_dict(torch.load(f"checkpoints/best_{arch}_afsl.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_results = {}

    print("\n=== FGSM on AFSL ===")
    fgsm = get_fgsm(model, eps=0.01)
    all_results["fgsm"] = eval_adversarial(
        model, test_loader, fgsm,
        config["attacks"]["fgsm_epsilon"], device, "fgsm"
    )

    print("\n=== PGD on AFSL ===")
    pgd = get_pgd(model, eps=0.01, alpha=config["attacks"]["pgd_alpha"], steps=config["attacks"]["pgd_steps"])
    all_results["pgd"] = eval_adversarial(
        model, test_loader, pgd,
        config["attacks"]["pgd_epsilon"], device, "pgd"
    )

    # save
    os.makedirs("results/tables", exist_ok=True)
    clean_results = {}
    for attack_name, attack_results in all_results.items():
        clean_results[attack_name] = {}
        for intensity, metrics in attack_results.items():
            clean_results[attack_name][str(intensity)] = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "confusion_matrix": metrics["confusion_matrix"].tolist(),
            }

    with open("results/tables/attack_results_afsl.json", "w") as f:
        json.dump(clean_results, f, indent=2)
    print("\nSaved to results/tables/attack_results_afsl.json")


if __name__ == "__main__":
    main()
