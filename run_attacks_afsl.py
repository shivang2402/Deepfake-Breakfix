# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Run FGSM and PGD attacks on AFSL defended model

# standard library
import json
import os
import sys

# third party
import torch

# local
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.models.classifier import get_model
from src.attacks.adversarial import get_fgsm, get_pgd
from src.attacks.evaluate_attacks import eval_adversarial


# pick the best available device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# convert metrics dict with numpy arrays into plain types for json
def clean_for_json(all_results):
    out = {}
    for attack_name, attack_results in all_results.items():
        out[attack_name] = {}
        for intensity, metrics in attack_results.items():
            out[attack_name][str(intensity)] = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "confusion_matrix": metrics["confusion_matrix"].tolist(),
            }
    return out


# main function
def main(argv):
    # load config and pick device
    config = load_config("configs/default.yaml")
    device = get_device()
    print("Using device:", device)

    # build test loader
    loaders = create_dataloaders(
        data_dir="data/real-vs-fake",
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    test_loader = loaders["test"]

    # load AFSL checkpoint
    arch = config["model"]["architecture"]
    model = get_model(arch=arch, num_classes=2)
    ckpt_path = "checkpoints/best_%s_afsl.pth" % arch
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_results = {}

    # run FGSM and PGD on AFSL model
    print("\n=== FGSM on AFSL ===")
    fgsm = get_fgsm(model, eps=0.01)
    all_results["fgsm"] = eval_adversarial(
        model, test_loader, fgsm, config["attacks"]["fgsm_epsilon"], device, "fgsm"
    )

    print("\n=== PGD on AFSL ===")
    pgd = get_pgd(
        model, eps=0.01,
        alpha=config["attacks"]["pgd_alpha"], steps=config["attacks"]["pgd_steps"]
    )
    all_results["pgd"] = eval_adversarial(
        model, test_loader, pgd, config["attacks"]["pgd_epsilon"], device, "pgd"
    )

    # save to json
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/attack_results_afsl.json", "w") as f:
        json.dump(clean_for_json(all_results), f, indent=2)
    print("\nSaved to results/tables/attack_results_afsl.json")

    return


if __name__ == "__main__":
    main(sys.argv)
