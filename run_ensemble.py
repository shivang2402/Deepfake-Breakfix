# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Evaluate ensemble defense (ResNet 18 + EfficientNet B0)

# standard library
import json
import os
import sys

# third party
import torch
import torchattacks
from tqdm import tqdm

# local
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.models.classifier import get_model
from src.defense.ensemble import EnsembleModel
from src.utils.metrics import compute_metrics


# pick the best available device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# evaluate ensemble against an optional attack
def eval_ensemble(ensemble, loader, attack, device, desc=""):
    ensemble.eval()
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        adv_imgs = attack(imgs, labels) if attack is not None else imgs
        with torch.no_grad():
            probs = ensemble(adv_imgs)
            preds = probs.argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    return compute_metrics(all_labels, all_preds)


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

    # load both baseline models and build ensemble
    resnet = get_model(arch="resnet18", num_classes=2)
    resnet.load_state_dict(torch.load("checkpoints/best_resnet18.pth", map_location=device, weights_only=True))
    resnet.to(device)
    effnet = get_model(arch="efficientnet_b0", num_classes=2)
    effnet.load_state_dict(torch.load("checkpoints/best_efficientnet_b0.pth", map_location=device, weights_only=True))
    effnet.to(device)
    ensemble = EnsembleModel([resnet, effnet]).to(device)

    results = {}

    # clean accuracy baseline
    print("Ensemble clean accuracy:")
    m = eval_ensemble(ensemble, test_loader, None, device, "Clean")
    results["clean"] = {"accuracy": m["accuracy"], "f1": m["f1"]}
    print("  Accuracy: %.4f" % m["accuracy"])

    eps_values = [0.005, 0.01, 0.02, 0.04, 0.08]

    # white box FGSM on the full ensemble
    print("\n=== FGSM on ensemble ===")
    results["fgsm"] = {}
    for eps in eps_values:
        attack = torchattacks.FGSM(ensemble, eps=eps)
        m = eval_ensemble(ensemble, test_loader, attack, device, "FGSM eps=%s" % eps)
        results["fgsm"][str(eps)] = {"accuracy": m["accuracy"], "f1": m["f1"]}
        print("  FGSM eps=%s: acc=%.4f" % (eps, m["accuracy"]))

    # white box PGD on the full ensemble
    print("\n=== PGD on ensemble ===")
    results["pgd"] = {}
    for eps in eps_values:
        attack = torchattacks.PGD(ensemble, eps=eps, alpha=eps / 4, steps=10)
        m = eval_ensemble(ensemble, test_loader, attack, device, "PGD eps=%s" % eps)
        results["pgd"][str(eps)] = {"accuracy": m["accuracy"], "f1": m["f1"]}
        print("  PGD eps=%s: acc=%.4f" % (eps, m["accuracy"]))

    # black box: attack only ResNet, evaluate on full ensemble
    print("\n=== Black-box: attack ResNet, test ensemble ===")
    results["blackbox_resnet"] = {}
    for eps in eps_values:
        attack = torchattacks.PGD(resnet, eps=eps, alpha=eps / 4, steps=10)
        m = eval_ensemble(ensemble, test_loader, attack, device, "BB PGD eps=%s" % eps)
        results["blackbox_resnet"][str(eps)] = {"accuracy": m["accuracy"], "f1": m["f1"]}
        print("  BB PGD eps=%s: acc=%.4f" % (eps, m["accuracy"]))

    # save results
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/tables/ensemble_results.json")

    return


if __name__ == "__main__":
    main(sys.argv)
