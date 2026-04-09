"""Evaluate ensemble defense against attacks."""
import json
import os
import torch
import torchattacks
from tqdm import tqdm
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.models.classifier import get_model
from src.defense.ensemble import EnsembleModel
from src.utils.metrics import compute_metrics


def eval_ensemble(ensemble, loader, attack, device, desc=""):
    ensemble.eval()
    all_preds = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        if attack is not None:
            adv_imgs = attack(imgs, labels)
        else:
            adv_imgs = imgs

        with torch.no_grad():
            probs = ensemble(adv_imgs)
            preds = probs.argmax(1).cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    return compute_metrics(all_labels, all_preds)


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

    # load both models
    resnet = get_model(arch="resnet18", num_classes=2)
    resnet.load_state_dict(torch.load("checkpoints/best_resnet18.pth", map_location=device, weights_only=True))
    resnet.to(device)

    effnet = get_model(arch="efficientnet_b0", num_classes=2)
    effnet.load_state_dict(torch.load("checkpoints/best_efficientnet_b0.pth", map_location=device, weights_only=True))
    effnet.to(device)

    ensemble = EnsembleModel([resnet, effnet]).to(device)

    results = {}

    # clean accuracy
    print("Ensemble clean accuracy:")
    m = eval_ensemble(ensemble, test_loader, None, device, "Clean")
    results["clean"] = {"accuracy": m["accuracy"], "f1": m["f1"]}
    print(f"  Accuracy: {m['accuracy']:.4f}")

    # attack resnet only (black-box for effnet)
    eps_values = [0.005, 0.01, 0.02, 0.04, 0.08]

    print("\n=== FGSM on ensemble ===")
    results["fgsm"] = {}
    for eps in eps_values:
        # attack the ensemble directly
        attack = torchattacks.FGSM(ensemble, eps=eps)
        m = eval_ensemble(ensemble, test_loader, attack, device, f"FGSM eps={eps}")
        results["fgsm"][str(eps)] = {"accuracy": m["accuracy"], "f1": m["f1"]}
        print(f"  FGSM eps={eps}: acc={m['accuracy']:.4f}")

    print("\n=== PGD on ensemble ===")
    results["pgd"] = {}
    for eps in eps_values:
        attack = torchattacks.PGD(ensemble, eps=eps, alpha=eps/4, steps=10)
        m = eval_ensemble(ensemble, test_loader, attack, device, f"PGD eps={eps}")
        results["pgd"][str(eps)] = {"accuracy": m["accuracy"], "f1": m["f1"]}
        print(f"  PGD eps={eps}: acc={m['accuracy']:.4f}")

    # black-box: attack resnet, test on ensemble
    print("\n=== Black-box: attack ResNet, test ensemble ===")
    results["blackbox_resnet"] = {}
    for eps in eps_values:
        attack = torchattacks.PGD(resnet, eps=eps, alpha=eps/4, steps=10)
        m = eval_ensemble(ensemble, test_loader, attack, device, f"BB PGD eps={eps}")
        results["blackbox_resnet"][str(eps)] = {"accuracy": m["accuracy"], "f1": m["f1"]}
        print(f"  BB PGD eps={eps}: acc={m['accuracy']:.4f}")

    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/tables/ensemble_results.json")


if __name__ == "__main__":
    main()
