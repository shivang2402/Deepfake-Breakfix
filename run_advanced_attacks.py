# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Fine grained epsilon sweep, Carlini and Wagner, and EOT PGD attacks

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
from src.utils.metrics import compute_metrics
from src.attacks.eot_attack import eot_pgd
from src.attacks.cw_attack import get_cw


# pick the best available device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# run an attack over the loader and return metrics
def eval_attack_batch(model, loader, attack_fn, device, desc=""):
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        adv_imgs = attack_fn(imgs, labels)
        with torch.no_grad():
            preds = model(adv_imgs).argmax(1).cpu().tolist()
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

    # load baseline model
    arch = config["model"]["architecture"]
    model = get_model(arch=arch, num_classes=2)
    ckpt_path = "checkpoints/best_%s.pth" % arch
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)

    results = {}
    fine_eps = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08]

    # fine grained FGSM sweep
    print("\n=== Fine-grained FGSM ===")
    results["fgsm_fine"] = {}
    for eps in fine_eps:
        attack = torchattacks.FGSM(model, eps=eps)
        m = eval_attack_batch(model, test_loader, attack, device, "FGSM eps=%s" % eps)
        results["fgsm_fine"][str(eps)] = {"accuracy": m["accuracy"], "f1": m["f1"]}
        print("  FGSM eps=%s: acc=%.4f" % (eps, m["accuracy"]))

    # fine grained PGD sweep
    print("\n=== Fine-grained PGD ===")
    results["pgd_fine"] = {}
    for eps in fine_eps:
        attack = torchattacks.PGD(model, eps=eps, alpha=eps / 4, steps=10)
        m = eval_attack_batch(model, test_loader, attack, device, "PGD eps=%s" % eps)
        results["pgd_fine"][str(eps)] = {"accuracy": m["accuracy"], "f1": m["f1"]}
        print("  PGD eps=%s: acc=%.4f" % (eps, m["accuracy"]))

    # Carlini and Wagner L2 attack at several tradeoff values
    print("\n=== Carlini & Wagner L2 ===")
    results["cw"] = {}
    for c in [0.1, 1.0, 10.0]:
        cw = get_cw(model, c=c, steps=50, lr=0.01)
        m = eval_attack_batch(model, test_loader, cw, device, "CW c=%s" % c)
        results["cw"][str(c)] = {"accuracy": m["accuracy"], "f1": m["f1"]}
        print("  CW c=%s: acc=%.4f" % (c, m["accuracy"]))

    # EOT PGD: survives JPEG and resize transforms during attack
    print("\n=== EOT-PGD (survives JPEG/resize) ===")
    results["eot_pgd"] = {}
    for eps in [0.01, 0.02, 0.04, 0.08]:
        # bind current eps into the attack function
        def eot_fn(imgs, labels, e=eps):
            return eot_pgd(model, imgs, labels, eps=e, alpha=e / 4, steps=20, n_transforms=5)
        m = eval_attack_batch(model, test_loader, eot_fn, device, "EOT-PGD eps=%s" % eps)
        results["eot_pgd"][str(eps)] = {"accuracy": m["accuracy"], "f1": m["f1"]}
        print("  EOT-PGD eps=%s: acc=%.4f" % (eps, m["accuracy"]))

    # save results
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/advanced_attack_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/tables/advanced_attack_results.json")

    return


if __name__ == "__main__":
    main(sys.argv)
