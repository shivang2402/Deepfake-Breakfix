# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Black box transferability: attack one architecture, test on the other

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


# pick the best available device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# generate adversarial examples on source, evaluate on both source and target
def attack_and_transfer(source_model, target_model, loader, attack, device, desc=""):
    source_model.eval()
    target_model.eval()

    preds_source, preds_target, all_labels = [], [], []
    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        adv_imgs = attack(imgs, labels)
        with torch.no_grad():
            source_out = source_model(adv_imgs)
            target_out = target_model(adv_imgs)
        preds_source.extend(source_out.argmax(1).cpu().tolist())
        preds_target.extend(target_out.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return compute_metrics(all_labels, preds_source), compute_metrics(all_labels, preds_target)


# load a checkpoint into a given architecture
def load_model(arch, ckpt_name, device):
    model = get_model(arch=arch, num_classes=2)
    model.load_state_dict(torch.load("checkpoints/" + ckpt_name, map_location=device, weights_only=True))
    model.to(device)
    return model


# run transfer sweep across epsilon values for one direction
def run_sweep(src_name, tgt_name, src_model, tgt_model, attack_fn, test_loader, eps_values, device, tag):
    results = {}
    for eps in eps_values:
        attack = attack_fn(src_model, eps)
        src_m, tgt_m = attack_and_transfer(
            src_model, tgt_model, test_loader, attack, device,
            "%s eps=%s %s->%s" % (tag, eps, src_name, tgt_name)
        )
        results[str(eps)] = {"source_acc": src_m["accuracy"], "target_acc": tgt_m["accuracy"]}
        print("  %s eps=%s: %s=%.4f -> %s=%.4f" % (
            tag, eps, src_name, src_m["accuracy"], tgt_name, tgt_m["accuracy"]
        ))
    return results


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

    # load both baseline models
    resnet = load_model("resnet18", "best_resnet18.pth", device)
    effnet = load_model("efficientnet_b0", "best_efficientnet_b0.pth", device)

    eps_values = [0.001, 0.003, 0.005, 0.01, 0.02, 0.04, 0.08]
    results = {}

    # PGD in both directions
    def pgd_attack(model, eps):
        return torchattacks.PGD(model, eps=eps, alpha=eps / 4, steps=10)

    print("\n=== Attack ResNet-18, transfer to EfficientNet-B0 (PGD) ===")
    results["resnet_to_effnet"] = run_sweep(
        "resnet", "effnet", resnet, effnet, pgd_attack, test_loader, eps_values, device, "PGD"
    )

    print("\n=== Attack EfficientNet-B0, transfer to ResNet-18 (PGD) ===")
    results["effnet_to_resnet"] = run_sweep(
        "effnet", "resnet", effnet, resnet, pgd_attack, test_loader, eps_values, device, "PGD"
    )

    # FGSM in both directions
    def fgsm_attack(model, eps):
        return torchattacks.FGSM(model, eps=eps)

    print("\n=== FGSM Transferability ===")
    results["fgsm_resnet_to_effnet"] = run_sweep(
        "resnet", "effnet", resnet, effnet, fgsm_attack, test_loader, eps_values, device, "FGSM"
    )
    results["fgsm_effnet_to_resnet"] = run_sweep(
        "effnet", "resnet", effnet, resnet, fgsm_attack, test_loader, eps_values, device, "FGSM"
    )

    # save results
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/transferability_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/tables/transferability_results.json")

    return


if __name__ == "__main__":
    main(sys.argv)
