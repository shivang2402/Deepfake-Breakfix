"""Run advanced attacks: more epsilons, C&W, EOT."""
import json
import os
import torch
import torchattacks
from tqdm import tqdm
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.models.classifier import get_model
from src.utils.metrics import compute_metrics
from src.attacks.eot_attack import eot_pgd
from src.attacks.cw_attack import get_cw


def eval_attack_batch(model, loader, attack_fn, device, desc=""):
    """Generic eval: attack_fn takes (images, labels) and returns adv_images."""
    model.eval()
    all_preds = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        adv_imgs = attack_fn(imgs, labels)
        with torch.no_grad():
            preds = model(adv_imgs).argmax(1).cpu().tolist()
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

    arch = config["model"]["architecture"]
    model = get_model(arch=arch, num_classes=2)
    model.load_state_dict(torch.load(f"checkpoints/best_{arch}.pth", map_location=device, weights_only=True))
    model.to(device)

    results = {}

    # --- finer epsilon sweep ---
    print("\n=== Fine-grained FGSM ===")
    fine_eps = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08]
    results["fgsm_fine"] = {}
    for eps in fine_eps:
        attack = torchattacks.FGSM(model, eps=eps)
        m = eval_attack_batch(model, test_loader, attack, device, f"FGSM eps={eps}")
        results["fgsm_fine"][str(eps)] = {"accuracy": m["accuracy"], "f1": m["f1"]}
        print(f"  FGSM eps={eps}: acc={m['accuracy']:.4f}")

    print("\n=== Fine-grained PGD ===")
    results["pgd_fine"] = {}
    for eps in fine_eps:
        attack = torchattacks.PGD(model, eps=eps, alpha=eps/4, steps=10)
        m = eval_attack_batch(model, test_loader, attack, device, f"PGD eps={eps}")
        results["pgd_fine"][str(eps)] = {"accuracy": m["accuracy"], "f1": m["f1"]}
        print(f"  PGD eps={eps}: acc={m['accuracy']:.4f}")

    # --- C&W attack ---
    print("\n=== Carlini & Wagner L2 ===")
    results["cw"] = {}
    for c in [0.1, 1.0, 10.0]:
        cw = get_cw(model, c=c, steps=50, lr=0.01)
        m = eval_attack_batch(model, test_loader, cw, device, f"CW c={c}")
        results["cw"][str(c)] = {"accuracy": m["accuracy"], "f1": m["f1"]}
        print(f"  CW c={c}: acc={m['accuracy']:.4f}")

    # --- EOT attack ---
    print("\n=== EOT-PGD (survives JPEG/resize) ===")
    results["eot_pgd"] = {}
    for eps in [0.01, 0.02, 0.04, 0.08]:
        def eot_fn(imgs, labels, e=eps):
            return eot_pgd(model, imgs, labels, eps=e, alpha=e/4, steps=20, n_transforms=5)
        m = eval_attack_batch(model, test_loader, eot_fn, device, f"EOT-PGD eps={eps}")
        results["eot_pgd"][str(eps)] = {"accuracy": m["accuracy"], "f1": m["f1"]}
        print(f"  EOT-PGD eps={eps}: acc={m['accuracy']:.4f}")

    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/advanced_attack_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/tables/advanced_attack_results.json")


if __name__ == "__main__":
    main()
