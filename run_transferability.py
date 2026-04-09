"""Black-box transferability: attack one model, test on the other."""
import json
import os
import torch
import torchattacks
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.models.classifier import get_model
from src.utils.metrics import compute_metrics
from tqdm import tqdm


def attack_and_transfer(source_model, target_model, loader, attack, device, desc=""):
    """Generate adversarial examples on source, evaluate on target."""
    source_model.eval()
    target_model.eval()

    all_preds_source = []
    all_preds_target = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        # generate adversarial examples using source model
        adv_imgs = attack(imgs, labels)

        # evaluate on both models
        with torch.no_grad():
            source_out = source_model(adv_imgs)
            target_out = target_model(adv_imgs)

        all_preds_source.extend(source_out.argmax(1).cpu().tolist())
        all_preds_target.extend(target_out.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    source_metrics = compute_metrics(all_labels, all_preds_source)
    target_metrics = compute_metrics(all_labels, all_preds_target)
    return source_metrics, target_metrics


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

    eps_values = [0.001, 0.003, 0.005, 0.01, 0.02, 0.04, 0.08]
    results = {}

    # attack resnet, test on effnet
    print("\n=== Attack ResNet-18, transfer to EfficientNet-B0 ===")
    results["resnet_to_effnet"] = {}
    for eps in eps_values:
        attack = torchattacks.PGD(resnet, eps=eps, alpha=eps/4, steps=10)
        src, tgt = attack_and_transfer(resnet, effnet, test_loader, attack, device,
                                       f"PGD eps={eps} resnet->effnet")
        results["resnet_to_effnet"][str(eps)] = {
            "source_acc": src["accuracy"],
            "target_acc": tgt["accuracy"],
        }
        print(f"  eps={eps}: resnet={src['accuracy']:.4f}, effnet={tgt['accuracy']:.4f}")

    # attack effnet, test on resnet
    print("\n=== Attack EfficientNet-B0, transfer to ResNet-18 ===")
    results["effnet_to_resnet"] = {}
    for eps in eps_values:
        attack = torchattacks.PGD(effnet, eps=eps, alpha=eps/4, steps=10)
        src, tgt = attack_and_transfer(effnet, resnet, test_loader, attack, device,
                                       f"PGD eps={eps} effnet->resnet")
        results["effnet_to_resnet"][str(eps)] = {
            "source_acc": src["accuracy"],
            "target_acc": tgt["accuracy"],
        }
        print(f"  eps={eps}: effnet={src['accuracy']:.4f}, resnet={tgt['accuracy']:.4f}")

    # FGSM transferability too
    print("\n=== FGSM Transferability ===")
    results["fgsm_resnet_to_effnet"] = {}
    results["fgsm_effnet_to_resnet"] = {}
    for eps in eps_values:
        # resnet -> effnet
        attack = torchattacks.FGSM(resnet, eps=eps)
        src, tgt = attack_and_transfer(resnet, effnet, test_loader, attack, device,
                                       f"FGSM eps={eps} resnet->effnet")
        results["fgsm_resnet_to_effnet"][str(eps)] = {
            "source_acc": src["accuracy"],
            "target_acc": tgt["accuracy"],
        }
        print(f"  FGSM eps={eps}: resnet={src['accuracy']:.4f} -> effnet={tgt['accuracy']:.4f}")

        # effnet -> resnet
        attack = torchattacks.FGSM(effnet, eps=eps)
        src, tgt = attack_and_transfer(effnet, resnet, test_loader, attack, device,
                                       f"FGSM eps={eps} effnet->resnet")
        results["fgsm_effnet_to_resnet"][str(eps)] = {
            "source_acc": src["accuracy"],
            "target_acc": tgt["accuracy"],
        }
        print(f"  FGSM eps={eps}: effnet={src['accuracy']:.4f} -> resnet={tgt['accuracy']:.4f}")

    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/transferability_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/tables/transferability_results.json")


if __name__ == "__main__":
    main()
