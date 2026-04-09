"""Run attacks on test set and collect metrics at each intensity."""
import torch
from tqdm import tqdm
from src.utils.metrics import compute_metrics
from src.attacks.degradation import apply_degradation


def eval_degradation(model, loader, attack_type, intensities, device):
    """Evaluate model under degradation attack at various intensities.
    Returns dict of {intensity: metrics}.
    """
    results = {}
    model.eval()

    for intensity in intensities:
        all_preds = []
        all_labels = []

        for imgs, labels in tqdm(loader, desc=f"{attack_type} @ {intensity}", leave=False):
            # apply degradation to each image in the batch
            degraded = []
            for img in imgs:
                d = apply_degradation(img, attack_type, intensity)
                degraded.append(d)
            degraded = torch.stack(degraded).to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(degraded)
                preds = outputs.argmax(dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        results[intensity] = compute_metrics(all_labels, all_preds)
        print(f"  {attack_type} @ {intensity}: acc={results[intensity]['accuracy']:.4f}")

    return results


def eval_adversarial(model, loader, attack, eps_values, device, attack_name="fgsm"):
    """Evaluate model under adversarial attack at various epsilon values.
    Returns dict of {eps: metrics}.
    """
    results = {}
    model.eval()

    for eps in eps_values:
        # update epsilon
        attack.eps = eps
        if hasattr(attack, 'alpha'):
            attack.alpha = eps / 4  # standard ratio

        all_preds = []
        all_labels = []

        for imgs, labels in tqdm(loader, desc=f"{attack_name} eps={eps}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            # generate adversarial examples
            adv_imgs = attack(imgs, labels)

            with torch.no_grad():
                outputs = model(adv_imgs)
                preds = outputs.argmax(dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        results[eps] = compute_metrics(all_labels, all_preds)
        print(f"  {attack_name} eps={eps}: acc={results[eps]['accuracy']:.4f}")

    return results
