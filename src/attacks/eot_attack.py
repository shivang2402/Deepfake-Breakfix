"""EOT (Expectation over Transformation) attack.
Generates adversarial perturbations that survive JPEG compression and resizing.
"""
import io
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def random_jpeg(img_tensor, quality_range=(50, 95)):
    """Apply random JPEG compression."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)

    # undo normalize
    img = img_tensor * std + mean
    img = img.clamp(0, 1)

    quality = torch.randint(quality_range[0], quality_range[1], (1,)).item()
    pil = transforms.ToPILImage()(img.cpu())
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(compressed).to(img_tensor.device)


def random_resize(img_tensor, scale_range=(0.5, 1.0)):
    """Random downscale then upscale back."""
    _, h, w = img_tensor.shape
    scale = torch.empty(1).uniform_(scale_range[0], scale_range[1]).item()
    new_h, new_w = int(h * scale), int(w * scale)
    down = torch.nn.functional.interpolate(
        img_tensor.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    up = torch.nn.functional.interpolate(
        down, size=(h, w), mode="bilinear", align_corners=False
    )
    return up.squeeze(0)


def eot_pgd(model, images, labels, eps=0.02, alpha=0.005, steps=20, n_transforms=5):
    """PGD attack with expectation over transformations.
    At each step, apply random transforms and average the gradient.
    """
    model.eval()
    images = images.clone().detach()
    labels = labels.clone().detach()

    adv = images.clone().detach()
    adv.requires_grad = True

    for step in range(steps):
        total_grad = torch.zeros_like(images)

        for _ in range(n_transforms):
            # apply random transform to each image in batch
            transformed = []
            for i in range(adv.shape[0]):
                t = adv[i]
                # randomly pick jpeg or resize or both
                r = torch.rand(1).item()
                if r < 0.33:
                    t = random_jpeg(t)
                elif r < 0.66:
                    t = random_resize(t)
                else:
                    t = random_jpeg(random_resize(t))
                transformed.append(t)
            transformed = torch.stack(transformed).requires_grad_(True)

            outputs = model(transformed)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()

            total_grad += transformed.grad.detach()

        # average gradient over transforms
        avg_grad = total_grad / n_transforms

        # PGD step
        adv = adv.detach() + alpha * avg_grad.sign()
        delta = torch.clamp(adv - images, min=-eps, max=eps)
        adv = torch.clamp(images + delta, min=-3, max=3).detach()  # rough clamp for normalized imgs
        adv.requires_grad = True

    return adv.detach()
