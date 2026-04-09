"""Grad-CAM heatmaps to see where the model is looking."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms


def unnormalize(tensor):
    """Undo ImageNet normalization, return numpy (H, W, C) in [0, 1]."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def get_gradcam(model, target_layer):
    """Create a GradCAM object for the given model and layer."""
    return GradCAM(model=model, target_layers=[target_layer])


def generate_heatmap(cam, img_tensor, label):
    """Generate a single Grad-CAM heatmap overlay."""
    # cam expects batch input
    input_tensor = img_tensor.unsqueeze(0)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0]

    rgb_img = unnormalize(img_tensor)
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return overlay, rgb_img, grayscale_cam


def save_gradcam_grid(model, target_layer, samples, save_path, title="Grad-CAM"):
    """Generate and save a grid of Grad-CAM heatmaps.

    samples: list of (img_tensor, label, description) tuples
    """
    cam = get_gradcam(model, target_layer)
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        axes = [axes]

    for i, (img_tensor, label, desc) in enumerate(samples):
        overlay, rgb_img, _ = generate_heatmap(cam, img_tensor, label)

        axes[i][0].imshow(rgb_img)
        axes[i][0].set_title(f"Original ({desc})")
        axes[i][0].axis("off")

        axes[i][1].imshow(overlay)
        axes[i][1].set_title("Grad-CAM")
        axes[i][1].axis("off")

        # show just the heatmap
        axes[i][2].imshow(generate_heatmap(cam, img_tensor, label)[2], cmap="jet")
        axes[i][2].set_title("Heatmap")
        axes[i][2].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
