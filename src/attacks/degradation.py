"""Image degradation attacks — JPEG compression, noise, blur, downscale/upscale."""
import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def jpeg_compress(img_tensor, quality=50):
    """Apply JPEG compression to a tensor image. Returns tensor."""
    # img_tensor: (C, H, W), normalized
    # undo normalization first
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1)

    # convert to PIL, compress, convert back
    pil_img = transforms.ToPILImage()(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")

    # re-normalize
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(compressed)


def add_gaussian_noise(img_tensor, sigma=0.05):
    """Add Gaussian noise to tensor image."""
    noise = torch.randn_like(img_tensor) * sigma
    return img_tensor + noise


def gaussian_blur(img_tensor, kernel_size=5):
    """Apply Gaussian blur."""
    # need batch dim for torch blur
    blurred = transforms.GaussianBlur(kernel_size=kernel_size, sigma=(1.0, 1.0))(img_tensor)
    return blurred


def downscale_upscale(img_tensor, factor=4):
    """Downscale then upscale back — simulates low quality resize."""
    _, h, w = img_tensor.shape
    small_h, small_w = h // factor, w // factor
    down = torch.nn.functional.interpolate(
        img_tensor.unsqueeze(0), size=(small_h, small_w), mode="bilinear", align_corners=False
    )
    up = torch.nn.functional.interpolate(
        down, size=(h, w), mode="bilinear", align_corners=False
    )
    return up.squeeze(0)


def apply_degradation(img_tensor, attack_type, intensity):
    """Apply a degradation attack given type and intensity."""
    if attack_type == "jpeg":
        return jpeg_compress(img_tensor, quality=int(intensity))
    elif attack_type == "gaussian_noise":
        return add_gaussian_noise(img_tensor, sigma=intensity)
    elif attack_type == "gaussian_blur":
        return gaussian_blur(img_tensor, kernel_size=int(intensity))
    elif attack_type == "downscale":
        return downscale_upscale(img_tensor, factor=int(intensity))
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
