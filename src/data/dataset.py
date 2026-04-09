import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DeepfakeDataset(Dataset):
    """Dataset for real vs fake face classification."""

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = ["fake", "real"]
        self.class_to_idx = {"fake": 0, "real": 1}

        # go through each class folder and collect image paths
        self.samples = []
        for cls in self.classes:
            folder = os.path.join(self.root_dir, cls)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Missing folder: {folder}")
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(folder, fname)
                    label = self.class_to_idx[cls]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_counts(self):
        counts = {"fake": 0, "real": 0}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts


def get_transforms(image_size=224, split="train"):
    """Get transforms for given split. Augmentation only for train."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
