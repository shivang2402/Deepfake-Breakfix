from torch.utils.data import DataLoader
from .dataset import DeepfakeDataset, get_transforms


def create_dataloaders(data_dir, image_size=224, batch_size=32, num_workers=4):
    """Make train, valid, test dataloaders."""
    loaders = {}
    for split in ["train", "valid", "test"]:
        tf = get_transforms(image_size=image_size, split=split)
        ds = DeepfakeDataset(data_dir, split=split, transform=tf)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
