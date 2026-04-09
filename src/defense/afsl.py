"""Adversarial Feature Similarity Learning (AFSL).
Based on Goswami et al. (arXiv:2403.08806).
Train model so clean and adversarial versions produce similar features.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchattacks


class FeatureExtractor(nn.Module):
    """Wraps a model to also return intermediate features."""

    def __init__(self, model, arch="resnet18"):
        super().__init__()
        self.model = model
        self.arch = arch
        self._features = None

        # hook into the layer before the classifier
        if arch == "resnet18":
            self.model.avgpool.register_forward_hook(self._hook)
        elif arch == "efficientnet_b0":
            self.model.avgpool.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._features = output.flatten(1)

    def forward(self, x):
        logits = self.model(x)
        return logits, self._features


def feature_similarity_loss(clean_features, adv_features):
    """Cosine similarity loss between clean and adversarial features."""
    cos_sim = F.cosine_similarity(clean_features, adv_features, dim=1)
    return (1 - cos_sim).mean()


def afsl_train_one_epoch(feat_model, loader, criterion, optimizer, device, eps=0.02, lambda_fs=1.0):
    feat_model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_fs_loss = 0.0
    correct = 0
    total = 0

    attack = torchattacks.PGD(feat_model.model, eps=eps, alpha=eps/4, steps=10)

    for imgs, labels in tqdm(loader, desc="AFSL Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        # generate adversarial examples
        feat_model.eval()
        adv_imgs = attack(imgs, labels)
        feat_model.train()

        # forward pass on clean
        clean_logits, clean_feats = feat_model(imgs)
        cls_loss = criterion(clean_logits, labels)

        # forward pass on adversarial
        adv_logits, adv_feats = feat_model(adv_imgs)
        adv_cls_loss = criterion(adv_logits, labels)

        # feature similarity loss
        fs_loss = feature_similarity_loss(clean_feats, adv_feats)

        # total loss
        loss = cls_loss + adv_cls_loss + lambda_fs * fs_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_cls_loss += (cls_loss.item() + adv_cls_loss.item()) * imgs.size(0)
        running_fs_loss += fs_loss.item() * imgs.size(0)

        preds = clean_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    n = total
    return running_loss / n, running_cls_loss / n, running_fs_loss / n, correct / n


def train_afsl(model, arch, train_loader, val_loader, config, device, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    feat_model = FeatureExtractor(model, arch=arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        feat_model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    epochs = config["adversarial_training"]["epochs"]
    eps = config["adversarial_training"]["epsilon"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    patience = config["training"]["early_stopping_patience"]
    no_improve = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        total_loss, cls_loss, fs_loss, train_acc = afsl_train_one_epoch(
            feat_model, train_loader, criterion, optimizer, device, eps=eps
        )
        print(f"  Train Loss: {total_loss:.4f} (cls: {cls_loss:.4f}, fs: {fs_loss:.4f}) | Acc: {train_acc:.4f}")

        # validate on clean data
        feat_model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _ = feat_model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        print(f"  Val Acc: {val_acc:.4f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            path = os.path.join(save_dir, f"best_{arch}_afsl.pth")
            torch.save(model.state_dict(), path)
            print(f"  Saved best AFSL model (val_acc={val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping")
                break

    return model, best_val_acc
