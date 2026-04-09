"""Adversarial training — mix clean and adversarial examples during training."""
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torchattacks


def adv_train_one_epoch(model, loader, criterion, optimizer, device, eps=0.02, mix_ratio=0.5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # PGD attack for generating adversarial examples during training
    attack = torchattacks.PGD(model, eps=eps, alpha=eps/4, steps=10)

    for imgs, labels in tqdm(loader, desc="Adv Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size = imgs.size(0)

        # split batch into clean and adversarial
        n_adv = int(batch_size * mix_ratio)
        if n_adv > 0:
            model.eval()  # attack needs eval mode
            adv_imgs = attack(imgs[:n_adv], labels[:n_adv])
            model.train()
            # combine adversarial + clean
            mixed_imgs = torch.cat([adv_imgs, imgs[n_adv:]], dim=0)
        else:
            mixed_imgs = imgs

        optimizer.zero_grad()
        outputs = model(mixed_imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def train_robust_model(model, train_loader, val_loader, config, device, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["adversarial_training"]["epochs"]
    )

    eps = config["adversarial_training"]["epsilon"]
    mix_ratio = config["adversarial_training"]["mix_ratio"]
    epochs = config["adversarial_training"]["epochs"]
    patience = config["training"]["early_stopping_patience"]

    best_val_acc = 0.0
    no_improve = 0
    model.to(device)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = adv_train_one_epoch(
            model, train_loader, criterion, optimizer, device, eps=eps, mix_ratio=mix_ratio
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        arch = config["model"]["architecture"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            path = os.path.join(save_dir, f"best_{arch}_robust.pth")
            torch.save(model.state_dict(), path)
            print(f"  Saved best robust model (val_acc={val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping after {patience} epochs without improvement")
                break

    return model, best_val_acc
