import os, time
import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np

DATA_DIR = Path("datasets/cls")
BATCH    = 128
EPOCHS   = 20
LR       = 1e-3
SEED     = 42
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = max(2, (os.cpu_count() or 4) // 2)
PIN_MEMORY  = (DEVICE == "cuda")

torch.manual_seed(SEED)
np.random.seed(SEED)

def make_loaders():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.9,1.0)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])
    tr = datasets.ImageFolder(DATA_DIR/"train", transform=train_tf)
    va = datasets.ImageFolder(DATA_DIR/"val",   transform=val_tf)

    tr_loader = DataLoader(tr, batch_size=BATCH, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    va_loader = DataLoader(va, batch_size=BATCH, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return tr_loader, va_loader, tr.classes, len(tr), len(va)

class Head(nn.Module):
    def __init__(self, in_feat, ncls):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_feat, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, ncls)
        )
    def forward(self, x): return self.mlp(x)

def build_model(n_classes: int):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    # lepší pro malé vstupy 64x64
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    in_feat = model.fc.in_features
    model.fc = Head(in_feat, n_classes)
    return model

def accuracy(pred, target):
    return (pred.argmax(1) == target).float().mean().item()

def train():
    train_loader, val_loader, classes, n_train, n_val = make_loaders()
    print(f"Device   : {DEVICE}")
    print(f"Classes  : {classes}")
    print(f"Train/Val: {n_train} / {n_val} samples")
    print(f"Batch    : {BATCH}, Epochs: {EPOCHS}, LR: {LR}\n")

    model = build_model(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    best_acc = 0.0
    best_path = "cls_best.pth"

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_acc  = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [train]", leave=False)
        for x, y in pbar:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            seen += bs
            running_loss += loss.item() * bs
            running_acc  += (logits.argmax(1) == y).sum().item()

            avg_loss = running_loss / seen
            avg_acc  = running_acc  / seen
            imgs_per_sec = seen / max(1e-6, (time.time()-t0))
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc":  f"{avg_acc:.3f}",
                "lr":   f"{optimizer.param_groups[0]['lr']:.2e}",
                "ips":  f"{imgs_per_sec:.1f}"
            })

        # VALIDACE
        model.eval()
        val_loss = 0.0
        val_acc  = 0.0
        val_seen = 0
        pbar_v = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [val  ]", leave=False)
        with torch.no_grad():
            for x, y in pbar_v:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                bs = x.size(0)
                val_seen += bs
                val_loss += loss.item() * bs
                val_acc  += (logits.argmax(1) == y).sum().item()
                pbar_v.set_postfix({
                    "loss": f"{val_loss/val_seen:.4f}",
                    "acc":  f"{val_acc/val_seen:.3f}"
                })

        scheduler.step()

        epoch_time = time.time() - t0
        tr_loss = running_loss / max(1, seen)
        tr_acc  = running_acc  / max(1, seen)
        va_loss = val_loss     / max(1, val_seen)
        va_acc  = val_acc      / max(1, val_seen)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
              f"time {epoch_time:.1f}s")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(),
                        "classes": classes}, best_path)
            print(f"  ↳ Saved best to {best_path} (val acc {best_acc:.3f})")

    print(f"\n✔ Done. Best val acc = {best_acc:.3f}  (saved: {best_path})")

if __name__ == "__main__":
    train()
