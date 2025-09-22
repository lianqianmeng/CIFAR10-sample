# train_cifar10.py
# Python 3.12 + PyTorch 2.8.0
import argparse
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

# ----------------------------
# Model: 简洁 CNN
# ----------------------------
class SmallCIFARConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 16x16
            nn.Dropout(0.2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 8x8
            nn.Dropout(0.3),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 4x4
            nn.Dropout(0.4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)
        return x

# ----------------------------
# Train / Eval loops
# ----------------------------
def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = targets.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy_from_logits(logits, targets) * bs
        n += bs
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        logits = model(imgs)
        loss = F.cross_entropy(logits, targets)
        bs = targets.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy_from_logits(logits, targets) * bs
        n += bs
    return running_loss / n, running_acc / n

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN Trainer (PyTorch 2.8)")
    parser.add_argument("--data_dir", type=str, default="./Train_CIFAR10/data", help="CIFAR-10 数据目录")
    parser.add_argument("--epochs", type=int, default=30, help="最大训练 epoch 数")
    parser.add_argument("--target_acc", type=float, default=95, help="目标 Top-1 准确率（在测试集上），达到则提前停止")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # CIFAR-10 均值与方差
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2470, 0.2435, 0.2616)

    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    # Datasets / Loaders
    data_dir = Path(args.data_dir)
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), persistent_workers=args.num_workers > 0
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), persistent_workers=args.num_workers > 0
    )

    # Model / Optim
    model = SmallCIFARConvNet(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 简洁：可选的余弦退火调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 记录
    hist_train_loss, hist_train_acc = [], []
    hist_val_loss,   hist_val_acc   = [], []

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, device)
        scheduler.step()

        end_time = time.time()

        epoch_time = end_time - start_time

        hist_train_loss.append(train_loss)
        hist_train_acc.append(train_acc)
        hist_val_loss.append(val_loss)
        hist_val_acc.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"Epoch [{epoch:03d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
              f"Best: {best_acc*100:.2f}% || "
              f"Time_cost: {epoch_time:.2f}Second")

        # 达到目标精度则提前停止
        if val_acc >= args.target_acc:
            print(f"Target accuracy {args.target_acc*100:.2f}% reached at epoch {epoch}. Early stopping.")
            break

    # 恢复最佳模型并保存
    if best_state is not None:
        model.load_state_dict(best_state)
    out_dir = Path("./Train_CIFAR10/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    time_f = time.strftime("%Y-%m-%d-%Hh%Mm%Ss", time.localtime())
    pt_path = time_f + 'cifar10_cnn_best.pt'
    torch.save(model.state_dict(), out_dir / pt_path)
    print(f"Best model saved to {out_dir / pt_path} with Val Acc={best_acc*100:.2f}%")

    # ----------------------------
    # 绘图：Loss 与 Accuracy
    # ----------------------------
    epochs_ran = range(1, len(hist_train_loss) + 1)

    # Loss
    loss_fig_path = time_f +'-loss_curve.png'
    plt.figure()
    plt.plot(list(epochs_ran), hist_train_loss, label="Train Loss")
    plt.plot(list(epochs_ran), hist_val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CIFAR-10 Loss vs Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / loss_fig_path, dpi=150)
    plt.close()

    # Accuracy
    accuracy_fig_path = time_f +'-accuracy_curve.png'
    plt.figure()
    plt.plot(list(epochs_ran), [a * 100 for a in hist_train_acc], label="Train Acc (%)")
    plt.plot(list(epochs_ran), [a * 100 for a in hist_val_acc], label="Val Acc (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("CIFAR-10 Accuracy vs Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / accuracy_fig_path, dpi=150)
    plt.close()

    print(f"Saved plots to {out_dir / loss_fig_path} and {out_dir / accuracy_fig_path}")

if __name__ == "__main__":
    main()
