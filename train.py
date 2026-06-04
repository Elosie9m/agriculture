"""
模型训练脚本
包含：
  1. ResNet50 分类器训练（判断有无倒伏）
  2. U-Net 分割模型训练（像素级倒伏区域）
作者：202321156060 刘敏

使用方法：
    python train.py                        # 训练全部模型
    python train.py --task cls             # 只训练分类器
    python train.py --task seg             # 只训练分割模型
    python train.py --epochs 50 --batch_size 16
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.unet_model import UNet
from models.classifier import LodgingClassifier
from utils.data_loader import (
    generate_demo_data,
    build_seg_dataloaders,
    build_cls_dataloaders,
)


# ─────────────────────────────────────────────
# 损失函数
# ─────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).view(-1)
        t = targets.view(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class CombinedSegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)


# ─────────────────────────────────────────────
# 评估指标
# ─────────────────────────────────────────────

def iou_score(logits, targets, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * targets).sum().item()
    union = (pred + targets).clamp(0, 1).sum().item()
    return inter / (union + 1e-8)


def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ─────────────────────────────────────────────
# 分类器训练
# ─────────────────────────────────────────────

def train_classifier(args, device):
    print("\n" + "=" * 55)
    print("  [1/2] 训练 ResNet50 分类器")
    print("=" * 55)

    cls_dir = os.path.join(args.data_dir, "cls")
    train_loader, val_loader = build_cls_dataloaders(
        cls_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
    )

    model = LodgingClassifier(num_classes=2, pretrained=True, freeze_backbone=False).to(device)

    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path = weights_dir / "classifier_best.pth"

    # 断点续训
    start_epoch = 1
    best_acc = 0.0
    if args.cls_resume and os.path.exists(args.cls_resume):
        state = torch.load(args.cls_resume, map_location=device)
        model.load_state_dict(state)
        print(f"  [续训] 已加载分类器权重: {args.cls_resume}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    patience_cnt = 0

    for epoch in range(1, args.epochs + 1):
        # 训练
        model.train()
        t0 = time.time()
        trn_loss, trn_acc = 0.0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            trn_loss += loss.item()
            trn_acc  += accuracy(logits.detach(), labels)

        trn_loss /= len(train_loader)
        trn_acc  /= len(train_loader)

        # 验证
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item()
                val_acc  += accuracy(logits, labels)
        val_loss /= len(val_loader)
        val_acc  /= len(val_loader)
        scheduler.step()

        print(f"  Cls Epoch [{epoch:3d}/{args.epochs}] "
              f"Train Loss:{trn_loss:.4f} Acc:{trn_acc:.4f} | "
              f"Val Loss:{val_loss:.4f} Acc:{val_acc:.4f} | "
              f"Time:{time.time()-t0:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"    ✓ 保存最优分类器 Val Acc={best_acc:.4f}")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  早停（{args.patience}轮无改善）")
                break

    print(f"\n  分类器训练完成  最优 Val Acc: {best_acc:.4f}")
    print(f"  权重保存至: {best_path}")
    return best_acc


# ─────────────────────────────────────────────
# 分割模型训练
# ─────────────────────────────────────────────

def train_segmentation(args, device):
    print("\n" + "=" * 55)
    print("  [2/2] 训练 U-Net 分割模型")
    print("=" * 55)

    img_dir  = os.path.join(args.data_dir, "images")
    mask_dir = os.path.join(args.data_dir, "masks")
    train_loader, val_loader = build_seg_dataloaders(
        img_dir, mask_dir,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
    )

    model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)

    weights_dir = Path(args.weights_dir)
    best_path   = weights_dir / "unet_best.pth"

    # 断点续训
    best_iou = 0.0
    if args.seg_resume and os.path.exists(args.seg_resume):
        state = torch.load(args.seg_resume, map_location=device)
        model.load_state_dict(state)
        print(f"  [续训] 已加载 U-Net 权重: {args.seg_resume}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CombinedSegLoss()

    patience_cnt = 0

    for epoch in range(1, args.epochs + 1):
        # 训练
        model.train()
        t0 = time.time()
        trn_loss, trn_iou = 0.0, 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            trn_loss += loss.item()
            trn_iou  += iou_score(logits.detach(), masks)

        trn_loss /= len(train_loader)
        trn_iou  /= len(train_loader)

        # 验证
        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
                val_iou  += iou_score(logits, masks)
        val_loss /= len(val_loader)
        val_iou  /= len(val_loader)
        scheduler.step()

        print(f"  Seg Epoch [{epoch:3d}/{args.epochs}] "
              f"Train Loss:{trn_loss:.4f} IoU:{trn_iou:.4f} | "
              f"Val Loss:{val_loss:.4f} IoU:{val_iou:.4f} | "
              f"Time:{time.time()-t0:.1f}s")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), best_path)
            print(f"    ✓ 保存最优 U-Net  Val IoU={best_iou:.4f}")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  早停（{args.patience}轮无改善）")
                break

    print(f"\n  U-Net 训练完成  最优 Val IoU: {best_iou:.4f}")
    print(f"  权重保存至: {best_path}")
    return best_iou


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 设备: {device}")

    # 若数据集为空，自动生成合成数据
    img_dir = Path(args.data_dir) / "images"
    cls_dir = Path(args.data_dir) / "cls"
    if not img_dir.exists() or not any(img_dir.iterdir()):
        print("[INFO] 未找到数据集，自动生成合成演示数据...")
        generate_demo_data(args.data_dir, num_samples=args.demo_samples)

    results = {}
    if args.task in ("all", "cls"):
        results["cls_acc"] = train_classifier(args, device)
    if args.task in ("all", "seg"):
        results["seg_iou"] = train_segmentation(args, device)

    print("\n" + "=" * 55)
    print("  训练完成汇总")
    print("=" * 55)
    if "cls_acc" in results:
        print(f"  分类器  最优 Val Accuracy : {results['cls_acc']:.4f} ({results['cls_acc']*100:.1f}%)")
    if "seg_iou" in results:
        print(f"  U-Net   最优 Val IoU      : {results['seg_iou']:.4f}")
    print(f"  权重目录: {args.weights_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="农作物倒伏检测训练脚本")
    p.add_argument("--task",         type=str,   default="all",    choices=["all","cls","seg"])
    p.add_argument("--data_dir",     type=str,   default="dataset")
    p.add_argument("--weights_dir",  type=str,   default="weights")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--img_size",     type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--val_ratio",    type=float, default=0.2)
    p.add_argument("--patience",     type=int,   default=15)
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--demo_samples", type=int,   default=300)
    p.add_argument("--cls_resume",   type=str,   default=None,  help="分类器断点续训权重路径")
    p.add_argument("--seg_resume",   type=str,   default=None,  help="U-Net断点续训权重路径")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
