"""
数据加载与预处理模块
支持分类任务（ResNet50）和分割任务（U-Net）的数据加载
合成数据生成：模拟真实无人机农田纹理
作者：202321156060 刘敏
"""

import os
import random
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────
# 合成数据生成（更真实的农田纹理）
# ─────────────────────────────────────────────

def _make_farmland_background(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """生成真实感农田背景：垄行纹理 + 光照渐变"""
    img = np.zeros((H, W, 3), dtype=np.float32)

    # 基础绿色调（模拟健康作物）
    base_g = rng.integers(90, 140)
    base_r = rng.integers(30, 60)
    base_b = rng.integers(20, 50)

    img[:, :, 0] = base_r
    img[:, :, 1] = base_g
    img[:, :, 2] = base_b

    # 垄行纹理（水平条纹，模拟作物行）
    row_spacing = rng.integers(8, 18)
    row_width   = rng.integers(3, 7)
    for y in range(0, H, row_spacing):
        brightness = rng.uniform(0.85, 1.15)
        y_end = min(y + row_width, H)
        img[y:y_end, :, 1] = np.clip(img[y:y_end, :, 1] * brightness, 0, 255)

    # 随机噪声（模拟叶片细节）
    noise = rng.normal(0, 8, (H, W, 3))
    img = np.clip(img + noise, 0, 255)

    # 光照渐变（模拟无人机拍摄角度）
    gradient = np.linspace(0.88, 1.12, W, dtype=np.float32)
    img = np.clip(img * gradient[np.newaxis, :, np.newaxis], 0, 255)

    return img.astype(np.uint8)


def _make_lodging_region(
    img: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    num_regions: int,
):
    """在图像上叠加倒伏区域（黄褐色 + 纹理紊乱）"""
    H, W = img.shape[:2]
    for _ in range(num_regions):
        # 随机椭圆形倒伏区域
        cx = rng.integers(W // 5, 4 * W // 5)
        cy = rng.integers(H // 5, 4 * H // 5)
        rx = rng.integers(W // 8, W // 3)
        ry = rng.integers(H // 8, H // 3)
        angle = rng.uniform(0, np.pi)

        Y, X = np.ogrid[:H, :W]
        # 旋转椭圆
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xr = (X - cx) * cos_a + (Y - cy) * sin_a
        yr = -(X - cx) * sin_a + (Y - cy) * cos_a
        ellipse = (xr ** 2 / max(rx, 1) ** 2 + yr ** 2 / max(ry, 1) ** 2) <= 1

        if ellipse.sum() == 0:
            continue

        n_pix = ellipse.sum()
        # 倒伏颜色：黄褐色（茎秆倒伏后的颜色）
        img[ellipse, 0] = np.clip(rng.integers(150, 200, n_pix) + rng.normal(0, 10, n_pix), 100, 230)
        img[ellipse, 1] = np.clip(rng.integers(110, 155, n_pix) + rng.normal(0, 10, n_pix), 80, 180)
        img[ellipse, 2] = np.clip(rng.integers(20,  60,  n_pix) + rng.normal(0, 8,  n_pix), 0,  100)
        mask[ellipse] = 255

    return img, mask


def generate_demo_data(
    save_dir: str = "dataset",
    num_samples: int = 200,
    image_size: Tuple[int, int] = (256, 256),
    lodging_ratio: float = 0.6,   # 60% 样本含倒伏
    seed: int = 42,
):
    """
    生成高质量合成演示数据集

    Args:
        save_dir:      保存根目录
        num_samples:   总样本数
        image_size:    图像尺寸
        lodging_ratio: 含倒伏样本的比例
        seed:          随机种子
    """
    rng = np.random.default_rng(seed)
    H, W = image_size

    img_dir  = Path(save_dir) / "images"
    mask_dir = Path(save_dir) / "masks"
    # 分类目录
    cls_lodge_dir  = Path(save_dir) / "cls" / "lodging"
    cls_normal_dir = Path(save_dir) / "cls" / "normal"
    for d in [img_dir, mask_dir, cls_lodge_dir, cls_normal_dir]:
        d.mkdir(parents=True, exist_ok=True)

    n_lodging = int(num_samples * lodging_ratio)
    n_normal  = num_samples - n_lodging

    for i in range(num_samples):
        has_lodging = i < n_lodging
        img  = _make_farmland_background(H, W, rng)
        mask = np.zeros((H, W), dtype=np.uint8)

        if has_lodging:
            num_regions = rng.integers(1, 4)
            img, mask = _make_lodging_region(img, mask, rng, num_regions)

        # 随机轻微模糊（模拟不同飞行高度）
        pil_img = Image.fromarray(img)
        if rng.random() < 0.3:
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))

        fname = f"sample_{i:04d}"
        pil_img.save(img_dir / f"{fname}.jpg", quality=92)
        Image.fromarray(mask).save(mask_dir / f"{fname}.png")

        # 同时保存分类目录（供 ResNet 分类器使用）
        cls_dir = cls_lodge_dir if has_lodging else cls_normal_dir
        pil_img.save(cls_dir / f"{fname}.jpg", quality=92)

    print(f"[INFO] 已生成 {num_samples} 张合成图像（倒伏:{n_lodging} 正常:{n_normal}）→ {save_dir}/")


# ─────────────────────────────────────────────
# 分割数据集（U-Net 用）
# ─────────────────────────────────────────────

class JointTransform:
    """图像与掩膜同步数据增强"""

    def __init__(self, image_size=(256, 256), augment=True):
        self.image_size = image_size
        self.augment    = augment
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        self.normalize    = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, self.image_size, interpolation=Image.BILINEAR)
        mask  = TF.resize(mask,  self.image_size, interpolation=Image.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
            mask  = TF.rotate(mask,  angle, interpolation=Image.NEAREST)
            image = self.color_jitter(image)

        img_t  = TF.to_tensor(image)
        mask_t = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)
        if mask_t.max() > 1.0:
            mask_t = mask_t / 255.0
        img_t = self.normalize(img_t)
        return img_t, mask_t


class SegmentationDataset(Dataset):
    """分割数据集：images/ + masks/"""

    EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.transform = transform or JointTransform(augment=False)
        self.samples   = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in self.EXTS
        )
        if not self.samples:
            raise FileNotFoundError(f"未在 {image_dir} 找到图像文件")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path  = self.samples[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L") if mask_path.exists() \
                else Image.fromarray(np.zeros(image.size[::-1], dtype=np.uint8))
        return self.transform(image, mask)


# ─────────────────────────────────────────────
# 分类数据集（ResNet50 用）
# ─────────────────────────────────────────────

class ClassificationDataset(Dataset):
    """
    分类数据集，目录结构：
        cls/
        ├── lodging/   *.jpg  (label=1)
        └── normal/    *.jpg  (label=0)
    """

    EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(self, cls_dir, transform=None, augment=True):
        self.transform = transform
        self.augment   = augment
        self.samples: List[Tuple[Path, int]] = []

        cls_root = Path(cls_dir)
        label_map = {"lodging": 1, "normal": 0}
        for cls_name, label in label_map.items():
            folder = cls_root / cls_name
            if folder.exists():
                for p in sorted(folder.iterdir()):
                    if p.suffix.lower() in self.EXTS:
                        self.samples.append((p, label))

        if not self.samples:
            raise FileNotFoundError(f"未在 {cls_dir} 找到分类图像")

        n_lodge  = sum(1 for _, l in self.samples if l == 1)
        n_normal = sum(1 for _, l in self.samples if l == 0)
        print(f"[INFO] 分类数据集：倒伏={n_lodge}  正常={n_normal}  共={len(self.samples)}")

        # 默认变换
        if self.transform is None:
            if augment:
                self.transform = T.Compose([
                    T.Resize((224, 224)),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(30),
                    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────
# DataLoader 工厂
# ─────────────────────────────────────────────

def build_seg_dataloaders(image_dir, mask_dir, image_size=(256,256),
                          batch_size=8, val_ratio=0.2, num_workers=0, seed=42):
    """构建分割任务 DataLoader"""
    train_tf = JointTransform(image_size=image_size, augment=True)
    val_tf   = JointTransform(image_size=image_size, augment=False)

    full_ds = SegmentationDataset(image_dir, mask_dir, transform=None)
    total   = len(full_ds)
    val_sz  = max(1, int(total * val_ratio))
    trn_sz  = total - val_sz

    g = torch.Generator().manual_seed(seed)
    trn_sub, val_sub = random_split(full_ds, [trn_sz, val_sz], generator=g)

    train_ds = _SubsetWithTransform(trn_sub, train_tf, mode="seg")
    val_ds   = _SubsetWithTransform(val_sub, val_tf,   mode="seg")

    print(f"[INFO] 分割数据集：训练={trn_sz}  验证={val_sz}")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


def build_cls_dataloaders(cls_dir, batch_size=16, val_ratio=0.2, num_workers=0, seed=42):
    """构建分类任务 DataLoader"""
    full_ds = ClassificationDataset(cls_dir, transform=None, augment=False)
    total   = len(full_ds)
    val_sz  = max(1, int(total * val_ratio))
    trn_sz  = total - val_sz

    g = torch.Generator().manual_seed(seed)
    trn_sub, val_sub = random_split(full_ds, [trn_sz, val_sz], generator=g)

    train_ds = _SubsetWithTransform(trn_sub, None, mode="cls", augment=True)
    val_ds   = _SubsetWithTransform(val_sub, None, mode="cls", augment=False)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


class _SubsetWithTransform(Dataset):
    def __init__(self, subset, transform, mode="seg", augment=True):
        self.subset    = subset
        self.transform = transform
        self.mode      = mode
        self.augment   = augment

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        real_idx = self.subset.indices[idx]
        ds = self.subset.dataset

        if self.mode == "seg":
            img_path  = ds.samples[real_idx]
            mask_path = ds.mask_dir / (img_path.stem + ".png")
            image = Image.open(img_path).convert("RGB")
            mask  = Image.open(mask_path).convert("L") if mask_path.exists() \
                    else Image.fromarray(np.zeros(image.size[::-1], dtype=np.uint8))
            return self.transform(image, mask)

        else:  # cls
            path, label = ds.samples[real_idx]
            image = Image.open(path).convert("RGB")
            tf = T.Compose([
                T.Resize((224, 224)),
                *([ T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(30),
                    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                ] if self.augment else []),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            return tf(image), torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    generate_demo_data("dataset", num_samples=100)
