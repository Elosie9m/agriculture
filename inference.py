"""
推理脚本 — 三模型串联流水线
流程：图像 → ResNet50分类 → (有倒伏) → U-Net分割 → 灾害评估
作者：202321156060 刘敏

使用方法：
    python inference.py --image path/to/image.jpg
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from models.unet_model import UNet, get_model
from models.classifier import LodgingClassifier, get_classifier
from utils.segmentation_utils import full_assessment, postprocess_mask
from utils.visualize import (
    create_annotated_image,
    create_mask_visualization,
    numpy_to_pil,
    pil_to_numpy,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# 预处理
# ─────────────────────────────────────────────

def preprocess_for_cls(image: Image.Image) -> torch.Tensor:
    """分类器预处理：224×224"""
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(image.convert("RGB")).unsqueeze(0)


def preprocess_for_seg(image: Image.Image, size=(256, 256)) -> torch.Tensor:
    """分割模型预处理：256×256"""
    img = TF.resize(image.convert("RGB"), size, interpolation=Image.BILINEAR)
    t   = TF.to_tensor(img)
    t   = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(t)
    return t.unsqueeze(0)


def postprocess_seg(logits: torch.Tensor, orig_size: Tuple[int,int],
                    threshold=0.5) -> np.ndarray:
    """分割输出 → 原始尺寸二值掩膜"""
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    prob_pil = Image.fromarray((prob * 255).astype(np.uint8))
    prob_pil = prob_pil.resize((orig_size[1], orig_size[0]), Image.BILINEAR)
    binary   = (np.array(prob_pil) / 255.0 > threshold).astype(np.uint8) * 255
    return postprocess_mask(binary)


# ─────────────────────────────────────────────
# 三模型串联推理器
# ─────────────────────────────────────────────

class LodgingPipeline:
    """
    农作物倒伏检测三阶段流水线
      Stage 1: ResNet50 分类（有无倒伏）
      Stage 2: U-Net 分割（倒伏区域像素级定位）
      Stage 3: 灾害评估（面积占比 + 风险等级）
    """

    def __init__(
        self,
        cls_weights:  Optional[str] = None,
        seg_weights:  Optional[str] = None,
        device:       Optional[str] = None,
        cls_threshold: float = 0.5,
        seg_threshold: float = 0.5,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device        = device
        self.cls_threshold = cls_threshold
        self.seg_threshold = seg_threshold

        # 分类器
        self.classifier = get_classifier(cls_weights, device)
        self.classifier.eval()

        # 分割模型
        self.segmentor = get_model(n_classes=1, pretrained_path=seg_weights, device=device)
        self.segmentor.eval()

        if cls_weights is None:
            print("[WARNING] 分类器使用随机权重（演示模式），请先运行 train.py")
        if seg_weights is None:
            print("[WARNING] 分割模型使用随机权重（演示模式），请先运行 train.py")

    def predict(self, image: Image.Image) -> Dict:
        """
        完整推理

        Returns dict:
            has_lodging      - 分类结果（bool）
            cls_confidence   - 倒伏置信度（0~1）
            lodging_ratio    - 倒伏面积占比（%）
            risk_level       - 风险等级（低/中/高）
            risk_color       - 风险颜色
            lodging_pixels   - 倒伏像素数
            total_pixels     - 总像素数
            original_image   - 原始图 numpy (H,W,3)
            annotated_image  - 标注图 numpy (H,W,3)
            mask_vis         - 掩膜可视化图 numpy (H,W,3)
            binary_mask      - 二值掩膜 numpy (H,W)
        """
        image_rgb  = image.convert("RGB")
        orig_np    = pil_to_numpy(image_rgb)
        orig_size  = (orig_np.shape[0], orig_np.shape[1])

        # ── Stage 1: 分类 ──
        cls_input = preprocess_for_cls(image_rgb).to(self.device)
        with torch.no_grad():
            cls_logits = self.classifier(cls_input)
            cls_proba  = torch.softmax(cls_logits, dim=1)
        cls_confidence = cls_proba[0, 1].item()   # 倒伏概率
        has_lodging    = cls_confidence >= self.cls_threshold

        # ── Stage 2: 分割（无论分类结果都做，保证可视化完整） ──
        seg_input = preprocess_for_seg(image_rgb).to(self.device)
        with torch.no_grad():
            seg_logits = self.segmentor(seg_input)
        binary_mask = postprocess_seg(seg_logits, orig_size, self.seg_threshold)

        # 若分类判断无倒伏，清空掩膜（以分类结果为准）
        if not has_lodging:
            binary_mask = np.zeros_like(binary_mask)

        # ── Stage 3: 灾害评估 ──
        assessment = full_assessment(binary_mask)

        # ── 可视化 ──
        annotated = create_annotated_image(
            orig_np, binary_mask,
            ratio=assessment["lodging_ratio"],
            risk_level=assessment["risk_level"],
        )
        mask_vis = create_mask_visualization(binary_mask)

        return {
            "has_lodging":    has_lodging,
            "cls_confidence": round(cls_confidence * 100, 1),
            "binary_mask":    binary_mask,
            "original_image": orig_np,
            "annotated_image": annotated,
            "mask_vis":       mask_vis,
            **assessment,
        }


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────

def run_inference(args):
    if not os.path.exists(args.image):
        print(f"[ERROR] 图像不存在: {args.image}")
        return

    image = Image.open(args.image).convert("RGB")
    print(f"[INFO] 图像: {args.image}  尺寸: {image.size}")

    cls_w = args.cls_weights if os.path.exists(args.cls_weights) else None
    seg_w = args.seg_weights if os.path.exists(args.seg_weights) else None

    pipeline = LodgingPipeline(cls_weights=cls_w, seg_weights=seg_w)
    result   = pipeline.predict(image)

    print("\n" + "=" * 50)
    print("  农作物倒伏检测结果")
    print("=" * 50)
    print(f"  分类结果:     {'存在倒伏 ⚠️' if result['has_lodging'] else '正常 ✅'}")
    print(f"  倒伏置信度:   {result['cls_confidence']:.1f}%")
    print(f"  倒伏面积占比: {result['lodging_ratio']:.2f}%")
    print(f"  风险等级:     {result['risk_level']}")
    print(f"  倒伏像素数:   {result['lodging_pixels']:,}")
    print("=" * 50)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    numpy_to_pil(result["annotated_image"]).save(out_dir / f"{stem}_annotated.png")
    numpy_to_pil(result["mask_vis"]).save(out_dir / f"{stem}_mask.png")
    print(f"[INFO] 结果已保存至 {out_dir}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",       type=str, required=True)
    p.add_argument("--cls_weights", type=str, default="weights/classifier_best.pth")
    p.add_argument("--seg_weights", type=str, default="weights/unet_best.pth")
    p.add_argument("--output_dir",  type=str, default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
