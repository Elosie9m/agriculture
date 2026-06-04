"""
可视化工具模块
生成倒伏区域标注图、掩膜叠加图、结果对比图
作者：202321156060 刘敏
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


# ─────────────────────────────────────────────
# 颜色常量
# ─────────────────────────────────────────────
LODGING_COLOR_RGB = (220, 50, 50)      # 倒伏区域标注色（红色）
OVERLAY_ALPHA = 0.45                    # 叠加透明度
CONTOUR_COLOR_BGR = (0, 0, 220)        # 轮廓颜色（BGR，用于 OpenCV）
CONTOUR_THICKNESS = 2


# ─────────────────────────────────────────────
# 核心可视化函数
# ─────────────────────────────────────────────

def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = LODGING_COLOR_RGB,
    alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """
    将分割掩膜以半透明颜色叠加到原始图像上

    Args:
        image: 原始图像 (H, W, 3)，uint8，RGB
        mask:  二值掩膜 (H, W)，值为 0/1 或 0/255
        color: 叠加颜色 (R, G, B)
        alpha: 叠加透明度 [0, 1]

    Returns:
        result: 叠加后的图像 (H, W, 3)，uint8，RGB
    """
    image = image.copy().astype(np.float32)
    binary = (mask > 0)

    overlay = np.zeros_like(image)
    overlay[binary] = color

    result = image.copy()
    result[binary] = (1 - alpha) * image[binary] + alpha * overlay[binary]
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_contours_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (220, 50, 50),
    thickness: int = CONTOUR_THICKNESS,
) -> np.ndarray:
    """
    在图像上绘制倒伏区域轮廓

    Args:
        image:     原始图像 (H, W, 3)，uint8，RGB
        mask:      二值掩膜 (H, W)
        color:     轮廓颜色 (R, G, B)
        thickness: 轮廓线宽

    Returns:
        result: 绘制轮廓后的图像
    """
    result = image.copy()
    binary = (mask > 0).astype(np.uint8) * 255

    # OpenCV 使用 BGR
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bgr_color = (color[2], color[1], color[0])
    cv2.drawContours(result_bgr, contours, -1, bgr_color, thickness)

    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def create_annotated_image(
    image: np.ndarray,
    mask: np.ndarray,
    ratio: float,
    risk_level: str,
    draw_contour: bool = True,
) -> np.ndarray:
    """
    生成完整标注图：半透明叠加 + 轮廓 + 文字信息

    Args:
        image:       原始图像 (H, W, 3)，uint8，RGB
        mask:        二值掩膜 (H, W)
        ratio:       倒伏面积占比（%）
        risk_level:  风险等级字符串
        draw_contour: 是否绘制轮廓

    Returns:
        annotated: 标注后的图像
    """
    # 1. 半透明叠加
    annotated = overlay_mask_on_image(image, mask)

    # 2. 轮廓
    if draw_contour:
        annotated = draw_contours_on_image(annotated, mask)

    # 3. 文字标注（使用 PIL 支持中文）
    pil_img = Image.fromarray(annotated)
    draw = ImageDraw.Draw(pil_img)

    # 风险等级颜色映射
    risk_color_map = {"低": (82, 196, 26), "中": (250, 173, 20), "高": (245, 34, 45)}
    text_color = risk_color_map.get(risk_level, (245, 34, 45))

    # 背景矩形
    text_lines = [
        f"Lodging: {ratio:.1f}%",
        f"Risk: {risk_level}",
    ]
    font_size = max(14, image.shape[0] // 20)
    line_height = font_size + 6
    box_h = line_height * len(text_lines) + 10
    box_w = 200

    draw.rectangle([8, 8, 8 + box_w, 8 + box_h], fill=(0, 0, 0, 160))
    for i, line in enumerate(text_lines):
        draw.text((14, 12 + i * line_height), line, fill=text_color)

    return np.array(pil_img)


def create_mask_visualization(mask: np.ndarray) -> np.ndarray:
    """
    生成掩膜可视化图（伪彩色）

    Args:
        mask: 二值掩膜 (H, W)

    Returns:
        vis: 伪彩色掩膜图 (H, W, 3)，RGB
    """
    binary = (mask > 0).astype(np.uint8) * 255
    # 应用 JET 伪彩色映射
    colored = cv2.applyColorMap(binary, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def create_comparison_grid(
    original: np.ndarray,
    annotated: np.ndarray,
    mask_vis: np.ndarray,
    target_height: int = 300,
) -> np.ndarray:
    """
    将三张图拼接为横向对比图

    Args:
        original:      原始图像
        annotated:     标注图
        mask_vis:      掩膜可视化图
        target_height: 统一高度

    Returns:
        grid: 拼接后的对比图 (target_height, W*3, 3)
    """
    def resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
        ratio = h / img.shape[0]
        w = int(img.shape[1] * ratio)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    imgs = [
        resize_to_height(original, target_height),
        resize_to_height(annotated, target_height),
        resize_to_height(mask_vis, target_height),
    ]

    # 添加标题栏
    titles = ["原始图像", "倒伏标注图", "掩膜图"]
    title_h = 30
    result_imgs = []
    for img, title in zip(imgs, titles):
        title_bar = np.zeros((title_h, img.shape[1], 3), dtype=np.uint8)
        pil_bar = Image.fromarray(title_bar)
        draw = ImageDraw.Draw(pil_bar)
        draw.text((img.shape[1] // 2 - len(title) * 7, 6), title, fill=(255, 255, 255))
        title_bar = np.array(pil_bar)
        result_imgs.append(np.vstack([title_bar, img]))

    return np.hstack(result_imgs)


# ─────────────────────────────────────────────
# PIL Image 版本（供 Gradio 直接使用）
# ─────────────────────────────────────────────

def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """numpy (H,W,3) uint8 → PIL Image"""
    return Image.fromarray(img.astype(np.uint8))


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """PIL Image → numpy (H,W,3) uint8"""
    return np.array(img.convert("RGB"))


if __name__ == "__main__":
    # 快速测试
    H, W = 256, 256
    dummy_img = np.random.randint(50, 200, (H, W, 3), dtype=np.uint8)
    dummy_mask = np.zeros((H, W), dtype=np.uint8)
    dummy_mask[80:160, 60:200] = 255

    annotated = create_annotated_image(dummy_img, dummy_mask, ratio=18.5, risk_level="中")
    mask_vis = create_mask_visualization(dummy_mask)
    grid = create_comparison_grid(dummy_img, annotated, mask_vis)

    Image.fromarray(annotated).save("test_annotated.png")
    Image.fromarray(grid).save("test_grid.png")
    print("可视化测试完成，已保存 test_annotated.png 和 test_grid.png")
