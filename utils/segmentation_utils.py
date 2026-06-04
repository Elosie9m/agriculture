"""
分割后处理与灾害评估工具
包含像素统计、倒伏面积计算、风险等级评定
作者：202321156060 刘敏
"""

from typing import Dict, Tuple, Optional
import numpy as np
import cv2


# ─────────────────────────────────────────────
# 风险等级阈值（可按需调整）
# ─────────────────────────────────────────────
RISK_THRESHOLDS = {
    "低": (0.0, 10.0),    # 倒伏面积占比 < 10%
    "中": (10.0, 30.0),   # 10% ≤ 占比 < 30%
    "高": (30.0, 100.0),  # 占比 ≥ 30%
}

RISK_COLORS = {
    "低": "#52c41a",   # 绿色
    "中": "#faad14",   # 橙色
    "高": "#f5222d",   # 红色
}


# ─────────────────────────────────────────────
# 核心计算函数
# ─────────────────────────────────────────────

def compute_lodging_ratio(
    mask: np.ndarray,
    farmland_mask: Optional[np.ndarray] = None,
) -> float:
    """
    计算倒伏面积占比

    公式：倒伏面积占比 = 倒伏区域像素数 / 农田区域总像素数 × 100%

    Args:
        mask:          二值分割掩膜，形状 (H, W)，倒伏=1/255，背景=0
        farmland_mask: 农田区域掩膜（可选）；若为 None，则以整张图像为农田区域

    Returns:
        ratio: 倒伏面积占比（%），范围 [0, 100]
    """
    # 统一归一化为 0/1
    binary_mask = (mask > 0).astype(np.uint8)

    if farmland_mask is not None:
        farmland_binary = (farmland_mask > 0).astype(np.uint8)
        total_pixels = int(farmland_binary.sum())
        lodging_pixels = int((binary_mask & farmland_binary).sum())
    else:
        total_pixels = binary_mask.size
        lodging_pixels = int(binary_mask.sum())

    if total_pixels == 0:
        return 0.0

    ratio = lodging_pixels / total_pixels * 100.0
    return round(ratio, 2)


def assess_risk(ratio: float) -> str:
    """
    根据倒伏面积占比评定风险等级

    Args:
        ratio: 倒伏面积占比（%）

    Returns:
        risk_level: "低" / "中" / "高"
    """
    for level, (low, high) in RISK_THRESHOLDS.items():
        if low <= ratio < high:
            return level
    return "高"  # ratio == 100% 时兜底


def get_risk_color(risk_level: str) -> str:
    """返回风险等级对应的十六进制颜色"""
    return RISK_COLORS.get(risk_level, "#f5222d")


def full_assessment(
    mask: np.ndarray,
    farmland_mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    完整灾害评估：计算占比 + 风险等级 + 像素统计

    Returns:
        dict 包含：
            lodging_ratio   - 倒伏面积占比（%）
            risk_level      - 风险等级
            risk_color      - 风险颜色（十六进制）
            lodging_pixels  - 倒伏像素数
            total_pixels    - 农田总像素数
            has_lodging     - 是否存在倒伏（bool）
    """
    binary_mask = (mask > 0).astype(np.uint8)

    if farmland_mask is not None:
        farmland_binary = (farmland_mask > 0).astype(np.uint8)
        total_pixels = int(farmland_binary.sum())
        lodging_pixels = int((binary_mask & farmland_binary).sum())
    else:
        total_pixels = binary_mask.size
        lodging_pixels = int(binary_mask.sum())

    ratio = round(lodging_pixels / max(total_pixels, 1) * 100.0, 2)
    risk = assess_risk(ratio)

    return {
        "lodging_ratio": ratio,
        "risk_level": risk,
        "risk_color": get_risk_color(risk),
        "lodging_pixels": lodging_pixels,
        "total_pixels": total_pixels,
        "has_lodging": lodging_pixels > 0,
    }


# ─────────────────────────────────────────────
# 掩膜后处理
# ─────────────────────────────────────────────

def postprocess_mask(
    mask: np.ndarray,
    min_area: int = 200,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    对原始预测掩膜进行形态学后处理，去除噪声小区域

    Args:
        mask:        原始二值掩膜 (H, W)，值为 0/1 或 0/255
        min_area:    最小连通区域面积（像素），小于此值的区域将被删除
        kernel_size: 形态学操作核大小

    Returns:
        cleaned_mask: 处理后的二值掩膜 (H, W)，值为 0/255
    """
    binary = (mask > 0).astype(np.uint8) * 255

    # 闭运算：填充小孔洞
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 开运算：去除小噪点
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # 删除面积过小的连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    cleaned = np.zeros_like(opened)
    for label_id in range(1, num_labels):  # 0 是背景
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_id] = 255

    return cleaned


def mask_to_contours(mask: np.ndarray) -> list:
    """
    提取掩膜轮廓，用于可视化

    Returns:
        contours: OpenCV 轮廓列表
    """
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_bounding_boxes(mask: np.ndarray) -> list:
    """
    获取倒伏区域的外接矩形列表

    Returns:
        boxes: [(x, y, w, h), ...] 列表
    """
    contours = mask_to_contours(mask)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
    return boxes


if __name__ == "__main__":
    # 单元测试
    H, W = 256, 256
    test_mask = np.zeros((H, W), dtype=np.uint8)
    test_mask[50:150, 60:180] = 255  # 模拟倒伏区域

    result = full_assessment(test_mask)
    print("=== 灾害评估结果 ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    cleaned = postprocess_mask(test_mask)
    print(f"\n后处理前像素数: {(test_mask > 0).sum()}")
    print(f"后处理后像素数: {(cleaned > 0).sum()}")
