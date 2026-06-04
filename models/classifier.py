"""
ResNet50 图像分类模型
用于判断农田图像中是否存在倒伏现象（二分类）
迁移学习：使用 ImageNet 预训练权重
作者：202321156060 刘敏
"""

import torch
import torch.nn as nn
from torchvision import models


class LodgingClassifier(nn.Module):
    """
    基于 ResNet50 的倒伏二分类器
    输入：(B, 3, 224, 224)
    输出：(B, 2)  logits，0=正常 1=倒伏
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()

        # 加载 ResNet50 骨干网络
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # 冻结骨干（可选，用于数据量少时防止过拟合）
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # 替换最后的全连接层
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """返回 softmax 概率 (B, 2)"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        返回预测标签和置信度
        Returns:
            labels: (B,) 0=正常 1=倒伏
            confidence: (B,) 倒伏概率
        """
        proba = self.predict_proba(x)
        confidence = proba[:, 1]          # 倒伏类概率
        labels = (confidence >= threshold).long()
        return labels, confidence


def get_classifier(pretrained_path: str = None, device: str = "cpu") -> LodgingClassifier:
    """
    工厂函数：创建分类器并可选加载权重
    """
    model = LodgingClassifier(num_classes=2, pretrained=(pretrained_path is None))
    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location=device)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"[INFO] 已加载分类器权重：{pretrained_path}")
    model.to(device)
    return model


if __name__ == "__main__":
    model = LodgingClassifier(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"输入: {x.shape}  输出: {out.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {params:,}")
