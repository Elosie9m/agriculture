"""
U-Net 模型定义
用于农作物倒伏区域的像素级语义分割
作者：202321156060 刘敏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双卷积块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """下采样：MaxPool -> DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样：Upsample / ConvTranspose2d -> DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # 处理尺寸不匹配（padding）
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积层"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    标准 U-Net 网络
    输入：RGB 图像 (B, 3, H, W)
    输出：分割掩膜 logits (B, num_classes, H, W)
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True):
        """
        Args:
            n_channels: 输入通道数（RGB=3）
            n_classes:  输出类别数（二分类倒伏=1）
            bilinear:   True 使用双线性上采样，False 使用转置卷积
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # 编码器（下采样路径）
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # 解码器（上采样路径）
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # 输出层
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码（跳跃连接）
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        推理接口：返回二值掩膜
        Args:
            x: 输入图像张量
            threshold: sigmoid 阈值
        Returns:
            binary_mask: (B, 1, H, W) 二值掩膜
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            prob = torch.sigmoid(logits)
            binary_mask = (prob > threshold).float()
        return binary_mask


def get_model(n_classes: int = 1, pretrained_path: str = None, device: str = "cpu") -> UNet:
    """
    工厂函数：创建并可选加载预训练权重
    Args:
        n_classes:       分割类别数
        pretrained_path: 权重文件路径（.pth），None 则随机初始化
        device:          运行设备
    Returns:
        model: UNet 实例
    """
    model = UNet(n_channels=3, n_classes=n_classes, bilinear=True)
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location=device)
        # 兼容 DataParallel 保存的权重
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"[INFO] 已加载预训练权重：{pretrained_path}")
    model.to(device)
    return model


if __name__ == "__main__":
    # 快速验证模型结构
    model = UNet(n_channels=3, n_classes=1)
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print(f"输入形状: {dummy.shape}")
    print(f"输出形状: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {total_params:,}")
