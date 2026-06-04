# 基于无人机图像的农作物倒伏检测与灾害评估系统

**作者：202321156060 刘敏**

---

## 项目简介

本系统利用 U-Net 深度学习模型，对无人机拍摄的农田图像进行像素级语义分割，自动完成：

1. **倒伏分类** — 判断图像中是否存在倒伏现象
2. **区域定位** — 精确标注倒伏区域（红色高亮）
3. **灾害评估** — 计算倒伏面积占比，评定风险等级
4. **可视化展示** — Gradio Web 界面展示原图、标注图、掩膜图及评估结果

---

## 项目结构

```
project/
├── train.py                  # 模型训练脚本（数据增强 + 训练循环）
├── inference.py              # 推理脚本（加载模型、预测、计算倒伏面积）
├── app.py                    # Gradio 前端界面
├── models/
│   ├── __init__.py
│   └── unet_model.py         # U-Net 模型定义
├── utils/
│   ├── __init__.py
│   ├── data_loader.py        # 数据加载、增强、划分
│   ├── segmentation_utils.py # 分割后处理、像素统计、灾害评估
│   └── visualize.py          # 标注可视化工具
├── weights/                  # 存放训练好的 .pth 权重
├── dataset/
│   ├── images/               # 无人机农田图像
│   └── masks/                # 对应二值分割掩膜
└── requirements.txt          # 依赖库
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

**方式 A：使用真实数据集**

将图像放入 `dataset/images/`，对应掩膜放入 `dataset/masks/`（文件名需一致）。

**方式 B：自动生成合成演示数据**

```bash
python -c "from utils.data_loader import generate_demo_data; generate_demo_data('dataset', num_samples=100)"
```

### 3. 训练模型

```bash
# 基础训练（无真实数据时自动生成演示数据）
python train.py

# 自定义参数
python train.py --data_dir dataset --epochs 100 --batch_size 8 --lr 0.001
```

训练完成后权重保存至 `weights/unet_best.pth`。

### 4. 命令行推理

```bash
python inference.py --image path/to/image.jpg --weights weights/unet_best.pth
```

### 5. 启动 Web 界面

```bash
python app.py
# 或指定权重和端口
python app.py --weights weights/unet_best.pth --port 7860
```

浏览器访问 `http://localhost:7860`

---

## 灾害评估规则

| 风险等级 | 倒伏面积占比 | 颜色 |
|:-------:|:-----------:|:----:|
| 🟢 低   | < 10%       | 绿色 |
| 🟡 中   | 10% ~ 30%   | 橙色 |
| 🔴 高   | ≥ 30%       | 红色 |

**评估公式：**
```
倒伏面积占比 = 倒伏区域像素数 / 农田区域总像素数 × 100%
```

---

## 模型架构

采用经典 **U-Net** 编解码结构：

- **编码器**：4 层下采样（MaxPool + DoubleConv），通道数 64→128→256→512→1024
- **解码器**：4 层上采样（双线性插值 + DoubleConv），跳跃连接融合多尺度特征
- **输出层**：1×1 卷积，输出单通道 logits，sigmoid 后二值化得到分割掩膜
- **损失函数**：BCE Loss + Dice Loss 组合（各占 50%）
- **优化器**：AdamW + Cosine Annealing 学习率调度

---

## 数据增强策略

| 增强方式 | 参数 |
|---------|------|
| 随机水平翻转 | p=0.5 |
| 随机垂直翻转 | p=0.5 |
| 随机旋转 | ±30° |
| 亮度抖动 | ±30% |
| 对比度抖动 | ±30% |
| 饱和度抖动 | ±20% |
| ImageNet 标准化 | mean=[0.485,0.456,0.406] |

---

## 测试案例

系统内置 4 个预设测试案例：

| 案例 | 说明 | 预期结果 |
|------|------|---------|
| ✅ 正常案例 | 无倒伏农田 | 占比≈0%，低风险 |
| ⚠️ 中度倒伏 | 约20%倒伏 | 占比≈20%，中风险 |
| 🚨 严重倒伏 | 约45%倒伏 | 占比≈45%，高风险 |
| 🌫️ 模糊图像 | 图像质量差 | 系统仍可处理，结果仅供参考 |
