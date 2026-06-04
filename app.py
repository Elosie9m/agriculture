"""
农作物倒伏检测与灾害评估系统 — Gradio 前端
三模型串联：ResNet50分类 → U-Net分割 → 灾害评估
作者：202321156060 刘敏

启动：python app.py
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import gradio as gr

from inference import LodgingPipeline


# ─────────────────────────────────────────────
# 全局流水线（延迟初始化）
# ─────────────────────────────────────────────

_pipeline: Optional[LodgingPipeline] = None

def get_pipeline(cls_w=None, seg_w=None) -> LodgingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LodgingPipeline(
            cls_weights=cls_w if (cls_w and os.path.exists(cls_w)) else None,
            seg_weights=seg_w if (seg_w and os.path.exists(seg_w)) else None,
        )
    return _pipeline


# ─────────────────────────────────────────────
# 预设测试案例生成
# ─────────────────────────────────────────────

def _make_test_image(lodging_ratio=0.0, blur=False, size=(400,400), seed=0) -> Image.Image:
    """生成合成测试图像"""
    import numpy as np
    rng = np.random.default_rng(seed)
    H, W = size

    # 绿色农田背景 + 垄行纹理
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 0] = rng.integers(30, 60,  (H, W))
    img[:, :, 1] = rng.integers(90, 145, (H, W))
    img[:, :, 2] = rng.integers(20, 50,  (H, W))
    for y in range(0, H, 12):
        img[y:y+5, :, 1] = np.clip(img[y:y+5, :, 1] * 1.1, 0, 255)

    if lodging_ratio > 0:
        cx, cy = W // 2, H // 2
        rx = int(W * (lodging_ratio ** 0.5) * 0.75)
        ry = int(H * (lodging_ratio ** 0.5) * 0.6)
        Y, X = np.ogrid[:H, :W]
        ell = ((X-cx)**2/max(rx,1)**2 + (Y-cy)**2/max(ry,1)**2) <= 1
        n = ell.sum()
        img[ell, 0] = rng.integers(150, 200, n)
        img[ell, 1] = rng.integers(110, 155, n)
        img[ell, 2] = rng.integers(20,  60,  n)

    pil = Image.fromarray(img)
    if blur:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=3))
    return pil


DEMO_CASES = {
    "✅ 正常农田（无倒伏）":      _make_test_image(0.00, seed=1),
    "⚠️ 轻度倒伏（约8%）":       _make_test_image(0.08, seed=2),
    "🟡 中度倒伏（约20%）":      _make_test_image(0.20, seed=3),
    "🔴 严重倒伏（约45%）":      _make_test_image(0.45, seed=4),
    "🌫️ 异常：模糊图像":         _make_test_image(0.15, blur=True, seed=5),
    "🌾 异常：大面积倒伏（60%）": _make_test_image(0.60, seed=6),
}


# ─────────────────────────────────────────────
# 核心推理回调
# ─────────────────────────────────────────────

def analyze(image, cls_weights, seg_weights, cls_thr, seg_thr):
    if image is None:
        empty = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
        return empty, empty, empty, "—", "—", "<div>请上传图像</div>", "—"

    try:
        global _pipeline
        _pipeline = None   # 每次重新初始化（支持动态切换权重）
        pipeline = get_pipeline(cls_weights.strip() or None, seg_weights.strip() or None)
        pipeline.cls_threshold = float(cls_thr)
        pipeline.seg_threshold = float(seg_thr)

        result = pipeline.predict(image)

        orig_pil = Image.fromarray(result["original_image"])
        anno_pil = Image.fromarray(result["annotated_image"])
        mask_pil = Image.fromarray(result["mask_vis"])

        ratio     = result["lodging_ratio"]
        risk      = result["risk_level"]
        color     = result["risk_color"]
        has_lodge = result["has_lodging"]
        conf      = result["cls_confidence"]

        cls_text  = f"{'⚠️ 存在倒伏' if has_lodge else '✅ 正常'} （置信度 {conf:.1f}%）"
        ratio_text = f"{ratio:.2f}%"

        icon = {"低": "🟢", "中": "🟡", "高": "🔴"}.get(risk, "⚪")
        risk_html = f"""
        <div style="text-align:center;padding:14px;border-radius:10px;
                    background:{color}22;border:2px solid {color};">
            <div style="font-size:2.4em;">{icon}</div>
            <div style="font-size:1.7em;font-weight:700;color:{color};">{risk}风险</div>
        </div>"""

        detail = (
            f"【分类结果】{'存在倒伏' if has_lodge else '未检测到倒伏'}\n"
            f"【倒伏置信度】{conf:.1f}%\n"
            f"【倒伏面积占比】{ratio:.2f}%\n"
            f"【风险等级】{risk}\n"
            f"【倒伏像素数】{result['lodging_pixels']:,}\n"
            f"【农田总像素数】{result['total_pixels']:,}\n"
            f"【分类阈值】{cls_thr:.2f}  【分割阈值】{seg_thr:.2f}"
        )

        return orig_pil, anno_pil, mask_pil, cls_text, ratio_text, risk_html, detail

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        empty = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
        return empty, empty, empty, "推理失败", "—", f"<p style='color:red'>{e}</p>", tb


def load_demo(name):
    return DEMO_CASES.get(name, list(DEMO_CASES.values())[0])


# ─────────────────────────────────────────────
# Gradio 界面
# ─────────────────────────────────────────────

def build_ui(default_cls="weights/classifier_best.pth",
             default_seg="weights/unet_best.pth") -> gr.Blocks:

    css = """
    #header { background:linear-gradient(135deg,#1a5c2e,#2d9e4f,#1a5c2e);
              border-radius:12px; padding:22px 30px; text-align:center; color:#fff; margin-bottom:14px; }
    #badge  { display:inline-block; background:rgba(255,255,255,.2);
              border:1px solid rgba(255,255,255,.5); border-radius:20px;
              padding:4px 18px; font-size:.95em; margin-top:6px; }
    .card   { background:#f8f9fa; border-radius:10px; padding:14px;
              border:1px solid #e0e0e0; }
    """

    with gr.Blocks(title="农作物倒伏检测系统",
                   theme=gr.themes.Soft(primary_hue="green"), css=css) as demo:

        # 页眉
        gr.HTML("""
        <div id="header">
          <h1 style="margin:0;font-size:1.75em;font-weight:700;">
            🌾 基于无人机图像的农作物倒伏检测与灾害评估系统
          </h1>
          <p style="margin:6px 0 2px;opacity:.9;">
            ResNet50 分类 &nbsp;→&nbsp; U-Net 分割 &nbsp;→&nbsp; 灾害评估
          </p>
          <div id="badge">👤 202321156060 &nbsp; 刘敏</div>
        </div>
        """)

        with gr.Row():
            # ── 左侧输入 ──
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 📤 上传图像")
                inp_img = gr.Image(type="pil", label="无人机农田图像（JPG/PNG）", height=260)

                with gr.Accordion("⚙️ 模型设置", open=False):
                    cls_w = gr.Textbox(value=default_cls, label="分类器权重路径")
                    seg_w = gr.Textbox(value=default_seg, label="分割模型权重路径")
                    cls_thr = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="分类阈值")
                    seg_thr = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="分割阈值")

                btn = gr.Button("🔍 开始检测", variant="primary", size="lg")

                gr.Markdown("### 🧪 预设测试案例")
                demo_dd  = gr.Dropdown(choices=list(DEMO_CASES.keys()),
                                       value=list(DEMO_CASES.keys())[0],
                                       label="选择案例")
                load_btn = gr.Button("📂 加载案例", size="sm")

            # ── 右侧结果 ──
            with gr.Column(scale=2):
                gr.Markdown("### 📊 检测结果")

                with gr.Row():
                    out_orig = gr.Image(label="① 原始图像",       height=210)
                    out_anno = gr.Image(label="② 倒伏标注图",     height=210)
                    out_mask = gr.Image(label="③ 倒伏掩膜图",     height=210)

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🤖 分类结果")
                        out_cls = gr.Textbox(label="", interactive=False, elem_classes=["card"])
                    with gr.Column():
                        gr.Markdown("#### 📐 倒伏面积占比")
                        out_ratio = gr.Textbox(label="", interactive=False, elem_classes=["card"])
                    with gr.Column():
                        gr.Markdown("#### 🚦 风险等级")
                        out_risk = gr.HTML("<div style='text-align:center;padding:18px'>—</div>")

                with gr.Accordion("📋 详细信息", open=True):
                    out_detail = gr.Textbox(label="", lines=8, interactive=False)

        # 风险说明
        gr.Markdown("""
        ---
        ### 📖 风险等级 & 流水线说明

        | 等级 | 倒伏面积占比 | 建议 |
        |:---:|:---:|:---|
        | 🟢 低 | < 10% | 正常监测 |
        | 🟡 中 | 10%~30% | 加强巡查，评估干预必要性 |
        | 🔴 高 | ≥ 30% | 立即启动灾害应急响应 |

        **三阶段流水线**：`ResNet50（分类）` → `U-Net（像素级分割）` → `灾害评估公式`

        > 倒伏面积占比 = 倒伏区域像素数 ÷ 农田总像素数 × 100%
        """)

        gr.HTML("""
        <div style="text-align:center;padding:10px;color:#999;font-size:.85em;
                    border-top:1px solid #eee;margin-top:12px;">
          基于无人机图像的农作物倒伏检测与灾害评估系统 &nbsp;|&nbsp;
          ResNet50 + U-Net &nbsp;|&nbsp; <strong>202321156060 刘敏</strong>
        </div>
        """)

        # 事件绑定
        btn.click(
            fn=analyze,
            inputs=[inp_img, cls_w, seg_w, cls_thr, seg_thr],
            outputs=[out_orig, out_anno, out_mask, out_cls, out_ratio, out_risk, out_detail],
        )
        inp_img.change(
            fn=analyze,
            inputs=[inp_img, cls_w, seg_w, cls_thr, seg_thr],
            outputs=[out_orig, out_anno, out_mask, out_cls, out_ratio, out_risk, out_detail],
        )
        load_btn.click(fn=load_demo, inputs=[demo_dd], outputs=[inp_img])

    return demo


# ─────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cls_weights", type=str, default="weights/classifier_best.pth")
    p.add_argument("--seg_weights", type=str, default="weights/unet_best.pth")
    p.add_argument("--port",        type=int, default=7860)
    p.add_argument("--share",       action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.cls_weights):
        print(f"[TIP] 分类器权重不存在，演示模式。运行 python train.py 训练后效果更好。")
    if not os.path.exists(args.seg_weights):
        print(f"[TIP] 分割权重不存在，演示模式。运行 python train.py 训练后效果更好。")

    demo = build_ui(args.cls_weights, args.seg_weights)
    demo.launch(server_port=args.port, share=args.share, show_error=True)
