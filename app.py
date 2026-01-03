import os
import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms
from pathlib import Path

# 环境配置
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pspnet import PSPNet, PSPNetConfig
from src.models.deeplabv3plus import DeepLabV3Plus, DeepLabV3PlusConfig
from src.models.fcn import FCNConfig, build_fcn
from src.data.camvid import CAMVID_11_COLORS, CAMVID_11_CLASSES

# --- 推理后端 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CROP_SIZE = (360, 480)
WEIGHTS_MAP = {
    ("PSPNet", "ResNet-50"): "experiments/segmentation/seg-v2-pspnet/best.pth",
    ("PSPNet", "ResNet-18"): "experiments/segmentation/seg-r18-pspnet/best.pth",
    ("FCN", "ResNet-50"): "experiments/segmentation/seg-v2-fcn/best.pth",
    ("FCN", "ResNet-18"): "experiments/segmentation/seg-r18-fcn/best.pth",
    ("DeepLabV3+", "ResNet-50"): "experiments/segmentation/seg-v2-deeplabv3plus/best.pth",
    ("DeepLabV3+", "ResNet-18"): "experiments/segmentation/seg-r18-deeplabv3plus/best.pth",
}

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MODEL_CACHE = {}

def load_model(model_type, backbone):
    path = WEIGHTS_MAP.get((model_type, backbone))
    if not path or not os.path.exists(path): return None
    bn = backbone.lower().replace("-", "")
    os_stride = 8 if bn == "resnet50" else 32
    if model_type == "PSPNet":
        m = PSPNet(PSPNetConfig(num_classes=11, backbone=bn, output_stride=os_stride, aux_loss=False))
    elif model_type == "DeepLabV3+":
        m = DeepLabV3Plus(DeepLabV3PlusConfig(num_classes=11, backbone=bn, output_stride=os_stride))
    else:
        m = build_fcn(FCNConfig(num_classes=11, backbone=bn))
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    m.load_state_dict(sd, strict=False)
    return m.to(DEVICE).eval()

def predict(img, m_type, backbone, alpha):
    if img is None: return None, None, None
    key = (m_type, backbone)
    if key not in MODEL_CACHE: MODEL_CACHE[key] = load_model(m_type, backbone)
    model = MODEL_CACHE[key]
    if model is None: return None, None, {"Error": "Missing Weights"}

    t = IMG_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t)
        if isinstance(out, dict): out = out["out"]
        if isinstance(out, tuple): out = out[0]
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

    mask_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for i in range(11): mask_rgb[pred == i] = CAMVID_11_COLORS[i]
    
    mask_pil = Image.fromarray(mask_rgb).resize(img.size, Image.NEAREST)
    overlay = Image.blend(img.resize(img.size), mask_pil.convert("RGB"), alpha=alpha)
    counts = np.bincount(pred.flatten(), minlength=11)
    stats = {CAMVID_11_CLASSES[i]: float(counts[i]/counts.sum()) for i in range(11)}
    return img, overlay, stats

# --- 极简学术 UI ---
theme = gr.themes.Soft(primary_hue="slate", font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"])

with gr.Blocks(title="Segmentation Analysis") as demo:
    gr.Markdown("<div style='text-align: center; margin-bottom: 20px;'> <h1 style='font-weight: 700;'>语义分割模型对比分析系统</h1> <p style='color: #666;'>基于 CamVid 数据集的算法复现与结构增益评估</p> </div>")
    
    with gr.Row():
        # 左侧：控制面板与统计
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 1. 模型参数配置")
                model_sel = gr.Dropdown(choices=["PSPNet", "DeepLabV3+", "FCN"], value="PSPNet", label="选择模型架构")
                backbone_sel = gr.Radio(choices=["ResNet-50", "ResNet-18"], value="ResNet-50", label="骨干网络 (Backbone)")
                alpha_slider = gr.Slider(0, 1, 0.5, step=0.1, label="融合透明度 (Alpha)")
                run_btn = gr.Button("运行推理分析", variant="primary")
            
            with gr.Group():
                gr.Markdown("### 2. 类别像素占比")
                output_stats = gr.Label(num_top_classes=6, label="Category Distribution", show_label=False)

        # 右侧：双图展示区
        with gr.Column(scale=2):
            gr.Markdown("### 3. 分割推理结果对比")
            with gr.Row():
                output_original = gr.Image(type="pil", interactive=False, label="原始街景图")
                output_overlay = gr.Image(type="pil", interactive=False, label="预测融合图")

    # 底部：案例库
    with gr.Row():
        with gr.Column():
            gr.Markdown("<hr style='margin: 30px 0;'>")
            gr.Markdown("### 4. 典型测试案例库 (CamVid Test Set)")
            input_img = gr.Image(type="pil", label="选定输入", visible=False)
            
            example_path = "data/camvid/test"
            if os.path.exists(example_path):
                examples_list = sorted([str(Path(example_path)/f) for f in os.listdir(example_path) if f.endswith(".png")])[:12]
                gr.Examples(
                    examples=examples_list,
                    inputs=input_img,
                    examples_per_page=12,
                    label=None
                )

    # 绑定逻辑
    run_btn.click(predict, [input_img, model_sel, backbone_sel, alpha_slider], [output_original, output_overlay, output_stats])
    input_img.change(predict, [input_img, model_sel, backbone_sel, alpha_slider], [output_original, output_overlay, output_stats])

if __name__ == "__main__":
    demo.launch(theme=theme)
