# prcv

这是一个计算机视觉课程项目，包含两个完整实验：
1) ResNet 在 CIFAR-10 上的论文复现
2) PSPNet 在 CamVid 上的语义分割实现与对比实验（FCN / DeepLabv3+）

## 1. 两个实验分别做了什么？

### A. ResNet（CIFAR-10 复现）
目标：复现 He 等人在 CIFAR-10 上的核心结论
- 验证“深层 plain 网络退化问题”
- 验证残差连接能缓解退化并提升性能
- 对照模型：Plain-20 / Plain-56 / ResNet-20 / ResNet-56
- 结果、曲线、分析已写入报告与 PPT

主要产物：
- 报告：`report/resnet_report.md`
- PPT：`resnet_ppt/resnet_ppt.pptx`

### B. PSPNet（CamVid 语义分割）
目标：实现 PSPNet，并与 FCN / DeepLabv3+ 做统一对照
- 比较强/弱 backbone（ResNet-50 vs ResNet-18）
- 分析 PPM 与 decoder 的结构贡献
- 输出训练曲线 + 定量 mIoU + 定性可视化

主要产物：
- 报告：`report/pspnet_report.md`
- PPT 包：`pspnet_ppt/`（含 `psp_ppt.pptx` 和素材）

## 2. 目录结构说明（常见问题）

```
.
├── code/                  # 训练 / 评测 / 可视化脚本
├── src/                   # 模型代码（ResNet/PSPNet/FCN/DeepLabv3+）
├── data/                  # 数据集（不跟踪）
├── experiments/           # 训练产物（不跟踪）
├── report/                # 报告与论文图表
├── resnet_ppt/            # ResNet PPT 最终版
├── pspnet_ppt/            # PSPNet PPT 制作包（含素材）
├── app.py                 # 分割演示前端（Gradio）
├── pixi.toml              # 环境与依赖
└── README.md              # 本文件
```

如果你只想看结果：
- ResNet 报告：`report/resnet_report.md`
- PSPNet 报告：`report/pspnet_report.md`
- ResNet PPT：`resnet_ppt/resnet_ppt.pptx`
- PSPNet PPT：`pspnet_ppt/psp_ppt.pptx`

## 3. 环境配置（建议步骤）

本项目用 `pixi` 管理环境。你可以按下面步骤配置：

1) 安装 pixi（只需一次）
官方安装方式参考：`https://pixi.sh`

2) 在项目根目录执行：

```bash
pixi install
```

3) 验证依赖是否可用：

```bash
pixi run python -c "import torch, torchvision, gradio; print('ok')"
```

依赖说明（简化理解）：
- `pytorch-gpu` / `torchvision`：深度学习框架
- `matplotlib` / `scienceplots`：画曲线
- `gradio`：演示前端

## 4. ResNet 实验如何复现？

### 4.1 数据准备（CIFAR-10）
项目默认使用 torchvision 的 CIFAR-10 下载逻辑（训练脚本会在首次运行时自动下载）。  
如果你所在环境无法联网，请提前准备数据并把 CIFAR-10 放到 `data/` 下。

### 4.2 训练与复现流程（建议步骤）
下面是一个“从零开始”的完整流程示例。具体参数可按需求调整。

1) 进入项目根目录  
2) 运行训练（示例命令）

```bash
# 以 20 层 ResNet 为例（可根据脚本参数调整）
pixi run python code/resnet/train_cifar.py --model resnet20 --epochs 200 --batch-size 128

# 以 56 层 ResNet 为例
pixi run python code/resnet/train_cifar.py --model resnet56 --epochs 200 --batch-size 128

# 以 20 层 Plain 为例
pixi run python code/resnet/train_cifar.py --model plain20 --epochs 200 --batch-size 128

# 以 56 层 Plain 为例
pixi run python code/resnet/train_cifar.py --model plain56 --epochs 200 --batch-size 128
```

> 说明：训练脚本路径/参数若与你本地版本不一致，请以 `code/resnet/README.md` 或脚本 `--help` 为准。

### 4.3 结果输出位置
训练完成后，日志与模型权重默认会输出到：  
`experiments/resnet/...`

曲线与可视化（若有）通常会输出到：  
`report/figures/`

### 4.4 报告与 PPT
- 报告：`report/resnet_report.md`
- PPT：`resnet_ppt/resnet_ppt.pptx`

### 4.5 代码入口（核心模型）
- `src/models/resnet_cifar.py`

## 5. PSPNet 实验如何复现？

### 5.1 数据准备（CamVid）
CamVid 数据目录结构要求如下（与你的实际路径一致即可）：  

```
data/camvid/
  train/      *.png
  trainannot/ *.png
  val/        *.png
  valannot/   *.png
  test/       *.png
  testannot/  *.png
```

### 5.2 训练命令（完整示例）
以下命令与 `code/segmentation/README.md` 保持一致：

```bash
# PSPNet（默认 ResNet-50 backbone）
pixi run python code/segmentation/train_camvid.py --model pspnet --epochs 50 --batch-size 4

# PSPNet + ResNet-18（弱 backbone 对比）
pixi run python code/segmentation/train_camvid.py --model pspnet --backbone resnet18 --epochs 50 --batch-size 4

# FCN baseline
pixi run python code/segmentation/train_camvid.py --model fcn --epochs 50 --batch-size 4

# FCN + ResNet-18
pixi run python code/segmentation/train_camvid.py --model fcn --backbone resnet18 --epochs 50 --batch-size 4

# DeepLabv3+（本项目实现）
pixi run python code/segmentation/train_camvid.py --model deeplabv3plus --epochs 50 --batch-size 4

# DeepLabv3+ + ResNet-18
pixi run python code/segmentation/train_camvid.py --model deeplabv3plus --backbone resnet18 --epochs 50 --batch-size 4
```

### 5.3 曲线绘制

```bash
pixi run python code/segmentation/plot_training_curves.py --run-dir experiments/segmentation/<run-id>
```

曲线默认保存到：  
`report/figures/segmentation/<run-id>/`

### 5.4 评测 + 可视化

```bash
pixi run python code/segmentation/eval_camvid.py --model pspnet --split val \
  --checkpoint experiments/segmentation/<run-id>/best.pth \
  --out-dir report/figures/segmentation/<run-id>/eval_pspnet
```

### 5.5 报告与 PPT
- 报告：`report/pspnet_report.md`
- PPT 包：`pspnet_ppt/`（含 `psp_ppt.pptx` 与素材）

## 6. 分割演示前端（Gradio）

这个前端会加载训练好的权重，输入一张图，输出分割结果。

### 6.1 启动方式

```bash
pixi run python app.py
```

### 6.2 需要的权重（默认路径）
- `experiments/segmentation/seg-v2-pspnet/best.pth`
- `experiments/segmentation/seg-r18-pspnet/best.pth`
- `experiments/segmentation/seg-v2-fcn/best.pth`
- `experiments/segmentation/seg-r18-fcn/best.pth`
- `experiments/segmentation/seg-v2-deeplabv3plus/best.pth`
- `experiments/segmentation/seg-r18-deeplabv3plus/best.pth`

### 6.3 需要的输入数据
- `data/camvid/test/` 读取测试图片样例

如果没有权重或数据集，前端会提示 “Missing Weights”。

## 7. 注意事项

- `data/` 和 `experiments/` 中的内容通常很大，不会提交到 git。
- 如果你克隆后没有训练产物，很多推理/可视化功能会无法运行，这是正常现象。
- 报告和 PPT 已经包含完整实验结论，阅读这些即可了解最终结果。
