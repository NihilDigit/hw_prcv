# prcv

这是一个计算机视觉课程项目，包含两个完整实验：
1) ResNet 在 CIFAR-10 上的论文复现
2) PSPNet 在 CamVid 上的语义分割实现与对比实验（FCN / DeepLabv3+）

本 README 写得啰嗦一些，适合基础较浅的读者从零开始理解项目结构和怎么跑起来。

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

（具体脚本可自行补充，这里只给结构理解）

核心模型代码：
- `src/models/resnet_cifar.py`

报告复现结果已经在：
- `report/resnet_report.md`
- 曲线图在 `report/figures/`（若有）

PPT 在：
- `resnet_ppt/resnet_ppt.pptx`

## 5. PSPNet 实验如何复现？

请先阅读更详细的说明：
- `code/segmentation/README.md`

该文件包含：
- CamVid 数据目录结构
- 训练命令（PSPNet / FCN / DeepLabv3+）
- 曲线绘制命令
- 评测 + 可视化命令

## 6. 分割演示前端（Gradio）

这个前端会加载训练好的权重，输入一张图，输出分割结果：

```bash
pixi run python app.py
```

前端需要的权重路径（默认）：
- `experiments/segmentation/seg-v2-pspnet/best.pth`
- `experiments/segmentation/seg-r18-pspnet/best.pth`
- `experiments/segmentation/seg-v2-fcn/best.pth`
- `experiments/segmentation/seg-r18-fcn/best.pth`
- `experiments/segmentation/seg-v2-deeplabv3plus/best.pth`
- `experiments/segmentation/seg-r18-deeplabv3plus/best.pth`

前端还会从：
- `data/camvid/test/` 读取测试图片样例

如果没有权重或数据集，前端会提示 “Missing Weights”。

## 7. 注意事项

- `data/` 和 `experiments/` 中的内容通常很大，不会提交到 git。
- 如果你克隆后没有训练产物，很多推理/可视化功能会无法运行，这是正常现象。
- 报告和 PPT 已经包含完整实验结论，阅读这些即可了解最终结果。
