# Project Context

## Purpose
PRCV（模式识别与计算机视觉）课程设计项目：
1. **课程设计**：基于 PSPNet 的城市街景语义分割（Cityscapes）
2. **论文复现**：Deep Residual Learning for Image Recognition（ResNet, CVPR 2016）

## Hardware
- CPU: Intel i7-12650H
- RAM: 24GB
- GPU: NVIDIA RTX 4060 Mobile (8GB VRAM)

## Tech Stack
- Python 3.12
- PyTorch 2.9.1 (GPU) + torch.compile 加速
- torchvision 0.24
- CUDA 13.0
- matplotlib + scienceplots (可视化)
- pixi (包管理)

## PyTorch 2.9 Optimization
- `torch.compile()` - 模型编译加速
- `torch.set_float32_matmul_precision('high')` - TF32 矩阵运算
- `torch.amp` - 自动混合精度 (AMP)
- `torch.utils.checkpoint` - 梯度检查点节省显存
- `scaled_dot_product_attention` - Flash Attention (如适用)

## Project Structure
```
data/
  camvid/              # CamVid 数据集 (train 367, val 101, test 233)
  cifar10/             # CIFAR-10 数据集
code/
  segmentation/        # 分割模型代码
  resnet/              # ResNet 复现代码
experiments/
  pspnet/              # PSPNet 实验
  fcn/                 # FCN 基线
  deeplabv3plus/       # DeepLabv3+ 对照
  resnet/              # ResNet 复现实验
report/
  figures/             # 报告图片
devlog/                # 开发日志
src/utils/             # 公共工具 (plotting等)
```

## Training Config

### Segmentation (CamVid)
| 参数 | PSPNet | FCN | DeepLabv3+ |
|-----|--------|-----|------------|
| Backbone | ResNet-50 | ResNet-50 | ResNet-50 |
| Pretrain | ImageNet | ImageNet | ImageNet |
| Batch Size | 8 | 8 | 8 |
| Crop Size | 480x360 | 480x360 | 480x360 |
| Optimizer | SGD | SGD | SGD |
| LR | 0.01 | 0.01 | 0.01 |
| LR Schedule | Poly | Poly | Poly |
| Epochs | 100 | 100 | 100 |
| AMP | Yes | Yes | Yes |
| Compile | Yes | Yes | Yes |
| Classes | 11 | 11 | 11 |

### ResNet (CIFAR-10)
| 参数 | Plain-20/56 | ResNet-20/56 |
|-----|-------------|--------------|
| Batch Size | 128 | 128 |
| Optimizer | SGD (momentum=0.9) | SGD (momentum=0.9) |
| LR | 0.1 | 0.1 |
| LR Schedule | Step (÷10 @ 100,150) | Step (÷10 @ 100,150) |
| Epochs | 200 | 200 |
| Weight Decay | 1e-4 | 1e-4 |

## Project Conventions

### Code Style
- PEP 8 规范
- 配置文件使用 YAML
- 实验结果保存为 JSON/CSV

### Plotting Style
- 字体: HarmonyOS Sans (HarmonyOS_Sans_Regular.ttf)
- 样式: SciencePlots science 风格
- 图片格式: PNG (300 dpi)
- 无水印，清晰图注

### Architecture Patterns
- 数据加载、模型、训练逻辑分离
- 配置驱动的实验管理

### Testing Strategy
- 分割：CamVid val/test 集评测 mIoU
- ResNet：CIFAR-10，plain vs residual 对照实验

## Important Constraints
- 相同数据划分和评测标准下对比
- 报告含原版论文PDF、开发日志、成员贡献说明

## External Dependencies
- CamVid: https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
- Pretrained weights: torchvision (ImageNet)
- CIFAR-10: torchvision (auto download)

## Evaluation
- 分割: mIoU (mean Intersection over Union), ignore_index=255 (Void)
- ResNet: train loss, test accuracy
