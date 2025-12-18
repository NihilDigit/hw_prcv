# Change: 统一训练配置与预训练策略

## Why

当前训练配置存在两个问题：
1. **所有模型从零训练**：backbone 未使用 ImageNet 预训练，导致收敛困难
2. **无学习率调度**：固定 lr=0.01 对复杂模型（PSPNet、DeepLabV3+）过于激进，导致训练不稳定

训练日志显示 PSPNet 的 val mIoU 剧烈震荡（epoch 44 从 0.498 崩到 0.172），最终 best mIoU=0.5539 低于 FCN baseline 的 0.5874。这不符合 PSPNet 理论上应优于 FCN 的预期。

## What Changes

### 预训练策略
- 所有模型 backbone（ResNet-50）统一使用 ImageNet 预训练权重
- 手工实现的模块保持随机初始化：
  - PSPNet: PPM + Classifier
  - DeepLabV3+: ASPP + Decoder
- FCN 使用 torchvision baseline（预训练 backbone）

### 学习率调度
- 添加 PolyLR 调度器：`lr = base_lr × (1 - iter/max_iter)^0.9`
- 这是 PSPNet/DeepLab 系列论文的标准做法

### 设计决策：保留手工实现
- **不使用** torchvision 的 DeepLabV3（只有 ASPP，无 decoder）
- **保留** 手工实现的 DeepLabV3+（ASPP + Decoder），体现课程设计工作量
- DeepLabV3+ 的 decoder 可融合低层特征，理论上边界恢复更好

## Impact

- Affected specs: `pspnet-segmentation`
- Affected code:
  - `src/models/pspnet.py` - backbone 加载预训练
  - `src/models/deeplabv3plus.py` - backbone 加载预训练
  - `code/segmentation/train_camvid.py` - FCN 预训练 + PolyLR
