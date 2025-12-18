# Change: 添加 ResNet18 轻量 backbone 对比实验

## Why

使用 ResNet50 预训练 backbone 时，PSPNet 和 FCN 在 CamVid 上取得了几乎相同的 mIoU（0.7535）。这说明：

1. **强 backbone 主导性能**：ResNet50 的特征表达能力足够强，架构差异（PPM vs 无）被掩盖
2. **小数据集效应**：CamVid 只有 367 张训练图，无法充分体现多尺度上下文的优势

为了更好地评估 PSPNet PPM 模块的贡献，需要使用**轻量 backbone**（ResNet18）进行对比实验。弱 backbone 的特征表达能力有限，此时 PPM 的多尺度聚合可能更有价值。

## What Changes

### 代码修改
- 放开 PSPNet/DeepLabV3+ 对 ResNet18 + output_stride=32 的支持
- 添加 `--backbone` 参数支持选择 backbone 类型
- FCN 使用 `fcn_resnet50` 或实现 ResNet18 版本

### 实验设计
- **实验组 A（已完成）**：ResNet50 backbone，验证预训练的重要性
- **实验组 B（新增）**：ResNet18 backbone，验证架构贡献

### 预期结果
- ResNet18 下各模型 mIoU 应低于 ResNet50
- PSPNet 与 FCN 的差距可能在 ResNet18 下更明显（PPM 弥补弱 backbone）

## Impact

- Affected specs: `pspnet-segmentation`
- Affected code:
  - `src/models/pspnet.py` - 支持 ResNet18 + output_stride=32
  - `src/models/deeplabv3plus.py` - 支持 ResNet18 + output_stride=32
  - `code/segmentation/train_camvid.py` - 添加 backbone 选择参数
