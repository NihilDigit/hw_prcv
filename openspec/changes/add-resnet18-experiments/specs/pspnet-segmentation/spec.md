## ADDED Requirements

### Requirement: 多 Backbone 对比实验
系统 SHALL 支持使用不同 backbone 进行对比实验，以评估模型架构的贡献。

#### Scenario: ResNet18 轻量 backbone
- **WHEN** 选择 ResNet18 作为 backbone
- **THEN** PSPNet/DeepLabV3+/FCN 均可正常训练和评测

#### Scenario: 架构贡献分析
- **WHEN** 对比 ResNet18 和 ResNet50 下各模型的 mIoU
- **THEN** 可分析 PPM/ASPP 模块在不同 backbone 强度下的贡献

## MODIFIED Requirements

### Requirement: PSPNet 模型实现
系统 SHALL 实现 PSPNet 架构用于语义分割。

#### Scenario: Pyramid Pooling Module
- **WHEN** 特征图输入 PPM
- **THEN** 输出融合了多尺度上下文的特征（1x1, 2x2, 3x3, 6x6 池化）

#### Scenario: 前向推理
- **WHEN** 输入 HxW 图像
- **THEN** 输出 HxW 的 11 类预测图

#### Scenario: 支持多种 backbone
- **WHEN** 配置使用 ResNet18 或 ResNet50
- **THEN** 模型使用对应的预训练 backbone 并调整 output_stride

### Requirement: 基线模型对比
系统 SHALL 支持至少一个基线模型（FCN）进行对比实验。

#### Scenario: FCN 基线
- **WHEN** 使用相同训练配置（预训练 backbone + PolyLR）
- **THEN** FCN 可在 CamVid val 上评测并输出 mIoU

#### Scenario: DeepLabv3+ 对照
- **WHEN** 选择 DeepLabv3+ 作为强对照
- **THEN** 使用手工实现的 ASPP + Decoder 架构进行对比

#### Scenario: 多 backbone 对比
- **WHEN** 分别使用 ResNet18 和 ResNet50 训练各模型
- **THEN** 可对比不同 backbone 强度下的架构贡献
