## ADDED Requirements

### Requirement: 统一预训练策略
系统 SHALL 对所有对比模型使用统一的 backbone 预训练策略以确保公平对比。

#### Scenario: ImageNet 预训练 backbone
- **WHEN** 初始化 PSPNet、DeepLabV3+、FCN 模型
- **THEN** backbone（ResNet-50）使用 ImageNet 预训练权重

#### Scenario: 手工实现模块随机初始化
- **WHEN** 初始化 PSPNet 的 PPM 或 DeepLabV3+ 的 ASPP+Decoder
- **THEN** 这些手工实现的模块使用随机初始化

### Requirement: PolyLR 学习率调度
系统 SHALL 使用 Polynomial 学习率调度器进行训练。

#### Scenario: Poly 衰减
- **WHEN** 训练进行到第 iter 次迭代（共 max_iter 次）
- **THEN** 学习率为 `base_lr × (1 - iter/max_iter)^0.9`

#### Scenario: 训练稳定性
- **WHEN** 使用 PolyLR 训练 PSPNet
- **THEN** val mIoU 曲线应平稳上升，无剧烈震荡

## MODIFIED Requirements

### Requirement: 基线模型对比
系统 SHALL 支持至少一个基线模型（FCN）进行对比实验。

#### Scenario: FCN 基线
- **WHEN** 使用相同训练配置（预训练 backbone + PolyLR）
- **THEN** FCN 可在 CamVid val 上评测并输出 mIoU

#### Scenario: DeepLabv3+ 对照
- **WHEN** 选择 DeepLabv3+ 作为强对照
- **THEN** 使用手工实现的 ASPP + Decoder 架构进行对比

#### Scenario: 预期性能排序
- **WHEN** 三个模型使用统一配置训练完成
- **THEN** 预期 mIoU 排序为 DeepLabV3+ >= PSPNet > FCN
