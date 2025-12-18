## ADDED Requirements

### Requirement: 双 Backbone 对比实验
系统 SHALL 支持 ResNet18 和 ResNet50 双 backbone 对比实验，以分析架构贡献与 backbone 强度的关系。

#### Scenario: 强 backbone 下架构趋同
- **WHEN** 使用 ResNet50 预训练 backbone 训练 PSPNet、FCN、DeepLabV3+
- **THEN** 三者 mIoU 接近（差距 < 1pp），说明预训练 backbone 主导性能

#### Scenario: 弱 backbone 下架构差异显现
- **WHEN** 使用 ResNet18 预训练 backbone 训练 PSPNet、FCN、DeepLabV3+
- **THEN** DeepLabV3+ 显著优于 PSPNet/FCN（差距 > 5pp）

### Requirement: 架构贡献分析
系统 SHALL 在报告中分析不同架构模块的贡献。

#### Scenario: PPM 贡献分析
- **WHEN** 对比 PSPNet 与 FCN 在相同 backbone 下的表现
- **THEN** 报告 PPM 多尺度池化的实际贡献（或缺乏贡献）

#### Scenario: Decoder 贡献分析
- **WHEN** 对比 DeepLabV3+ 与其他模型
- **THEN** 分析 Decoder 低层特征融合对边界恢复的作用

## MODIFIED Requirements

### Requirement: 失败案例分析
系统 SHALL 分析模型预测失败的典型案例。

#### Scenario: 小目标失败
- **WHEN** 模型对行人、路牌等小目标预测错误
- **THEN** 记录并分析失败原因

#### Scenario: 边界模糊
- **WHEN** 模型在类别边界处预测模糊
- **THEN** 对比不同模型的边界表现，特别是 Decoder 结构的优势

#### Scenario: 架构局限性分析
- **WHEN** PSPNet 表现不如预期
- **THEN** 分析 PPM 在小数据集/弱 backbone 下的局限性
