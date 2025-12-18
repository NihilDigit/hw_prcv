# ResNet 论文复现

## Purpose
复现 Deep Residual Learning for Image Recognition (CVPR 2016)，验证 degradation 现象与残差连接的效果。

## Requirements

### Requirement: Plain Network 实现
系统 SHALL 实现无 shortcut 的 plain 深层网络。

#### Scenario: Plain-20
- **WHEN** 构建 20 层 plain 网络
- **THEN** 仅使用卷积-BN-ReLU 堆叠，无跳跃连接

#### Scenario: Plain-56
- **WHEN** 构建 56 层 plain 网络
- **THEN** 结构同上，层数增加到 56

### Requirement: ResNet 实现
系统 SHALL 实现带 shortcut 的残差网络。

#### Scenario: ResNet-20
- **WHEN** 构建 20 层 ResNet
- **THEN** 每个 block 包含 shortcut 连接

#### Scenario: ResNet-56
- **WHEN** 构建 56 层 ResNet
- **THEN** 结构同上，层数增加到 56

#### Scenario: 维度匹配
- **WHEN** shortcut 两端维度不同
- **THEN** 使用 1x1 conv projection 或 zero-padding 匹配

### Requirement: CIFAR-10 训练
系统 SHALL 在 CIFAR-10 数据集上进行训练。

#### Scenario: 数据加载
- **WHEN** 加载 CIFAR-10
- **THEN** 使用标准 train/test 划分（50000/10000）

#### Scenario: 训练配置
- **WHEN** 开始训练
- **THEN** 使用 SGD + momentum，学习率按策略衰减

### Requirement: Degradation 现象验证
系统 SHALL 验证 plain 网络的退化现象。

#### Scenario: 训练曲线对比
- **WHEN** Plain-20 vs Plain-56 训练完成
- **THEN** 输出 train loss/error 曲线，验证深层 plain 训练误差更高

#### Scenario: 残差改善验证
- **WHEN** ResNet-20 vs ResNet-56 训练完成
- **THEN** 验证深层 ResNet 训练误差正常下降

### Requirement: 结果可视化
系统 SHALL 输出训练/测试曲线图。

#### Scenario: Loss 曲线
- **WHEN** 训练结束
- **THEN** 绘制 Plain-20/56, ResNet-20/56 的 train loss

#### Scenario: Accuracy 曲线
- **WHEN** 训练结束
- **THEN** 绘制 test accuracy 曲线对比
