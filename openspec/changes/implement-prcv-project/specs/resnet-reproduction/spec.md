## ADDED Requirements

### Requirement: Plain Network 实现
系统 SHALL 实现无 shortcut 的 plain 深层网络。

#### Scenario: Plain 网络构建
- **WHEN** 构建 20/56 层 plain 网络
- **THEN** 仅使用卷积-BN-ReLU 堆叠，无跳跃连接

### Requirement: ResNet 实现
系统 SHALL 实现带 shortcut 的残差网络。

#### Scenario: ResNet 构建
- **WHEN** 构建 20/56 层 ResNet
- **THEN** 每个 block 包含 shortcut 连接

### Requirement: CIFAR-10 训练
系统 SHALL 在 CIFAR-10 数据集上进行训练。

#### Scenario: 训练执行
- **WHEN** 开始训练
- **THEN** 使用 SGD + momentum，标准学习率策略

### Requirement: Degradation 现象验证
系统 SHALL 验证 plain 网络的退化现象。

#### Scenario: 曲线对比
- **WHEN** Plain vs ResNet 训练完成
- **THEN** 输出曲线验证深层 plain 网络 degradation

### Requirement: 结果可视化
系统 SHALL 输出训练/测试曲线图。

#### Scenario: 曲线绘制
- **WHEN** 训练结束
- **THEN** 绘制 train loss 和 test accuracy 对比曲线
