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
系统 SHALL 输出完整的训练/测试曲线图及网络结构示意图。

#### Scenario: Loss 曲线
- **WHEN** 训练结束
- **THEN** 绘制 Plain-20/56, ResNet-20/56 的 train loss 对比曲线

#### Scenario: Accuracy 曲线
- **WHEN** 训练结束
- **THEN** 绘制 test accuracy 曲线对比

#### Scenario: 网络结构图
- **WHEN** 报告需要解释网络结构
- **THEN** 提供 BasicBlock 示意图（含 shortcut 路径）和 Plain vs ResNet 整体架构对比图

#### Scenario: 学习率曲线
- **WHEN** 报告需要解释训练策略
- **THEN** 绘制 learning rate schedule 可视化图

#### Scenario: 超参数对比表
- **WHEN** 报告需要说明实验设置
- **THEN** 提供原论文 vs 本次复现的超参数对比表格

### Requirement: 学术规范报告
报告 SHALL 符合学术论文写作规范。

#### Scenario: 文献综述完整性
- **WHEN** 撰写背景介绍
- **THEN** 详细阐述 AlexNet、VGG、GoogLeNet、BatchNorm 的原理与网络结构

#### Scenario: 数学表述规范
- **WHEN** 解释算法原理
- **THEN** 使用规范的数学公式（残差公式、损失函数、学习率衰减策略）

#### Scenario: 术语一致性
- **WHEN** 全文引用概念
- **THEN** 使用统一的学术术语（degradation、shortcut、residual learning 等）

#### Scenario: 引用规范
- **WHEN** 引用他人工作
- **THEN** 使用标准学术引用格式，确保所有引用完整

### Requirement: 课程要求对齐
报告 SHALL 满足课程设计报告的全部要求。

#### Scenario: 章节结构
- **WHEN** 组织报告内容
- **THEN** 按照"背景介绍-算法原理-程序设计-程序测试-结论"五章结构撰写

#### Scenario: 附件完整性
- **WHEN** 提交最终报告
- **THEN** 包含开发日志、原版论文、成员分工说明

#### Scenario: 图表规范
- **WHEN** 插入图表
- **THEN** 图表有清晰标注说明、无水印、格式统一美观

