## ADDED Requirements

### Requirement: 数据准备
系统 SHALL 支持 CamVid 数据集的加载与预处理。

#### Scenario: 加载标注数据
- **WHEN** 指定 CamVid 数据集路径
- **THEN** 加载 train/val/test 图像与标注（train 367, val 101, test 233）

### Requirement: PSPNet 模型实现
系统 SHALL 实现 PSPNet 架构用于语义分割。

#### Scenario: Pyramid Pooling Module
- **WHEN** 特征图输入 PPM
- **THEN** 输出融合了多尺度上下文的特征

### Requirement: 基线模型对比
系统 SHALL 支持至少一个基线模型进行对比实验。

#### Scenario: FCN 基线
- **WHEN** 使用相同训练配置
- **THEN** FCN 可在 CamVid val 上评测并输出 mIoU

### Requirement: 统一评测
系统 SHALL 使用统一评测脚本计算 mIoU。

#### Scenario: mIoU 计算
- **WHEN** 模型在 val 集推理完成
- **THEN** 计算 11 类 mIoU

### Requirement: 失败案例分析
系统 SHALL 分析模型预测失败的典型案例。

#### Scenario: 典型失败分析
- **WHEN** 模型预测错误
- **THEN** 记录小目标、边界模糊、遮挡等失败类型并分析原因
