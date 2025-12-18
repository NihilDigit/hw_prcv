# PSPNet 城市街景语义分割

## Purpose
基于 PSPNet (Pyramid Scene Parsing Network) 的 CamVid 城市街景语义分割实验。通过 Pyramid Pooling Module 聚合多尺度上下文信息，实现像素级场景解析（11类），并与 FCN、DeepLabv3+ 等基线模型进行对比评测与失败案例分析。

## Requirements

### Requirement: 数据准备
系统 SHALL 支持 CamVid 数据集的加载与预处理。

#### Scenario: 加载标注数据
- **WHEN** 指定 CamVid 数据集路径
- **THEN** 加载 train/val/test 图像与标注（train 367, val 101, test 233）

#### Scenario: 数据增强
- **WHEN** 训练模式
- **THEN** 应用随机裁剪、翻转、归一化等增强

### Requirement: PSPNet 模型实现
系统 SHALL 实现 PSPNet 架构用于语义分割。

#### Scenario: Pyramid Pooling Module
- **WHEN** 特征图输入 PPM
- **THEN** 输出融合了多尺度上下文的特征（1x1, 2x2, 3x3, 6x6 池化）

#### Scenario: 前向推理
- **WHEN** 输入 HxW 图像
- **THEN** 输出 HxW 的 11 类预测图

### Requirement: 基线模型对比
系统 SHALL 支持至少一个基线模型（FCN）进行对比实验。

#### Scenario: FCN 基线
- **WHEN** 使用相同训练配置
- **THEN** FCN 可在 CamVid val 上评测并输出 mIoU

#### Scenario: DeepLabv3+ 对照
- **WHEN** 选择 DeepLabv3+ 作为强对照
- **THEN** 使用 ASPP + Decoder 架构进行对比

### Requirement: 统一评测
系统 SHALL 使用统一评测脚本计算 mIoU。

#### Scenario: mIoU 计算
- **WHEN** 模型在 val 集推理完成
- **THEN** 计算 11 类 mIoU

#### Scenario: 可视化输出
- **WHEN** 评测完成
- **THEN** 输出预测图与真值的并排对比图

### Requirement: 失败案例分析
系统 SHALL 分析模型预测失败的典型案例。

#### Scenario: 小目标失败
- **WHEN** 模型对行人、路牌等小目标预测错误
- **THEN** 记录并分析失败原因

#### Scenario: 边界模糊
- **WHEN** 模型在类别边界处预测模糊
- **THEN** 对比不同模型的边界表现
