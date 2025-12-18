# Delta Spec: PSPNet 城市街景语义分割（实验结论与分析结构更新）

## MODIFIED Requirements

### Requirement: 基线模型对比
系统 SHALL 支持 PSPNet、FCN、DeepLabv3+ 在同一评测协议下进行对比实验，并覆盖至少两种 backbone 强度（强：ResNet50；弱：ResNet18）。

#### Scenario: 双 backbone 对比表
- **WHEN** 完成 ResNet50 与 ResNet18 两组实验
- **THEN** 报告中给出每组 backbone 的对比表（mIoU + 训练配置要点 + run id）

## ADDED Requirements

### Requirement: Backbone 强度分析
系统 SHALL 对比强/弱 backbone 下模型表现差异，并给出可复现的结论与证据（来自训练日志与评测输出）。

#### Scenario: ΔmIoU 分析
- **WHEN** 取得 ResNet50 与 ResNet18 的 val mIoU
- **THEN** 计算并解释每个模型的 ΔmIoU（强 backbone → 弱 backbone）

### Requirement: 架构贡献分析
系统 SHALL 分析 PPM（PSPNet）与 decoder（DeepLabv3+）在不同 backbone 强度下的相对贡献。

#### Scenario: PPM vs Decoder 贡献对照
- **WHEN** 同一 backbone 下得到 PSPNet、FCN、DeepLabv3+ 的 mIoU
- **THEN** 在报告中给出差值对照（PSPNet-FCN、DeepLabv3+-PSPNet）并解释结论

### Requirement: 报告结论更新
系统 SHALL 在报告结论中反映对比实验的新发现（包括负面结果），并与失败案例分析一致。

#### Scenario: 结论与失败案例一致
- **WHEN** 更新第四章分析结构
- **THEN** 第五章结论应明确概括 backbone 与结构贡献，并与 失败案例分析 的观察一致
