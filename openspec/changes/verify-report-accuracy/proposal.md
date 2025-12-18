# Change: 核实报告中引用数据的准确性

## Why

当前报告中关于原论文的引用数据（如 CIFAR-10 实验结果、ImageNet top-5 错误率、网络参数量等）均基于记忆编写，未经原文核实。这可能导致：

1. **事实性错误**：引用的数值与原论文不符
2. **学术不规范**：未经核实的数据引用有损学术诚信
3. **评分风险**：课程报告中的错误数据可能影响评分

## What Changes

### 下载参考文献
- 下载 ResNet 原论文 (He et al., 2016) PDF
- 下载其他关键引用文献（AlexNet, VGGNet 等）

### 核实并修正数据
- 核实 CIFAR-10 实验结果数据
- 核实 ImageNet 实验结果数据
- 核实网络架构参数（层数、通道数、参数量）
- 核实其他引用数据（AlexNet/VGG/GoogLeNet 的性能数据）

### 修正报告内容
- 修正报告中所有不准确的数值
- 补充缺失的重要数据
- 确保引用格式正确

## Impact

- Affected specs: `resnet-reproduction`
- Affected files: `report/resnet_report.md`
- New files: `references/*.pdf`
