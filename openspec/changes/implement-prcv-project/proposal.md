# Change: 实施 PRCV 课程设计项目

## Why
完成 PRCV 课程设计与论文复现任务，包括 PSPNet 语义分割实验、模型对比评测、ResNet 论文复现，以及技术报告撰写。

## What Changes
- 实现 CamVid 数据加载与预处理流程
- 实现 PSPNet 模型并完成训练
- 实现 FCN 基线并进行统一评测对比
- 添加 DeepLabv3+ 作为强对照
- 复现 ResNet plain vs residual 对照实验
- 验证 degradation 现象并输出曲线图
- 完成失败案例分析与技术报告

## Impact
- Affected specs: pspnet-segmentation, resnet-reproduction
- Affected code: 新建 code/ 目录下的训练、评测、可视化代码
- 交付物: report.pdf, experiments/, devlog/

## Environment
- CPU: Intel i7-12650H
- RAM: 24GB
- GPU: NVIDIA RTX 4060 Mobile (8GB VRAM)
- Software: Python 3.12, PyTorch 2.9.1, torchvision 0.24, CUDA 13.0
- Datasets: CamVid (train 367, val 101, test 233), CIFAR-10

## Timeline Constraints
课程设计截止前完成，需要预留答辩准备时间。
