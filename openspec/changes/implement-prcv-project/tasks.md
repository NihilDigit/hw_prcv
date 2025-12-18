# Tasks: PRCV 课程设计实施

## 1. 环境与数据准备
- [x] 1.1 下载 CamVid 数据集
- [ ] 1.2 下载 CIFAR-10 数据集
- [x] 1.3 创建项目目录结构（code/, experiments/, report/, devlog/）
- [ ] 1.4 验证 GPU 环境可用（torch.cuda.is_available）

## 2. PSPNet 语义分割（组员A主责）
- [ ] 2.1 实现 CamVid 数据加载器（train/val split, augmentation）
- [ ] 2.2 实现 PSPNet 模型（ResNet backbone + PPM）
- [ ] 2.3 实现训练脚本（loss, optimizer, lr scheduler）
- [ ] 2.4 训练 PSPNet 并保存 checkpoint
- [ ] 2.5 在 val 集上评测，记录 mIoU
- [ ] 2.6 输出可视化预测结果

## 3. 分割基线对比（组员B主责）
- [ ] 3.1 实现/加载 FCN 基线模型
- [ ] 3.2 使用相同配置训练 FCN
- [ ] 3.3 在 val 集上评测 FCN
- [ ] 3.4 实现 DeepLabv3+ 作为强对照
- [ ] 3.5 统一评测脚本计算 mIoU
- [ ] 3.6 生成模型对比表格（mIoU, Params, FPS）
- [ ] 3.7 收集失败案例（小目标、边界、遮挡）
- [ ] 3.8 撰写失败案例分析

## 4. ResNet 论文复现（组员C主责）
- [ ] 4.1 实现 Plain-20/56 网络
- [ ] 4.2 实现 ResNet-20/56 网络
- [ ] 4.3 实现 CIFAR-10 训练脚本
- [ ] 4.4 训练四组实验：Plain-20, Plain-56, ResNet-20, ResNet-56
- [ ] 4.5 记录训练日志（loss, accuracy per epoch）
- [ ] 4.6 绘制 train loss 对比曲线
- [ ] 4.7 绘制 test accuracy 对比曲线
- [ ] 4.8 分析 degradation 现象并撰写解释

## 5. 报告撰写（组员C统编）
- [ ] 5.1 撰写第一章：背景介绍与文献综述
- [ ] 5.2 撰写第二章：算法原理（PSPNet + ResNet）
- [ ] 5.3 撰写第三章：程序设计与实现
- [ ] 5.4 撰写第四章：程序测试与结果分析
- [ ] 5.5 撰写第五章：结论
- [ ] 5.6 整理参考文献
- [ ] 5.7 统一格式、图表编号

## 6. 开发日志与答辩准备
- [ ] 6.1 每人至少 6 条开发日志
- [ ] 6.2 收集小组讨论照片
- [ ] 6.3 整理成员贡献说明
- [ ] 6.4 准备答辩 Q&A 题库
- [ ] 6.5 打包最终提交材料
