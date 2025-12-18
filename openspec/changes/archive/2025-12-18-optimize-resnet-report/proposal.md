# Change: 优化 ResNet 复现报告的学术规范与可视化

## Why

当前报告虽已完成基本框架，但存在以下问题：
1. **可视化不足**：仅有 2 张曲线图，缺少网络结构图、残差块示意图、训练流程图等关键图表
2. **学术风格欠缺**：部分表述存在口语化倾向，需要用更严谨的学术语言重写
3. **与课程要求未完全对齐**：根据 `record.md`，需要更详细的文献综述、更清晰的技术路线、更完整的实验对比

## What Changes

### 可视化增强
- 新增 ResNet BasicBlock 结构示意图（含 shortcut 路径）
- 新增 Plain vs ResNet 网络整体架构对比图
- 新增训练流程图（数据增强 → 前向 → 损失 → 反向 → 更新）
- 新增超参数对比表格（论文设置 vs 本次复现设置）
- 新增各 epoch 训练时间对比图（可选）

### 学术风格重写
- 消除口语化表述，使用规范学术用语
- 补充公式：残差学习公式、损失函数、学习率衰减策略
- 规范引用格式，增加必要的 citation
- 统一术语（如 degradation、shortcut、residual learning）

### 对齐课程要求
- **第一章**：扩展文献综述，补充 AlexNet/VGG/GoogLeNet/BN 的详细原理与网络结构说明
- **第二章**：增加算法伪代码或流程图
- **第三章**：增加超参数调整过程的记录与对比实验
- **第四章**：增加更多定量分析（如收敛速度对比、参数量对比）
- **附件**：补充开发过程照片、小组讨论记录、成员分工说明

## Impact

- Affected specs: `resnet-reproduction`
- Affected code: 
  - `report/resnet_report.md`（主报告）
  - `code/resnet/plot_reproduction.py`（新增图表生成）
  - `src/utils/plotting.py`（可能需要新增绘图工具）
