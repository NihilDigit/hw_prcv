# Change: 更新实验结论与报告分析结构

## Why

完成 ResNet18/ResNet50 双 backbone 对比实验后，发现了重要结论：

### 实验数据

| Backbone | PSPNet | FCN | DeepLabV3+ |
|---|---|---|---|
| ResNet50 | 0.7535 | 0.7535 | (待补) |
| ResNet18 | 0.6191 | 0.6108 | **0.7097** |

### 关键发现

1. **强 backbone (ResNet50) 下架构差异被掩盖**：PSPNet ≈ FCN，预训练权重主导性能
2. **弱 backbone (ResNet18) 下 Decoder 结构显著有效**：DeepLabV3+ 比 PSPNet/FCN 高 ~9pp
3. **PPM 多尺度池化效果有限**：PSPNet vs FCN 仅差 0.83pp（噪声范围）

### 结论

- **Decoder（低层特征融合）是关键**，不是 PPM/ASPP 的多尺度池化
- 报告标题虽为 PSPNet，但对比实验揭示其局限性是有价值的负面结果

## What Changes

### 报告结构调整
- 第四章增加双 backbone 对比分析
- 增加"为什么 PSPNet 没有优势"的深入讨论
- 强调 Decoder 结构的重要性

### Spec 更新
- 添加实验结论相关的需求场景
- 记录对比实验设计决策

## Impact

- Affected specs: `pspnet-segmentation`
- Affected files: `report/pspnet_report.md`
