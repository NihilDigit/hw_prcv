# Tasks: 核实报告数据准确性

## 1. 下载参考文献

- [x] 1.1 下载 ResNet 原论文 (He et al., CVPR 2016) → `references/He2016_ResNet_CVPR.pdf`
- [x] 1.2 下载 AlexNet 论文 (Krizhevsky et al., 2012) → 在线验证完成
- [x] 1.3 下载 VGGNet 论文 (Simonyan & Zisserman, 2014) → 在线验证完成
- [x] 1.4 下载 GoogLeNet/Inception 论文 (Szegedy et al., 2014) → 在线验证完成
- [x] 1.5 下载 Batch Normalization 论文 (Ioffe & Szegedy, 2015) → `references/Ioffe2015_BatchNorm.pdf`

## 2. 核实 ResNet 原论文数据

- [x] 2.1 核实 CIFAR-10 实验设置（epoch、lr schedule、batch size 等）
  - 原论文: 64K iterations (≈200 epochs), lr 0.1 → 0.01 at 32K iter → 0.001 at 48K iter
  - 我们: 200 epochs, lr 0.1 → 0.01 at epoch 100 → 0.001 at epoch 150 ✓ (基本一致)
- [x] 2.2 核实 CIFAR-10 Plain 网络结果（Plain-20, Plain-56 的 error rate）
  - **关键发现**: 原论文 Table 6 **仅包含 ResNet 结果**，Plain 网络数据只在 Figure 1/6 中以曲线展示，无具体数值
  - 已修正 Table 4-2，移除了虚假的 Plain-20 "91.25%" 数据
- [x] 2.3 核实 CIFAR-10 ResNet 结果（ResNet-20, ResNet-56, ResNet-110 的 error rate）
  - ResNet-20: 8.75% error → 91.25% accuracy ✓
  - ResNet-56: 6.97% error → 93.03% accuracy ✓
  - ResNet-110: 6.43% error → 93.57% accuracy (未在报告中使用)
- [x] 2.4 核实 ImageNet 结果（top-5 error rate）
  - ResNet-152 ensemble: 3.57% top-5 error ✓ (报告第33行正确)
- [x] 2.5 核实网络架构细节（层数计算方式、通道数配置）
  - depth = 6n + 2, channels {16, 32, 64} ✓

## 3. 核实其他引用数据

- [x] 3.1 核实 AlexNet 性能数据（top-5 error 15.3%? 16.4%?）
  - **ILSVRC-2012 (7 CNN ensemble): 15.3% top-5 error** ✓ (报告第51行正确)
  - ILSVRC-2010 (single): 17.0% top-5 error
  - 2011年冠军: 26.2% ✓ (报告第51行正确)
- [x] 3.2 核实 VGGNet 性能数据和参数量
  - VGG-16: **138M 参数** ✓ (报告第74行 "1.38亿" 正确)
  - VGG-19: 144M 参数
- [x] 3.3 核实 GoogLeNet 性能数据和参数量（500万参数?）
  - **~5M 参数** (比 AlexNet 少 12x，比 VGG-16 少 ~27x) ✓ (报告第85行正确)
- [x] 3.4 核实 ILSVRC 各年冠军数据
  - 报告中涉及的年份与结论（2012 AlexNet、2014 GoogLeNet、2015 ResNet）已按各自原论文表述核实

## 4. 修正报告

- [x] 4.1 修正摘要中的数值 → 无需修改，实验数据正确
- [x] 4.2 修正第一章文献综述中的数值 → 已验证正确
- [x] 4.3 修正第四章与原论文对比的数值 → **已修正 Table 4-2**
  - 移除了 Plain-20 的虚假"原论文数据"
  - 添加了说明：原论文仅报告 ResNet error rate，Plain 网络无具体数值
- [x] 4.4 更新参考文献列表（确保格式正确） → 无需修改

## 5. 验证

- [x] 5.1 全文检查数值一致性
- [x] 5.2 确认所有引用数据有原文出处

---

## 核实总结

### 发现的问题（已修复）

1. **Table 4-2 中 Plain-20 "原论文报告" = 91.25% 是错误的**
   - 原论文 Table 6 仅包含 ResNet 结果，不包含 Plain 网络
   - 91.25% 实际上是 ResNet-20 的准确率（8.75% error rate）
   - 已修正：将 Plain-20 和 Plain-56 的"原论文报告"列改为 "—"

### 验证通过的数据

| 数据项 | 报告中的值 | 原论文的值 | 状态 |
|--------|-----------|-----------|------|
| AlexNet top-5 error | 15.3% | 15.3% (ILSVRC-2012) | ✓ |
| VGG-16 参数量 | 1.38亿 | 138M | ✓ |
| GoogLeNet 参数量 | 500万 (VGG的1/27) | ~5M (VGG的1/27.6) | ✓ |
| ResNet-20 准确率 | 91.25% | 8.75% error = 91.25% | ✓ |
| ResNet-56 准确率 | 93.03% | 6.97% error = 93.03% | ✓ |
| ResNet-152 top-5 error | 3.57% | 3.57% | ✓ |
| 我们的 ResNet-20 | 91.44% | — | 复现成功 |
| 我们的 ResNet-56 | 93.58% | — | 复现成功 |
