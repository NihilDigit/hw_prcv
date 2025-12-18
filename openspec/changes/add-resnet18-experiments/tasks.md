# Tasks: 添加 ResNet18 轻量 backbone 对比实验

## 1. 代码修改

- [ ] 1.1 修改 `pspnet.py` 支持 ResNet18 预训练 + output_stride=32
- [ ] 1.2 修改 `deeplabv3plus.py` 支持 ResNet18 预训练 + output_stride=32
- [ ] 1.3 修改 `train_camvid.py` 添加 `--backbone` 参数
- [ ] 1.4 实现 ResNet18 版本的 FCN（或使用简化方案）

## 2. ResNet18 实验

- [ ] 2.1 训练 PSPNet + ResNet18（50 epochs）
- [ ] 2.2 训练 FCN + ResNet18（50 epochs）
- [ ] 2.3 训练 DeepLabV3+ + ResNet18（50 epochs）

## 3. 结果分析

- [ ] 3.1 对比 ResNet18 vs ResNet50 各模型表现
- [ ] 3.2 分析 PPM/ASPP 在弱 backbone 下的贡献
- [ ] 3.3 更新报告实验结果表格
