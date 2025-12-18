# Tasks: 统一训练配置与预训练策略

## 1. 模型预训练配置

- [ ] 1.1 修改 `src/models/pspnet.py` backbone 使用 `ResNet50_Weights.IMAGENET1K_V1`
- [ ] 1.2 修改 `src/models/deeplabv3plus.py` backbone 使用 `ResNet50_Weights.IMAGENET1K_V1`
- [ ] 1.3 修改 `code/segmentation/train_camvid.py` FCN 使用 `weights_backbone="IMAGENET1K_V1"`

## 2. 学习率调度

- [ ] 2.1 在 `train_camvid.py` 添加 PolyLR scheduler（power=0.9）
- [ ] 2.2 计算 max_iter = epochs × len(train_loader)

## 3. 重新训练与验证

- [ ] 3.1 重新训练 PSPNet（50 epochs）
- [ ] 3.2 重新训练 FCN（50 epochs）
- [ ] 3.3 重新训练 DeepLabV3+（50 epochs）
- [ ] 3.4 确认 PSPNet mIoU > FCN baseline
- [ ] 3.5 更新报告中的实验结果

## 4. 文档更新

- [ ] 4.1 更新 `report/pspnet_report.md` 第四章定量结果
- [ ] 4.2 更新训练曲线图
