# 开发日志：ResNet 论文复现（CIFAR-10）

> 说明：本日志仅记录 ResNet 方向的开发与复现过程，便于提交时作为“过程性材料”。

## 2025-12-18

### 13:19 数据准备

- 下载 CIFAR-10 并保存到 `data/cifar10/`
- 校验：存在 `cifar-10-python.tar.gz` 与 `cifar-10-batches-py/` 目录

### 13:27 模型与训练脚本实现

- 新增 CIFAR-10 dataloader：`src/data/cifar10.py`
- 新增 Plain/ResNet（20/56）实现：`src/models/resnet_cifar.py`
- 新增训练脚本：`code/resnet/train_cifar.py`
- 新增批量复现与绘图脚本：`code/resnet/run_reproduction.sh`、`code/resnet/plot_reproduction.py`

### 14:00 启动四组实验

- run id：`repro-20251218-140032`
- 输出目录：`experiments/resnet/reproduction/repro-20251218-140032/`

### 14:35 训练中断问题定位与处理

- 现象：`torch.compile`/Inductor 在 `resnet20` 阶段触发 Triton 编译错误，训练中断（详见 `nohup.log` 中的 Traceback）
- 处理：后续批量运行默认关闭 compile，确保复现实验完整跑通；复现结论依赖曲线与对照，编译加速不影响结论正确性

### 15:31 四组实验完成与产物整理

- 训练曲线图输出：
  - `report/figures/resnet/repro-20251218-140032/resnet_train_loss.png`
  - `report/figures/resnet/repro-20251218-140032/resnet_test_accuracy.png`
- 汇总信息：
  - `report/figures/resnet/repro-20251218-140032/summary.json`

