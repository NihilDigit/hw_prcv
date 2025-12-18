# ResNet 论文复现（CIFAR-10）说明

本目录包含对 He et al. 2016《Deep Residual Learning for Image Recognition》在 **CIFAR-10** 上的核心对照实验复现：对比 **Plain 网络**与 **ResNet** 在不同深度下的训练难度与精度表现，复现论文中提到的 degradation 现象以及残差连接带来的优化改善。

相关论文与报告：
- 原论文：`references/He2016_ResNet_CVPR.pdf`
- 项目报告：`report/resnet_report.md`
- 附录与复现命令：`report/resnet_appendix.md`

---

## 复现做了什么

### 复现实验目标
- 在 CIFAR-10 上训练 4 组模型并对比曲线：
  - `plain20`、`plain56`（无 shortcut 的 Plain 网络）
  - `resnet20`、`resnet56`（带 shortcut 的 ResNet）
- 观察并验证：
  - 深层 Plain 网络更难优化/出现性能退化（degradation）
  - 残差连接可显著缓解优化困难

### 模型实现（CIFAR 版本 BasicBlock）
- 代码位置：`src/models/resnet_cifar.py`
- 结构要点：
  - stem：`3×3 conv + BN + ReLU`（输出通道 16）
  - 3 个 stage：通道数 16/32/64，stage 之间 stride=2 下采样
  - BasicBlock：`3×3 conv + BN + ReLU -> 3×3 conv + BN -> (+ shortcut) -> ReLU`
- shortcut 类型：
  - `A`：下采样（步幅采样）+ 通道零填充（CIFAR 论文常用）
  - `B`：`1×1 conv` 投影 + BN

### 训练设置（与复现实验一致）
- 代码位置：`code/resnet/train_cifar.py`
- 默认超参：
  - epochs：200
  - optimizer：SGD（momentum=0.9，weight_decay=1e-4）
  - lr：0.1
  - scheduler：MultiStepLR milestones=[100, 150]，gamma=0.1
- 数据增强与归一化：
  - train：RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
  - test：Normalize
  - 代码位置：`src/data/cifar10.py`

---

## 数据集在哪里

本复现使用 CIFAR-10，默认数据目录为：`data/cifar10/`

注意：`src/data/cifar10.py` 使用 `torchvision.datasets.CIFAR10(..., download=False)`，因此需要先下载数据（或保证该目录下已存在 CIFAR-10 文件）。

下载命令：
```bash
pixi run python code/resnet/download_cifar10.py --root data/cifar10
```

---

## 如何重跑（推荐流程）

### 1) 安装/进入环境（pixi）
项目使用 `pixi` 管理环境：
```bash
pixi install
```

### 2) 准备数据
```bash
pixi run python code/resnet/download_cifar10.py --root data/cifar10
```

### 3) 运行单个模型训练
示例：训练 `resnet56`（输出到默认目录 `experiments/resnet/resnet56/<timestamp>/`）
```bash
pixi run python code/resnet/train_cifar.py \
  --model resnet56 \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.1 \
  --weight-decay 1e-4
```

常用参数：
- `--data-root data/cifar10`：数据目录（默认就是它）
- `--shortcut-type A|B`：ResNet shortcut 类型（默认 A）
- `--no-compile`：禁用 `torch.compile()`（更稳定）
- `--no-amp`：禁用 AMP（GPU 上一般不建议关，除非遇到数值/兼容性问题）
- `--out-dir <path>`：指定输出目录
- `--seed 42`：随机种子（注意本项目默认启用 cuDNN benchmark，严格可复现性不是目标）

### 4) 一键跑完整对照实验（4 个模型 + 出图）
脚本会按顺序训练 `plain20/plain56/resnet20/resnet56`，并生成对比曲线图与 summary：
```bash
pixi run bash code/resnet/run_reproduction.sh repro-$(date +%Y%m%d-%H%M%S)
```

脚本支持用环境变量快速改参数（不改脚本）：
```bash
EPOCHS=200 BATCH_SIZE=128 NUM_WORKERS=4 SHORTCUT_TYPE=A pixi run bash code/resnet/run_reproduction.sh my-run
```

默认为了稳定性 **关闭** `torch.compile`（脚本内 `NO_COMPILE=1`）。如果你明确想启用：
```bash
NO_COMPILE=0 pixi run bash code/resnet/run_reproduction.sh my-run
```

---

## 输出产物在哪、长什么样

### 单模型训练输出
默认输出目录：
- `experiments/resnet/<model>/<timestamp>/`

包含文件：
- `config.json`：训练配置（超参、是否 compile/amp 等）
- `metrics.json`：每个 epoch 的训练/测试指标（loss/acc/lr/time）
- `best.pth`：验证集（这里是 test set）精度最好的 checkpoint

### 一键复现实验输出
运行 `run_reproduction.sh` 后：
- 训练日志（4 个模型）：`experiments/resnet/reproduction/<run_id>/<model>/...`
- 曲线与汇总（默认输出到报告目录）：
  - `report/figures/resnet/<run_id>/resnet_train_loss.png`
  - `report/figures/resnet/<run_id>/resnet_test_accuracy.png`
  - `report/figures/resnet/<run_id>/summary.json`

也可以单独对某次 run 画图（当你已有 `metrics.json` 时）：
```bash
pixi run python code/resnet/plot_reproduction.py --run-dir experiments/resnet/reproduction/<run_id>
```

---

## 快速自检（不想跑 200 epoch）

用于验证代码/环境可跑通（速度快）：
```bash
pixi run python code/resnet/train_cifar.py \
  --model resnet20 \
  --epochs 1 \
  --max-steps-train 10 \
  --max-steps-test 10 \
  --no-compile
```

---

## 常见问题（Troubleshooting）

### 1) 找不到数据 / 报 CIFAR-10 文件不存在
原因：数据加载是 `download=False`，不会自动下载。

解决：先运行下载脚本，并确认 `data/cifar10/` 下有 `cifar-10-batches-py/`：
```bash
pixi run python code/resnet/download_cifar10.py --root data/cifar10
```

### 2) `torch.compile`/Triton 报错
现象：在某些 CUDA/Triton 组合下，`torch.compile` 可能触发编译错误。

解决：
- 复现脚本默认已禁用 compile（更稳定）
- 单跑时可显式加 `--no-compile`
- 如果你想启用 compile，请确保系统里可找到 CUDA 头文件 `cuda.h`（脚本会做检测并在缺失时自动降级）

---

## 相关代码入口（速查）

- 下载 CIFAR-10：`code/resnet/download_cifar10.py`
- 训练脚本：`code/resnet/train_cifar.py`
- 一键复现：`code/resnet/run_reproduction.sh`
- 训练曲线绘制：`code/resnet/plot_reproduction.py`
- 报告图表生成：`code/resnet/plot_report_figures.py`（生成报告用结构图/柱状图等）
- 数据加载：`src/data/cifar10.py`
- 模型定义：`src/models/resnet_cifar.py`

