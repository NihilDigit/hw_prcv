# 深度残差网络图像分类研究 - 附录

## 附录 A：代码结构与复现命令

### A.1 项目结构

```
prcv/
├── src/
│   ├── data/
│   │   └── cifar10.py          # CIFAR-10 数据加载
│   ├── models/
│   │   └── resnet_cifar.py     # Plain/ResNet 模型定义
│   └── utils/
│       ├── plotting.py         # 绘图工具
│       └── training.py         # 训练工具
├── code/resnet/
│   ├── train_cifar.py          # 训练脚本
│   ├── run_reproduction.sh     # 批量复现脚本
│   ├── plot_reproduction.py    # 训练曲线绘制
│   └── plot_report_figures.py  # 报告图表生成
├── experiments/resnet/         # 实验产物
└── report/figures/resnet/      # 报告图表
```

### A.2 环境配置

```bash
# 安装 pixi 包管理器后，自动配置环境
pixi install
```

### A.3 数据准备

```bash
pixi run python code/resnet/download_cifar10.py --root data/cifar10
```

### A.4 训练单个模型

```bash
pixi run python code/resnet/train_cifar.py \
    --model resnet56 \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.1 \
    --weight-decay 1e-4
```

### A.5 完整复现实验

```bash
pixi run bash code/resnet/run_reproduction.sh repro-20251218-140032
```

### A.6 生成报告图表

```bash
pixi run python code/resnet/plot_report_figures.py
pixi run python code/resnet/plot_reproduction.py \
    --run-dir experiments/resnet/reproduction/repro-20251218-140032
```

## 附录 B：开发日志

### 2025-12-18

| 时间 | 工作内容 |
|:---|:---|
| 13:19 | 下载 CIFAR-10 数据集至 `data/cifar10/`，验证数据完整性 |
| 13:27 | 完成模型实现（`resnet_cifar.py`）、数据加载（`cifar10.py`）、训练脚本（`train_cifar.py`） |
| 14:00 | 启动四组对照实验（run id: `repro-20251218-140032`） |
| 14:35 | 发现 `torch.compile` 与 Triton 兼容性问题，关闭编译加速以保证实验稳定性 |
| 15:31 | 四组实验全部完成，生成训练曲线图和汇总报告 |
| 16:00 | 撰写复现报告初稿 |

### 问题与解决

**问题**：`torch.compile` 在 `resnet20` 训练阶段触发 Triton 编译错误

**原因**：Triton 编译器对特定 CUDA 环境的兼容性问题

**解决**：批量运行时默认关闭 `torch.compile`（`--no-compile` 参数）。该处理不影响复现结论的正确性，因为编译加速仅影响训练速度，不改变模型的数学行为。

## 附录 C：小组分工说明

（根据实际情况填写各成员的具体工作内容）

| 成员 | 工作内容 | 贡献比例 |
|:---|:---|:---:|
| 成员 A | 模型实现、训练脚本开发 | —% |
| 成员 B | 实验运行、数据分析 | —% |
| 成员 C | 报告撰写、可视化 | —% |
| 成员 D | 文献调研、答辩准备 | —% |

## 附录 D：原版论文

原版论文 PDF 请见随附文件：

- `references/He2016_ResNet_CVPR.pdf`
- `references/Ioffe2015_BatchNorm.pdf`

