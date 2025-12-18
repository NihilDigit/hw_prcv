# 论文复现报告：Deep Residual Learning for Image Recognition（ResNet, CVPR 2016）

> 说明：本文件为 Markdown 源文件，后续可用 `pandoc` 转为 Word/PDF。  
> 复现目标：在 CIFAR-10 上完成 **Plain-20/56 vs ResNet-20/56** 对照实验，复现 **degradation** 现象，并解释 residual connection 缓解优化困难的原因。  
> 原版论文与相关参考文献请随最终提交材料一并提供（要求权威原文）。

---

# 目 录

- 第一章 背景介绍  
- 第二章 算法原理介绍  
- 第三章 程序设计与实现（训练过程，超参数的调整精度的对比）  
- 第四章 程序测试  
- 第五章 结论  
- 参考文献  
- 附件  
- 开发日志  

---

# 第一章 背景介绍

## 1.1 问题背景：深度越深一定越好吗？

深度学习在视觉任务中的成功很大程度上来自“深”：更深的网络能够学习更复杂、层次更丰富的特征表示。然而，工程实践会遇到一个非常现实的问题：当我们把网络堆得更深，训练会变得更难、收敛更慢，甚至出现“训练误差不降反升”的反直觉现象。也就是说，深度带来的**表达能力提升**并不必然转化为**可优化性提升**。

因此，深层网络的关键不仅是“结构更复杂”，更是“如何让它能被优化算法稳定地训练出来”。这就是 ResNet 这类结构的核心动机：通过结构设计改变优化难度，使更深的网络能够被训练，并能从深度中获得收益。

## 1.2 文献综述：从早期 CNN 到 ResNet

为了理解 ResNet 解决的问题，需要把图像分类网络的发展脉络串起来。下述节点并非罗列名词，而是解释“为什么 ResNet 必然出现”：

### 1.2.1 AlexNet：大规模视觉任务上的深度 CNN

AlexNet 在 ImageNet 上证明了深层 CNN 的有效性，并让 GPU 训练、ReLU、Dropout、数据增强等训练实践成为现代视觉网络的基础。其结论是：更深/更大的模型在大数据上能带来显著提升，但训练稳定性与工程细节同样关键。

### 1.2.2 VGG：用更深的堆叠验证深度的价值

VGG 通过重复堆叠小卷积核（3×3）构建更深网络，并在结构上保持相对“朴素”。它说明了“在同类 CNN 框架下，提高深度可以提升性能”，也推动了后续研究对更深网络的追求。

### 1.2.3 GoogLeNet/Inception：结构工程化与计算效率

GoogLeNet（Inception）通过多分支结构在控制计算量的前提下提升性能，强调“结构设计”与“计算效率”的权衡。这表明提升性能不只有“堆深度”一条路，还可以通过更合理的结构组合提升表示能力。

### 1.2.4 Batch Normalization：深层训练稳定性的重要组件

BN 通过对中间特征进行归一化，显著改善深层网络的训练稳定性，加速收敛并提升可训练性。即便如此，继续增加深度仍会遇到优化难题，说明“归一化”并不能完全消除深层优化困难。

### 1.2.5 Degradation：深层 plain 网络的优化困难

在更深的 plain 网络中，一个关键现象是训练误差可能不降反升（degradation）。理论上更深网络至少可以学到恒等映射而不变差，但实践中优化算法未必能找到这种“至少不差”的解。ResNet 正是针对这一现象提出结构性解决方案。

## 1.3 ResNet 之后的演进

ResNet 提出的 residual connection 成为后续视觉网络的基本组件之一，影响了大量工作。典型方向包括：

- **更强的残差变体与分组卷积**：ResNeXt 等通过在 residual 框架下改造 block 结构提升性能/效率。
- **特征重标定与注意力模块**：SENet 等将通道注意力引入残差网络，增强特征选择能力。
- **更深更稳的训练策略与正则化**：在残差结构基础上，通过更合理的归一化、初始化、学习率策略获得更稳定的深层训练。
- **从 CNN 到 Transformer 的架构迁移**：即便在 ViT 等 Transformer 框架下，残差连接仍是基础结构之一，说明其价值不局限于 CNN。

这些工作共同说明：ResNet 解决的是一个具有普适性的训练与优化问题（深层可优化性），因此 residual connection 被广泛继承和复用。

## 1.4 本次复现的目标与技术路线

本次复现的“灵魂目标”不在于追求某个极限精度，而在于做出论文核心论断所需要的证据链：

1. **对照实验设计**：在 CIFAR-10 上，统一数据增强、训练轮数、优化器与学习率策略，只改变网络结构（plain vs residual）与深度（20 vs 56）。
2. **复现 degradation**：验证更深的 plain 网络（Plain-56）在训练/测试上劣于 Plain-20。
3. **验证 residual 的改善**：对比 ResNet-20/56，展示更深的 ResNet 仍能稳定优化并获得更高测试精度。
4. **输出可视化证据**：给出 train loss 曲线与 test accuracy 曲线，并进行解释。

---

# 第二章 算法原理介绍

## 2.1 Plain 网络的基本结构（CIFAR-10 经典配置）

在 CIFAR-10 的经典设置中，网络由若干个 basic block 堆叠构成，每个 block 由卷积、批归一化与非线性激活组成。Plain 网络仅做顺序堆叠：

`x -> Conv-BN-ReLU -> Conv-BN -> ReLU -> ...`

当深度增加时，网络的表达能力提高，但优化难度也随之增加：反向传播需要经过更多层，梯度可能衰减、数值不稳定等问题更容易出现。

## 2.2 Residual Learning：残差学习的核心形式

ResNet 将 block 的输出写为：

`y = F(x) + x`

其中：

- `F(x)` 表示残差分支（通常由 2 个或 3 个卷积层构成）
- `x` 表示 shortcut 分支（identity 或投影）

这一形式的关键并不只是“多了一条相加”，而是改变了学习目标：从直接拟合 `H(x)` 变为拟合 `F(x)=H(x)-x`。当最优映射接近恒等时，`F(x)` 更容易被优化到 0 附近，从而使深层网络更易训练。

## 2.3 维度匹配与 shortcut 方案

当输入输出维度相同，shortcut 使用 identity：`x` 直接相加即可。当发生下采样或通道变化，需要匹配维度。常见方案：

- **Option A（CIFAR 常用）**：对空间维度按 stride 下采样；通道维度通过 zero-padding 补齐。
- **Option B（projection）**：使用 `1×1 conv` 对 shortcut 投影到目标维度。

本项目默认采用 Option A（并保留 `shortcut_type` 参数支持 Option B），以贴近 CIFAR 的经典复现设置。

## 2.4 为什么 residual 能缓解优化困难（答辩高频要点）

从优化视角给出可讲清楚的解释：

1. **恒等映射更易实现**：当 `F(x)=0` 时，block 输出即为 `x`，网络很容易“至少不变差”；而 plain 网络要通过多层卷积去近似恒等映射，未必容易到达。
2. **梯度通路更短**：反向传播时梯度可沿 shortcut 直接回传到浅层，缓解深层链式相乘导致的梯度衰减，使深层网络更稳定可训。
3. **更平滑/更易优化的参数化方式**：残差参数化改变了误差面形态，使优化算法更容易找到低训练误差的解。

---

# 第三章 程序设计与实现（训练过程，超参数的调整精度的对比）

## 3.1 实验环境说明

本次复现硬件与软件环境如下（表 3-1）：

| 项目 | 说明 |
|---|---|
| OS | Linux 6.17.9-arch1-1（x86_64） |
| CPU | Intel i7-12650H |
| RAM | 24GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU（8GB VRAM） |
| Python | 3.12.12（pixi） |
| PyTorch | 2.9.1 |
| CUDA（torch.version.cuda） | 12.9 |
| cuDNN | enabled（version 91002） |
| torchvision | 0.24.*（pixi 环境） |
| 包管理 | pixi |

## 3.2 数据准备与加载

### 3.2.1 数据集

- CIFAR-10：训练集 50000，测试集 10000（标准划分）

### 3.2.2 预处理与数据增强（控制变量）

为了保证对照实验公平，四组实验使用完全一致的数据处理流程：

- 训练增强：RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
- 测试增强：ToTensor + Normalize

对应实现：`src/data/cifar10.py`

## 3.3 模型实现

模型实现对应文件：`src/models/resnet_cifar.py`，包含：

- PlainCifarNet：`cifar_plain(depth=20/56)`
- ResNetCifar：`cifar_resnet(depth=20/56, shortcut_type="A"|"B")`

为对齐论文“plain vs residual”对照，本项目的 plain 网络与 residual 网络使用相同的 stem、stage 划分与通道配置，仅在 block 内是否相加 shortcut 上做差异。

## 3.4 训练流程与日志记录

训练脚本：`code/resnet/train_cifar.py`

流程包含：

1. 构建 dataloader（训练增强/测试预处理一致）
2. 构建模型并放置到 GPU
3. 训练循环：forward -> loss -> backward -> optimizer step
4. 测试循环：每 epoch 评测 test loss 与 test acc
5. 日志与产物：
   - `metrics.json`：每 epoch 的 train loss/train acc/test loss/test acc/lr/time
   - `best.pth`：按 best test acc 保存的 checkpoint
   - `config.json`：本次运行参数

## 3.5 训练超参数与对比（精度的对比）

### 3.5.1 主对照设置（本次复现采用）

为保证公平对照，四组实验统一使用：

- Optimizer：SGD（momentum=0.9）
- LR：0.1
- LR schedule：MultiStepLR（epoch 100/150 衰减为 0.01/0.001）
- Weight decay：1e-4
- Epochs：200
- Batch size：128

上述配置的选择理由是：在 CIFAR-10 上，这是被大量复现工作采用的经典训练组合；同时它能有效保证对照实验的公平性，使得性能差异主要来自网络结构（plain vs residual）与深度（20 vs 56）。

---

# 第四章 程序测试

## 4.1 测试设计：控制变量对照

本次测试以对照实验为核心：

- 深度对照：20 层 vs 56 层
- 结构对照：plain vs residual
- 控制变量：数据集、增强、优化器、学习率策略、训练轮数、batch size 全部一致

因此测试结果能够将性能差异更直接地归因于“深度与残差结构”。

## 4.2 实验运行与产物位置

本次完整复现实验 run id：

- `repro-20251218-140032`

训练产物目录：

- `experiments/resnet/reproduction/repro-20251218-140032/`

曲线图与汇总目录：

- `report/figures/resnet/repro-20251218-140032/`

## 4.3 曲线图（必须项）

训练 loss 曲线（Plain vs ResNet）：

![CIFAR-10 Train Loss](figures/resnet/repro-20251218-140032/resnet_train_loss.png)

测试 accuracy 曲线（Plain vs ResNet）：

![CIFAR-10 Test Accuracy](figures/resnet/repro-20251218-140032/resnet_test_accuracy.png)

## 4.4 定量结果汇总（核心表格）

以 200 epochs 内的 best test accuracy 作为对比指标：

| Model | Best Test Acc |
|---|---:|
| Plain-20 | 0.9127 |
| Plain-56 | 0.8769 |
| ResNet-20 | 0.9144 |
| ResNet-56 | 0.9358 |

补充：最后一个 epoch（Epoch 200）的 test acc：

| Model | Test Acc @ Epoch 200 |
|---|---:|
| Plain-20 | 0.9116 |
| Plain-56 | 0.8723 |
| ResNet-20 | 0.9126 |
| ResNet-56 | 0.9343 |

（以上数值来自 `report/figures/resnet/repro-20251218-140032/summary.json`）

## 4.5 结果分析：degradation 与 residual 的证据链

### 4.5.1 Degradation 是否出现？

本次实验明确观察到 degradation：

- **Plain-56 的测试精度显著低于 Plain-20**（best 0.8769 < 0.9127）。
- 训练曲线也显示 Plain-56 的训练 loss 长期更高，说明其性能下降更符合“优化困难”而非“过拟合”。

### 4.5.2 Residual connection 的改善

残差连接的效果同样清晰：

- **ResNet-56 显著优于 Plain-56**（best 0.9358 > 0.8769），说明 shortcut 显著改善深层网络的可训练性与最终性能。
- **深层 ResNet 能从加深中获益**：ResNet-56 优于 ResNet-20（0.9358 > 0.9144），而 deep plain 则相反（Plain-56 低于 Plain-20）。

### 4.5.3 关键结论的证据指向

为了便于将结论与证据对应起来：

- “degradation 的证据” → `resnet_train_loss.png`（Plain-56 训练 loss 更高）+ `resnet_test_accuracy.png`（Plain-56 测试精度更低）
- “residual 为什么有效” → 第 2.4 节的优化解释 + ResNet-56 曲线收敛更稳、精度更高

---

# 第五章 结论

1. 本次复现完成了 CIFAR-10 上 Plain-20/56 与 ResNet-20/56 的对照实验，并输出 train loss 与 test accuracy 曲线。
2. 实验复现了论文核心现象：深层 plain 网络出现 degradation（训练更难、测试更差），而残差连接显著改善深层网络可训练性与最终精度。
3. 复现的价值在于：通过控制变量对照，把性能差异更直接地归因于“残差连接改善优化”的机制，而不仅是“跑出一个数字”。

---

# 参考文献

1. He, K., Zhang, X., Ren, S., Sun, J. Deep Residual Learning for Image Recognition. CVPR, 2016.
2. Krizhevsky, A. Learning Multiple Layers of Features from Tiny Images. Technical Report, University of Toronto, 2009.（CIFAR-10）
3. Krizhevsky, A., Sutskever, I., Hinton, G. E. ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS, 2012.（AlexNet）
4. Simonyan, K., Zisserman, A. Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR, 2015.（VGG）
5. Szegedy, C., et al. Going Deeper with Convolutions. CVPR, 2015.（GoogLeNet / Inception v1）
6. Ioffe, S., Szegedy, C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML, 2015.（BN）
7. Xie, S., Girshick, R., Dollár, P., Tu, Z., He, K. Aggregated Residual Transformations for Deep Neural Networks. CVPR, 2017.（ResNeXt）
8. Hu, J., Shen, L., Sun, G. Squeeze-and-Excitation Networks. CVPR, 2018.（SENet）

---

# 附件

## 附件 A：复现实验复跑命令

### A.1 下载 CIFAR-10

```bash
pixi run python code/resnet/download_cifar10.py --root data/cifar10
```

### A.2 训练单个模型（示例）

```bash
pixi run python code/resnet/train_cifar.py --model resnet56 --epochs 200 --batch-size 128
```

### A.3 四组对照实验（推荐）

```bash
pixi run bash code/resnet/run_reproduction.sh repro-20251218-140032
```

## 附件 B：复现产物目录说明

```text
experiments/resnet/reproduction/repro-20251218-140032/
  plain20/  plain56/  resnet20/  resnet56/
    config.json
    metrics.json
    best.pth

report/figures/resnet/repro-20251218-140032/
  resnet_train_loss.png
  resnet_test_accuracy.png
  summary.json
```

---

# 开发日志

本节记录 ResNet 方向的关键开发与复现过程（以时间线形式给出），对应的代码、日志与产物均可在仓库中定位。

## 2025-12-18

1. 13:19 — 准备 CIFAR-10 数据集  
   - 下载并落盘到 `data/cifar10/`
   - 验证目录包含 `cifar-10-python.tar.gz` 与解压后的 `cifar-10-batches-py/`

2. 13:27 — 完成 ResNet 复现代码骨架与训练脚本  
   - 模型：`src/models/resnet_cifar.py`（Plain/ResNet，20/56 depth；shortcut A/B）
   - 数据加载：`src/data/cifar10.py`（增强与归一化）
   - 训练脚本：`code/resnet/train_cifar.py`（metrics/ckpt/config 输出）

3. 14:00 — 启动四组复现运行（run id：`repro-20251218-140032`）  
   - 批量脚本：`code/resnet/run_reproduction.sh`
   - 日志文件：`experiments/resnet/reproduction/repro-20251218-140032/nohup.log`

4. 14:35 — 复现过程中发现 `torch.compile`/Inductor 在特定阶段触发 Triton 编译失败（环境依赖问题）  
   - 现象：`nohup.log` 中出现 `InductorError`，训练在 `resnet20` 阶段中断
   - 处理：对后续批量运行关闭 compile 以保证复现可持续推进（结论不依赖 compile 与否）

5. 15:31 — 四组实验全部完成并生成曲线图  
   - 训练目录：`experiments/resnet/reproduction/repro-20251218-140032/`
   - 曲线图：`report/figures/resnet/repro-20251218-140032/resnet_train_loss.png`、`report/figures/resnet/repro-20251218-140032/resnet_test_accuracy.png`
   - 汇总：`report/figures/resnet/repro-20251218-140032/summary.json`

