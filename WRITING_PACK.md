# 项目总写作包（Markdown 母版｜偏啰嗦版）

**适用你们组题目：**

* 课程设计：**基于 PSPNet 的城市街景语义分割（Cityscapes）**
* 论文复现：**Deep Residual Learning for Image Recognition（ResNet, CVPR 2016）**

> 这份文档的目标：把“老师要什么、我们写什么、我们怎么跑实验、怎么分工、怎么答辩”一次性落地。你们写报告时，直接从这里复制每章骨架，把【待填】补上即可。

---

## 0. 你们这门课的“明牌得分点”（写作时要反复对齐）

根据你给的录音内容，老师关注点非常明确（用人话总结）：

1. **你不是“抄论文”，你是在“复现 + 补全 + 对比 + 讲明白”。**
   原论文（尤其是顶会论文）经常综述很短、细节省略很多；你们报告必须把课堂讲过的“前史”和“后续发展”写清楚，并且把自己的实验路线讲清楚。

2. **必须用权威论文原版 + 尽量选有公开实现的经典论文。**
   你们选的 ResNet（CVPR 2016）是非常典型的“权威 + 可复现”。([cv-foundation.org][1])
   PSPNet（CVPR 2017）同样是权威且有公开代码。([Open Access][2])

3. **别只做一个模型：同数据集至少做 2–3 个模型对比 = 直接加分。**
   尤其语义分割，PSPNet 作为主模型，建议至少再加一个 FCN（经典基线）([cv-foundation.org][3])，再加一个 DeepLabv3 或 DeepLabv3+（强对照）。DeepLabv3+ 在 Cityscapes 上有很强基准表现。([ECVA][4])

4. **答辩分很重，而且会点人：三个人都得能回答。**
   所以分工必须“每个人有自己负责的一块内容 + 必会题库”。

---

## 1. 你们最终要交的东西清单（按老师检查习惯写）

### 1.1 必交文件（建议你们最终打包成一个压缩包）

* 【报告正文】`report.pdf`（或 Word 转 PDF）
* 【原版论文】

  * PSPNet CVPR 2017 PDF ([Open Access][2])
  * ResNet CVPR 2016 PDF ([cv-foundation.org][1])
  * Cityscapes CVPR 2016 数据集论文（背景用）([cv-foundation.org][5])
  * FCN CVPR 2015（分割综述/基线用）([cv-foundation.org][3])
  * DeepLabv3（可选，作为对照路线：atrous/ASPP）([Florian Schroff][6])
  * DeepLabv3+ ECCV 2018（可选/推荐）([ECVA][4])
* 【代码与运行说明】`code/ + README.md`
* 【实验记录】训练日志、曲线图、关键配置（建议 `experiments/` 目录）
* 【开发日志】含小组讨论照片、过程截图、时间线（老师点名要）
* 【成员贡献说明】三个人各自贡献要“可交付物化”（后面给模板）

### 1.2 图片/表格硬要求（你录音里老师明确说）

* 图片表格**美观清楚**、有**图注/表注**
* **不能有水印**
* 全文**前后格式一致**（字体、编号、引用风格、图表命名）

---

## 2. 推荐目录结构（你们照着建文件夹就不乱）

```
ProjectRoot/
  report/
    report.md（可选：你们先在md写，再转Word）
    report.docx
    report.pdf
    figures/（报告用图，全部无水印）
    tables/（可选：导出的表）
    references.bib（可选）
  paper/
    PSPNet_CVPR2017.pdf
    ResNet_CVPR2016.pdf
    Cityscapes_CVPR2016.pdf
    FCN_CVPR2015.pdf
    DeepLabv3.pdf（可选）
    DeepLabv3+_ECCV2018.pdf（可选）
  code/
    segmentation/
    resnet_reproduce/
    README.md（如何复现实验：环境+命令+预期输出）
  experiments/
    seg_pspnet/
      configs/
      logs/
      curves/
      predictions/
    seg_baseline_xxx/
    resnet_plain_vs_res/
  devlog/
    devlog_memberA.md
    devlog_memberB.md
    devlog_memberC.md
    meeting_photos/
    screenshots/
```

---

# 写作包 A：课程设计——PSPNet 城市街景语义分割（Cityscapes）

## A0. 你们要在报告里“说清楚”的一句话版本

> 我们在 Cityscapes 城市街景数据集上复现 PSPNet 的语义分割流程，并与至少 1–2 个经典/强基线（如 FCN、DeepLabv3/v3+）在相同数据、相同评测标准下进行对比，给出定量指标（mIoU 等）+ 定性可视化 + 失败案例分析 + 速度/参数量（可选），并解释差异原因。

### A0.1 为什么 PSPNet 合理（可直接写进第一章）

PSPNet 提出 pyramid pooling module 来融合不同尺度的上下文信息，并在多个分割基准上取得很强表现；论文中明确给出单模型在 Cityscapes 上的 mIoU 结果。([Open Access][2])

---

## A1. 第一章 背景介绍（你们要“写全”的综述模板）

> 这一章很关键：老师明确说原论文综述短，但你们要补齐。

### A1.1 任务定义（建议 1–2 段）

* 语义分割：对图像每个像素赋予类别标签（道路、建筑、行人等）。
* 与检测/分类区别：输出是像素级 mask，定位要求更严格。

### A1.2 数据集：Cityscapes（必须写权威信息）

Cityscapes 是面向城市街景的语义理解数据集与评测基准，来自 50 个城市，提供高质量像素标注与大量粗标注；论文与官网都明确给出规模与用途。([cv-foundation.org][7])
你们报告可写（照抄结构，数值自己别改）：

* 精细标注（fine）：共 5000 张（train/val/test 分配详见论文/脚本说明）([cv-foundation.org][7])
* 粗标注（coarse）：额外约 20000 张，用于弱监督/半监督等([cv-foundation.org][7])
* 评测：官方提供评测脚本（建议你们至少在 val 上用官方脚本算 mIoU）。([GitHub][8])

### A1.3 指标（建议写“mIoU + 像素准确率 + 速度”）

* **mIoU（mean Intersection over Union）**：分割最常用指标（重点）
* Pixel Acc / mAcc：可选
* FPS / 推理时间：加分项（不是必须，但好写）

### A1.4 文献综述写作路线（建议按“时间线 + 关键思想”）

**（1）从 FCN 起步：分割深度学习的起点**
FCN 把分类网络改为全卷积结构，结合 skip 连接实现密集预测，是分割经典基线与“综述起点”。([cv-foundation.org][3])

**（2）多尺度上下文路线：DeepLab / PSPNet**

* DeepLabv3 核心点：atrous（空洞）卷积控制感受野，ASPP 捕获多尺度上下文；作者也分享了实现经验。([Florian Schroff][6])
* PSPNet 核心点：pyramid pooling module 聚合不同区域尺度上下文，提升场景解析质量。([Open Access][2])
* DeepLabv3+ 核心点：在 DeepLabv3 基础上加入 decoder，使边界更清晰；论文也报告了 Cityscapes 上的强结果。([ECVA][4])

> 你们写综述时的“老师喜欢句式”：
> “FCN 解决了端到端像素预测的基本范式；DeepLab/PSPNet 进一步针对多尺度上下文建模；DeepLabv3+ 又通过 encoder-decoder 强化边界细节恢复。”

### A1.5 你们项目目标与贡献（必须写“我们额外做了什么”）

可直接用下面这段（改【待填】即可）：

* 我们选择 PSPNet 作为主模型，在 Cityscapes【train/val】上训练，并在 val 上用统一评测脚本获得 mIoU。
* 我们额外实现/复现了【基线模型1：FCN】与【基线模型2：DeepLabv3/v3+ 或 UNet】，并在相同数据划分与相近训练策略下进行对比。
* 我们给出定量表格、典型可视化结果，并对失败案例（小目标、遮挡、边界混淆等）进行解释。

---

## A2. 第二章 算法原理介绍（PSPNet 写作骨架）

### A2.1 PSPNet 总体结构（建议配一张结构图）

* Backbone：通常是 ResNet 系列（你们用哪个写哪个）
* Pyramid Pooling Module：不同尺度池化 → 上采样 → concat → 融合
* 分类头：输出每个像素类别

PSPNet 论文明确描述了其通过 pyramid pooling 聚合全局上下文的动机与效果。([Open Access][2])

### A2.2 Pyramid Pooling Module（建议写到“可复现”的程度）

你们要写清楚：

* 为什么要多尺度：城市街景里“路面大区域 + 行人小区域”同时存在
* 多尺度池化怎么做：不同 bin 的 pooling 得到不同尺度上下文（写概念即可，不必硬背具体数值）
* 为什么 concat 后有效：把全局先验与局部细节合并

### A2.3 对比写法（加分，老师爱问）

用两小段对比 PSPNet vs DeepLab：

* DeepLabv3 用 atrous/ASPP 多尺度采样卷积特征。([Florian Schroff][6])
* PSPNet 用 pyramid pooling 在区域层面聚合上下文。([Open Access][9])
* DeepLabv3+ 额外加入 decoder 改善边界。([ECVA][4])

---

## A3. 第三章 程序设计与实现（老师第二大得分点）

> 原则：写成“工程复现说明”，让别人照你文档也能跑起来。

### A3.1 环境说明（表格必须有）

【表 3-1：实验环境】（模板）

* OS：
* Python：
* 框架：PyTorch / TF（版本）
* CUDA/cuDNN：
* GPU：
* 复现设置：随机种子 seed、是否开启 deterministic、混合精度等

### A3.2 数据准备（Cityscapes）

建议你们写两块：

1. 数据下载与目录结构（别贴长链接，写“按官方/脚本要求组织”）
2. 你们用的 split：train/val（测试集没有公开标注，一般不做 test server 提交）

**建议用官方 cityscapesScripts 做评测**（val 上可直接算 mIoU），脚本与数据结构说明很全。([GitHub][8])

### A3.3 训练流程（必须写）

* 数据预处理：resize/crop、normalize、augmentation
* 优化器与学习率策略：SGD/Adam，warmup、poly/cos 等（你们实际用啥写啥）
* loss：CE + auxiliary loss（如有）
* 保存 best checkpoint 的规则：以 val mIoU 为准（推荐）

### A3.4 超参数对比（至少 2 组）

【表 3-2：超参数与结果对比】（模板）

| 实验编号  | 模型         | lr | batch | crop size | scheduler | aug | 训练轮数/iter | val mIoU | 备注       |
| ----- | ---------- | -: | ----: | --------- | --------- | --- | --------: | -------: | -------- |
| Exp-1 | PSPNet     |    |       |           |           |     |           |          | baseline |
| Exp-2 | PSPNet     |    |       |           |           |     |           |          | 改了xx     |
| Exp-3 | FCN        |    |       |           |           |     |           |          | 基线       |
| Exp-4 | DeepLabv3+ |    |       |           |           |     |           |          | 强对照      |

> 小提醒：你们不需要“把所有超参都调一遍”，但一定要写清楚“为什么调这几个”。

---

## A4. 第四章 程序测试（一定要有：定量+定性+失败分析）

### A4.1 定量对比表（最关键）

【表 4-1：模型对比】（模板）

| 模型              | Backbone | val mIoU | Pixel Acc（可选） | Params（可选） | FPS（可选） |
| --------------- | -------- | -------: | ------------: | ---------: | ------: |
| FCN             |          |          |               |            |         |
| PSPNet          |          |          |               |            |         |
| DeepLabv3 / v3+ |          |          |               |            |         |

> 你们在“结果解释”里一定要写一句：
> “不同模型在 mIoU 上的差异可能来自：多尺度上下文方式不同、decoder 是否恢复边界、感受野与下采样率差异、训练策略与增强差异。”

### A4.2 定性可视化（并排图，老师直观看）

每张图建议做 4 列并排：

* 原图
* GT（真值）
* PSPNet 预测
* Baseline 预测（FCN 或 DeepLab）

### A4.3 失败案例分析（强烈建议写，属于“加分项集中区”）

固定写 3 类失败：

1. 小目标（行人/路牌/细杆）
2. 边界模糊（车与道路边缘、建筑轮廓）
3. 遮挡/反光/夜景（城市街景常见）

每类至少挑 1 张图：

* 说明错在哪里（比如把 sidewalk 预测成 road）
* 解释原因（上下文不足、边界恢复弱、类别相近）
* 给出改进思路（更强 decoder、边界损失、多尺度测试等）

---

## A5. 第五章 结论（模板）

* 本项目完成了 PSPNet 在 Cityscapes 上的训练与验证，获得 val mIoU【待填】。
* 与 FCN 等基线相比，PSPNet 在【待填类别/场景】上提升明显/一般，主要原因可能是【多尺度上下文/全局先验】。([Open Access][2])
* 与 DeepLabv3+ 对比时，若 DeepLabv3+ 边界更清晰，可用其 decoder 设计解释。([Open Access][10])
* 后续工作：更大分辨率 crop、更长训练、更强 backbone、多尺度测试、加入边界相关 loss 等。

---

## A6. 推荐“可交付的权威来源”（你们参考文献里建议出现）

* PSPNet（CVPR 2017）([Open Access][2])
* PSPNet 官方仓库（证明“有可复现实现”）([GitHub][11])
* Cityscapes 数据集论文 + 官网([cv-foundation.org][7])
* cityscapesScripts（评测脚本、数据结构、split 说明）([GitHub][8])
* FCN（CVPR 2015）([cv-foundation.org][3])
* DeepLabv3（atrous/ASPP 路线）([Florian Schroff][6])
* DeepLabv3+（ECCV 2018）([ECVA][4])
* TensorFlow 官方 DeepLab 代码（可选，用于“实现来源”引用）([GitHub][12])

---

# 写作包 B：论文复现——ResNet（CVPR 2016）

## B0. 你们复现的“灵魂目标”（一句话）

> 复现 ResNet 的关键不在“跑到多高精度”，而在：**复现 plain 深层网络的 degradation（越深反而更难训）现象，并展示 residual connection 如何缓解优化困难。**

ResNet 原论文在 CIFAR-10 上用图展示了 20-layer vs 56-layer 的 plain network：更深的 plain 网络训练误差更高，从而测试误差也更高（degradation problem）。([cv-foundation.org][13])

---

## B1. 第一章 背景介绍（分类网络脉络：写到 ResNet 必然出现）

建议结构：

1. 深度网络为什么重要：特征层次更丰富
2. 但更深会更难优化：梯度传播、初始化、训练误差不降反升（degradation）
3. ResNet 的贡献：把“学习映射”改写为“学习残差”，用 shortcut 连接让深层网络更好训。([cv-foundation.org][1])

---

## B2. 第二章 算法原理（必须写清楚 residual block 与 shortcut 方案）

### B2.1 Residual 基本形式（写到能讲出来）

* 核心思想：输出是 `F(x) + x`
* shortcut：同维度时用 identity；维度变化时可用 zero-padding 或 projection（1×1 conv）匹配维度，原文有讨论与示意。([Open Access][14])

### B2.2 为什么它能缓解优化困难（答辩高频题）

你们可以这样讲（建议背熟）：

* plain 网络想直接学习复杂映射 `H(x)`，深了以后更难优化；
* residual 学 `F(x)=H(x)-x`，如果最优接近 identity，那么 `F(x)` 更容易接近 0，从而优化更稳定；
* 实证上，残差网络更容易降低训练误差，并能从更深网络中获得收益。([cv-foundation.org][1])

---

## B3. 第三章 程序设计与实现（强烈建议你们做“对照实验”）

### B3.1 最推荐的实验组合（省力但很对味）

至少做：

* Plain-20 vs ResNet-20（同训练策略）
  再加分做：
* Plain-56 vs ResNet-56（更贴近论文那张图的逻辑）

> 你们不一定要“完全复刻论文所有设置”，但必须做到“控制变量合理”：
> **同数据集、同增强、同优化器与 lr schedule、同训练轮数**，只改结构（plain vs residual）。

### B3.2 环境与训练细节（可以引用原论文实现细节作为参考）

ResNet 论文写了典型 ImageNet 训练实践：SGD、batch size、lr 起始与衰减、weight decay、momentum、BN 放置位置等。你们可引用其作为“我们设置的依据/对齐对象”。([Open Access][14])

### B3.3 代码来源（建议写清楚）

* CVF 原论文 PDF（最权威）([cv-foundation.org][1])
* 官方模型仓库（证明“原作者/原团队提供可用资源”）([GitHub][15])

---

## B4. 第四章 程序测试（必须做曲线图）

你们至少要有两类图（强烈建议放在第四章最显眼位置）：

1. train error / train loss 曲线：plain vs res
2. test error / test acc 曲线：plain vs res

并且在文字里点题：

* “我们观察到：更深的 plain 网络训练误差不降反升，出现 degradation；加入 shortcut 后训练更易优化，测试性能也更好/更稳定。”([cv-foundation.org][13])

---

## B5. 第五章 结论（模板）

* 我们复现了 plain 深层网络的 degradation 现象（【是否出现】），并验证 residual connection 能显著改善优化与泛化（【你们结果】）。([cv-foundation.org][13])
* 与论文数值存在差距的原因可能包括：训练轮数不足、数据增强不同、实现细节差异、随机种子、算力限制。
* 后续：更长训练、更强增强、尝试不同 shortcut 方案（projection vs identity）等。

---

# 3 人小组贡献设计（可直接放进“成员贡献说明”章节）

> 核心原则：**每个人的贡献必须“可交付物化”**，并且**每个人答辩都有主场问题**。

## C1. 三人职责总表（建议就用这个）

| 组员                         | 主责（必须做完）                                      | 协作（可补位）                             | 交付物（写进报告的硬证据）                                                                                        |
| -------------------------- | --------------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **组员A：分割主模型负责人**           | Cityscapes 数据管线 + PSPNet 训练跑通 + 关键调参          | 给 B 的对比实验提供统一设置（crop/aug/scheduler） | ① 数据预处理/增强说明；② PSPNet 配置文件/超参表；③ 训练日志与曲线（loss、mIoU）；④ PSPNet 可视化结果若干                                 |
| **组员B：分割对比评测负责人**          | baseline（FCN + DeepLabv3/v3+ 或 UNet）训练 + 统一评测 | 帮 A 做 PSPNet 对照实验；输出失败案例分析素材        | ① baseline 配置与训练日志；② 统一评测脚本/流程（建议用 cityscapesScripts）([GitHub][8])；③ 模型对比总表（mIoU/FPS/参数量）；④ 失败案例三类素材 |
| **组员C：ResNet 复现负责人 + 总编辑** | ResNet：Plain vs Residual 对照复现 + 曲线图 + 退化现象解释  | 统一报告排版、编号、参考文献、答辩组织                 | ① ResNet 复现代码与配置；② plain vs res 曲线图；③ 关键现象解释（degradation）；④ 最终 report 总编排与答辩 Q&A 题库                  |

> 代码/资源建议：
>
> * PSPNet 官方仓库可作为实现来源之一。([GitHub][11])
> * ResNet 官方模型仓库可作为“资源出处”引用。([GitHub][15])

---

## C2. 章节写作分工（保证“每章有人兜底”）

* **第一章**：C 主写（分类脉络 + ResNet 背景），A/B 提供分割背景补充（Cityscapes 与分割任务）([cv-foundation.org][7])
* **第二章**：A 写 PSPNet 原理([Open Access][2])；C 写 ResNet 原理([cv-foundation.org][13])；B 写 DeepLab/FCN 对比段落([cv-foundation.org][3])
* **第三章**：A 写 PSPNet 实现；B 写 baseline 实现与评测；C 写 ResNet 实现
* **第四章**：B 主写分割结果对比；C 主写 ResNet 曲线与分析；A 补 PSPNet 可视化与关键调参解释
* **第五章**：C 统稿，A/B 各给 3 条“结论 + 后续”要点

---

## C3. 可直接粘贴的“贡献说明”段落模板

* 组员A：负责 Cityscapes 数据处理与 PSPNet 训练复现，完成数据预处理与增强配置、PSPNet 训练脚本与超参数对照实验，并输出训练日志、曲线图与预测可视化结果。
* 组员B：负责分割基线模型（FCN 与【DeepLabv3/v3+ 或 UNet】）训练与统一评测流程，完成模型对比表格、失败案例分析素材，并使用官方评测脚本对 val 结果进行计算与核对。([GitHub][8])
* 组员C：负责 ResNet 论文复现（Plain vs Residual 对照实验），完成训练/测试曲线与 degradation 现象解释，并负责最终报告排版统一、图表编号与参考文献整理，以及答辩材料与题库组织。([cv-foundation.org][13])

---

# 开发日志（老师点名要）——模板（每人一份）

> 建议：每人至少 6–10 条记录，不要最后一天一次补完（痕迹很明显）。

## D1. 单条日志模板（复制即可）

**日期：** 2025-xx-xx（周x）
**耗时：** x 小时
**负责人：** 组员A/B/C
**本次目标：**（一句话）
**完成内容：**

* 1. …
* 2. …
     **关键产出：**（必须能落到文件/截图）
* 文件：`experiments/...`
* 截图：训练曲线 / 控制台输出 / 可视化结果（无水印）
  **遇到的问题：**
* 问题1：…
  **解决方案：**
* …
  **下次计划：**
* …

## D2. 小组讨论记录模板（附照片）

**日期：**
**议题：**（比如：确定 baseline；统一 crop；答辩分工）
**结论：**（3–5 条）
**行动项：**（谁在什么时候前完成什么）
**现场照片：**（1–2 张即可，清晰，无水印）

---

# 答辩 Q&A 题库（建议背熟；老师很爱点这些）

> 规则：每题写“核心回答 2–3 句 + 证据指向（哪张图/哪张表）”。

## E1. PSPNet / 分割方向（组员A、B主答）

1. **PSPNet 的核心创新是什么？**
   答：通过 pyramid pooling module 聚合不同尺度上下文，使像素预测包含全局先验，从而提升场景解析质量。证据：第二章结构图 + PSPNet 论文描述。([Open Access][2])

2. **为什么同样 Cityscapes，不同模型 mIoU 差异很大？**
   答：多尺度上下文方式不同（ASPP vs pyramid pooling）、是否有 decoder 边界恢复、下采样率与感受野、训练策略与增强等差异。证据：表4-1 + 失败案例图。

3. **你们如何保证对比实验公平？**
   答：同 train/val 划分、同预处理/增强、同评测脚本（cityscapesScripts），尽量统一 backbone 与训练轮数，只改变模型结构。证据：表3-2 + 评测脚本说明。([GitHub][8])

4. **为什么 DeepLabv3+ 往往边界更清晰？**
   答：DeepLabv3+ 在 DeepLabv3 基础上加入 decoder 模块逐步恢复空间细节，专门改善边界。证据：第二章对比段 + 你们边界失败案例。([Open Access][10])

## E2. ResNet 方向（组员C主答）

1. **ResNet 解决了什么问题？什么是 degradation？**
   答：深层 plain 网络会出现训练误差反而更高的退化现象（degradation）；ResNet 用 residual learning + shortcut 让深层网络更易优化。证据：你们曲线图 + 论文 CIFAR-10 图示。([cv-foundation.org][13])

2. **shortcut 在维度变化时怎么办？**
   答：同维度用 identity；维度上升可用 zero-padding 或 1×1 projection 匹配维度，原论文明确讨论过。证据：第二章原理 + 论文段落/图。([Open Access][14])

3. **你们复现结果和论文不一致，可能原因？**
   答：训练轮数、增强策略、lr schedule、BN/初始化细节、随机种子与算力差异。证据：表3-1 环境表 + 表3-2 超参表 + 训练曲线。

---

# 最终提交前“自检清单”（照着勾就行）

## F1. 格式与内容

* [ ] 目录、章节编号正确（与要求一致：1~5章 + 参考文献 + 附件 + 开发日志）
* [ ] 图表全部无水印、清晰、带图注/表注
* [ ] 参考文献格式统一（至少包含 PSPNet、ResNet、Cityscapes、FCN；DeepLab 可选）([Open Access][2])
* [ ] 上传了**原版论文 PDF**（不是中文二手材料）

## F2. 实验与对比（最容易丢分/加分的地方）

* [ ] PSPNet 跑通并有 val 评测结果（mIoU）
* [ ] 至少 1 个基线（推荐 FCN）完成并有对比表([cv-foundation.org][3])
* [ ] 最好再加一个强对照（DeepLabv3/v3+），并解释差异([Open Access][10])
* [ ] ResNet 做了 plain vs residual 对照，并输出训练/测试曲线([cv-foundation.org][13])
* [ ] 至少 3 个失败案例分析（分割）+ 文字解释

## F3. 开发过程与贡献说明

* [ ] 三个成员各自开发日志 ≥ 6 条，且含截图/照片
* [ ] 贡献说明具体到“交付物”（脚本/表格/图/日志/统稿）

---

# 参考实现/资源（你们报告里“实现来源”可引用）

* PSPNet 官方代码仓库([GitHub][11])
* Cityscapes 官方脚本（评测与数据结构说明非常全）([GitHub][8])
* ResNet 官方模型仓库（MSRA 发布）([GitHub][15])
* TensorFlow 官方 DeepLab 目录（可选）([GitHub][12])

---

## 你们下一步最省脑的推进方式（按优先级）

1. **先锁定分割对比模型**：FCN（必做）+ DeepLabv3+（推荐）或 DeepLabv3。([cv-foundation.org][3])
2. **先把评测统一**：val 统一用 cityscapesScripts（避免“算出来的 IoU 不可比”）。([GitHub][8])
3. **ResNet 先做 plain-20 vs res-20**：先把“现象”跑出来，再决定要不要加深到 56。([cv-foundation.org][13])


[1]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html?utm_source=chatgpt.com "CVPR 2016 Open Access Repository"
[2]: https://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html?utm_source=chatgpt.com "CVPR 2017 Open Access Repository"
[3]: https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html?utm_source=chatgpt.com "CVPR 2015 Open Access Repository"
[4]: https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.php?utm_source=chatgpt.com "ECVA | European Computer Vision Association"
[5]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.html?utm_source=chatgpt.com "CVPR 2016 Open Access Repository"
[6]: https://www.florian-schroff.de/publications/chen_rethinking_atrous_v3.pdf?utm_source=chatgpt.com "Rethinking Atrous Convolution for Semantic Image Segmentation"
[7]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.pdf?utm_source=chatgpt.com "The Cityscapes Dataset for Semantic Urban Scene Understanding"
[8]: https://github.com/mcordts/cityscapesScripts?utm_source=chatgpt.com "GitHub - mcordts/cityscapesScripts: README and scripts for the Cityscapes Dataset"
[9]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf?utm_source=chatgpt.com "Pyramid Scene Parsing Network"
[10]: https://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf?utm_source=chatgpt.com "Encoder-Decoder with Atrous Separable"
[11]: https://github.com/hszhao/PSPNet?utm_source=chatgpt.com "GitHub - hszhao/PSPNet: Pyramid Scene Parsing Network, CVPR2017."
[12]: https://github.com/tensorflow/models/blob/master/research/deeplab/README.md?utm_source=chatgpt.com "models/research/deeplab/README.md at master · tensorflow/models · GitHub"
[13]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf?utm_source=chatgpt.com "Deep Residual Learning for Image Recognition"
[14]: https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf?utm_source=chatgpt.com "Deep Residual Learning for Image Recognition"
[15]: https://github.com/KaimingHe/deep-residual-networks?utm_source=chatgpt.com "GitHub - KaimingHe/deep-residual-networks: Deep Residual Learning for Image Recognition"
