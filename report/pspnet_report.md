# 基于 PSPNet 的 CamVid 语义分割：实现与对照实验分析

## 摘要

语义分割是像素级分类任务，是场景理解与自动驾驶中的基础能力。本报告以 CamVid 城市街景数据集为对象，实现 PSPNet 并完成训练、评测和可视化流程。为评估结构贡献，设置强弱两类骨干并引入 FCN、DeepLabv3+ 作为对照。结果显示，强骨干下 PPM 的增益很小；弱骨干下 decoder 对小目标与边界恢复的作用明显。报告给出实验设置、训练过程、定量结果与可视化分析，并结合失败案例提出改进方向，保证流程可复现、结论可解释。

关键词：语义分割；PSPNet；CamVid；mIoU；对照实验；backbone

# 目录

第一章 背景介绍
第二章 算法原理介绍
第三章 程序设计与实现
第四章 程序测试与结果分析
第五章 结论
参考文献
附录

# 第一章 背景介绍

## 1.1 任务定义与应用

语义分割要对每个像素赋予类别标签，例如道路、建筑、天空、行人等。与图像分类不同，它需要保留空间结构并输出细粒度结果，因此对分辨率、边界细节和上下文信息更敏感。在自动驾驶、机器人导航与智能交通中，它直接影响路径规划、可通行区域判断与场景理解的可靠性。

## 1.2 数据集：CamVid

CamVid[5] 是经典城市场景数据集，提供视频帧与像素级标注。本报告采用 11 类设置，使用常见的 train/val/test 划分：训练 367 张、验证 101 张、测试 233 张。CamVid 场景一致性强，但类别分布不均衡，小目标与细长结构样本稀少。其分辨率较高、类别覆盖较全，适合观察边界与细节恢复能力，同时规模适中，便于在有限算力下完成对照实验。

Road、Sky、Building 等大类像素占比高，模型通常更容易取得高 IoU；而 Pole、SignSymbol 等细长结构像素稀疏，训练信号弱，预测容易断裂或漏检。单一总体指标难以完整反映这些问题，需要结合可视化和逐类表现进行解释，这也有利于比较 PPM 与 decoder 的结构差异。

## 1.3 评测指标：mIoU

本报告采用 mean Intersection over Union 作为主要评测指标。对于类别 c 的 IoU 定义为

$$IoU_c = \frac{TP_c}{TP_c + FP_c + FN_c}$$

其中 TP、FP、FN 分别为该类的真阳性、假阳性与假阴性像素数。mIoU 为各类 IoU 的平均值，忽略未出现类别。与像素准确率相比，mIoU 对类别不均衡更敏感，更能反映小目标与难类别表现。

## 1.4 研究目标与报告结构

报告目标是实现 PSPNet 并形成完整的训练、评测与可视化流程，在统一设置下与 FCN、DeepLabv3+ 对照，评估 PPM 与 decoder 的实际贡献。内容覆盖原理、实现与训练配置、结果分析与可视化、失败案例与改进方向。重点是可复现流程和结构差异的可解释结论。

## 1.5 相关研究概述与本报告定位

语义分割从传统图模型逐步转向端到端深度学习。FCN 将分类网络改造成全卷积结构，推动了像素级预测；随后多尺度上下文成为主流，PSPNet 通过金字塔池化强化全局语义，DeepLab 系列通过空洞卷积与 decoder 提升边界恢复。本报告基于 CamVid 实现 PSPNet，并用对照实验比较不同结构模块在强弱骨干下的作用。

## 1.6 任务难点与评价重点

难点在于同时保证全局语义一致性与边界精度。城市场景既有大区域，也有小目标与细长结构，这些类别像素占比低但影响应用安全。只强调全局容易产生边界模糊；只强调局部容易引入噪声。评估时除总体 mIoU 外，还需关注边界与小目标表现。

# 第二章 算法原理介绍

## 2.1 PSPNet 总体结构

PSPNet[1] 在主干网络的高层语义特征上加入 Pyramid Pooling Module（PPM），通过多尺度池化聚合上下文信息。流程是：backbone 提取特征，PPM 在最高层特征上做多尺度聚合，再经融合与上采样输出像素级预测。相比只依赖局部感受野的结构，PSPNet 更强调全局一致性。

## 2.2 Pyramid Pooling Module

PPM 采用多分辨率池化，将特征图分别做 1×1、2×2、3×3、6×6 等网格池化。各分支用 1×1 卷积压缩通道后上采样回原尺寸，再与原特征拼接并融合。这样在通道维度上引入从全局到局部的上下文信息：大区域类别更稳定，小目标误检与漏检有所缓解。

实现上，各分支通常采用相同的通道压缩比例，控制参数规模；池化特征上采样回原尺寸后再融合，保证后续卷积一致性。PPM 以结构化方式扩大有效感受野，减少仅靠加深网络带来的优化负担。对 CamVid 而言，PPM 更有利于大区域一致性，对细粒度边界的提升有限，这在对照实验中也能看到。

## 2.3 对照模型与结构差异

为评估 PPM 与 decoder 的贡献，引入 FCN[2] 与 DeepLabv3+[3] 对照。FCN 是经典全卷积基线，缺少显式多尺度上下文模块；DeepLabv3+ 在 ASPP 基础上加入 decoder，用低层特征恢复边界细节。因此，PSPNet 与 FCN 的差异主要来自 PPM，DeepLabv3+ 与 PSPNet 的差异主要来自 decoder。统一训练设置有利于对比结构本身的作用。

FCN 结构简单、推理成本较低，但边界模糊和小目标遗漏较多。DeepLabv3+ 通过 ASPP 扩大感受野，再用 decoder 融合低层特征，边界精度通常更好。PSPNet 强调全局上下文一致性，大区域更稳定。将三者放到同一实验框架中，可以避免跨论文条件差异带来的干扰。

## 2.4 语义分割中的结构性权衡

语义分割需要在全局一致性与边界精度之间权衡。只靠高层特征会模糊边界；过度依赖低层特征可能引入噪声。PPM 增强高层语义，decoder 负责细节恢复。本报告通过 CamVid 对照实验判断两类结构在不同条件下的贡献。

## 2.5 多尺度上下文与细节恢复的互补关系

城市街景同时包含大尺度区域和小目标。道路、天空、建筑等更依赖全局上下文；行人、杆状物、交通标志等更依赖边界与纹理。PPM 通过多尺度池化注入全局语义，边界提升有限；decoder 借助低层特征恢复细节，对全局一致性的贡献相对弱。两者在功能上互补。

## 2.6 Backbone 选择与预训练影响

性能高度依赖骨干网络的表达能力。强骨干（如 ResNet-50[4]）提供更稳定的高层语义，结构即便简单也能得到不错结果；弱骨干（如 ResNet-18）语义不足，往往需要结构补偿，例如 decoder 融合低层细节。预训练权重可提升泛化并加快收敛。本报告通过强弱骨干对照，观察骨干强度与结构模块的交互影响。

# 第三章 程序设计与实现

## 3.1 系统实现与代码结构

系统按流程分为四部分：数据管线、模型构建、训练评测与可视化。数据管线负责读取 CamVid 标注并执行训练增强与验证预处理；模型构建提供 PSPNet、FCN、DeepLabv3+ 的统一接口，支持切换 backbone 与输出步幅；训练评测模块覆盖训练循环、验证评测、日志记录与最优权重保存；可视化模块输出训练曲线与预测样例，用于观察收敛与结构差异。整体设计以复现性和对照公平性为优先。

## 3.2 数据预处理与增强

训练阶段使用随机缩放、随机裁剪和随机水平翻转；验证与测试只做固定尺寸 resize，避免引入随机性。Void 类统一映射为 ignore_index=255，在损失与 mIoU 评估中忽略，以免污染统计。

## 3.3 训练配置与优化策略

训练采用带动量的 SGD（动量 0.9、权重衰减 1e-4），损失为 CrossEntropyLoss 并设置 ignore_index。输入裁剪尺寸为 360×480，训练 50 个 epoch。学习率调度采用 Poly 策略，按迭代次数衰减，形式为

$$lr = base\_lr \times (1 - \frac{iter}{max\_iter})^{0.9}$$

Poly 策略在分割任务中收敛稳定，后期学习率逐步降低便于细化更新。骨干使用 ImageNet 预训练权重以提升收敛速度和泛化能力。对照实验统一 batch size 为 4。

训练流程为“每轮训练 + 验证评测”。每个 epoch 结束后在验证集上计算 mIoU 与验证损失，记录学习率与耗时；若 mIoU 刷新最佳则保存 best checkpoint，并保留最后一轮权重用于收敛末期分析。CUDA 环境下启用自动混合精度；模型含辅助分支时，损失为主分支与辅助分支的加权和，以稳定早期梯度。

## 3.4 训练过程中的超参数调整与精度对比

本实验未做大规模网格搜索，只在固定骨干与训练轮数的前提下做小范围验证。学习率固定在 0.01，主要对 batch size 做对比，以兼顾显存占用与优化稳定性。表 3-1 给出 PSPNet 在 ResNet-50 条件下的结果。

表 3-1 PSPNet 超参数对比（ResNet-50，50 epoch）

| 设置 | batch size | 学习率 | Best mIoU (val) | Best epoch |
|:---|---:|---:|---:|---:|
| A | 8 | 0.01 | 0.5539 | 27 |
| B | 4 | 0.01 | 0.7535 | 48 |

结果显示，在当前实现与增强设置下，batch size=4 的验证 mIoU 更高，因此后续对照实验统一采用该配置。

## 3.5 复现实验设置与对照设计

设置强弱两类 backbone：ResNet-50 与 ResNet-18。每类 backbone 下训练 PSPNet、FCN 与 DeepLabv3+ 三个模型，共六组实验。该设计将结构差异与骨干强度分开，便于分析结构模块在不同条件下的作用。训练产物包含 best/last 权重与日志，评测产物包含 mIoU 与可视化样例。

## 3.6 评测与可视化流程

评测流程在验证集上前向推理，计算混淆矩阵并输出 mIoU，同时导出输入图、标注图与预测图。可视化用于观察边界恢复、细长结构与小目标差异。训练曲线由日志生成，展示 loss 与 mIoU 的收敛过程，便于对比稳定性。

## 3.7 训练稳定性与复现性控制

训练结果会受随机初始化、数据加载顺序与 GPU 并行计算影响。为降低波动，所有模型统一训练轮次、学习率调度与数据增强，并采用相同的 batch size 与权重衰减，使差异尽量来自结构本身。训练日志记录关键超参数与运行标识，评测输出保存指标与可视化目录，方便追溯与复现实验。

## 3.8 资源消耗与效率考量

分割模型计算开销较大，高分辨率与多尺度结构尤为明显。为适配算力，输入裁剪为 360×480，batch size 固定为 4，确保单卡可训练。PSPNet 与 DeepLabv3+ 结构更复杂，FCN 较轻量。虽然本报告以精度对比为主，但工程落地仍需考虑速度与显存，后续可补充统一硬件下的推理速度与显存占用。

资源配置会限制模型规模与训练设置。较小的 batch size 会带来更大的梯度噪声，但在显存受限时是必要选择。为保证公平，对所有模型采用相同输入尺寸与 batch size。若提高分辨率或 batch size，性能可能进一步提升，但需要更高的计算资源。

# 第四章 程序测试与结果分析

## 4.1 定量结果与对照分析

汇总两组 backbone 条件下三种模型在 CamVid 验证集的结果。表 4-1 为强 backbone，表 4-2 为弱 backbone。指标均取 best checkpoint，保证比较基于各自最优表现。

各模型的最佳 epoch 并不一致，说明结构对训练动态的响应不同。部分模型在中期达到峰值后进入平台期，表明 50 个 epoch 已接近性能上限。使用 best checkpoint 可减少过拟合或后期波动的影响。

表 4-1 ResNet-50 条件下的 mIoU 结果

| 模型 | Backbone | mIoU (val) | Best epoch | run id |
|:---|:---|---:|---:|:---|
| PSPNet | ResNet-50 | 0.7534 | 48 | seg-v2-pspnet |
| FCN | ResNet-50 | 0.7535 | 50 | seg-v2-fcn |
| DeepLabv3+ | ResNet-50 | 0.7603 | 48 | seg-v2-deeplabv3plus |

ResNet-50 条件下，PSPNet 与 FCN 的 mIoU 几乎一致，PPM 的增益很小。DeepLabv3+ 略高，但优势不大，说明强骨干下结构差异被削弱。

表 4-2 ResNet-18 条件下的 mIoU 结果

| 模型 | Backbone | mIoU (val) | Best epoch | run id |
|:---|:---|---:|---:|:---|
| PSPNet | ResNet-18 | 0.6191 | 40 | seg-r18-pspnet |
| FCN | ResNet-18 | 0.6108 | 38 | seg-r18-fcn |
| DeepLabv3+ | ResNet-18 | 0.7097 | 45 | seg-r18-deeplabv3plus |

ResNet-18 条件下整体性能明显下降，但结构差异被放大。DeepLabv3+ 比 PSPNet 与 FCN 高出接近 9 个百分点，说明弱骨干下 decoder 的低层特征融合贡献更明显。

## 4.2 Backbone 强度影响

对比两类骨干可见性能受骨干强度影响明显：PSPNet 从 0.7534 降至 0.6191，FCN 从 0.7535 降至 0.6108，DeepLabv3+ 从 0.7603 降至 0.7097。DeepLabv3+ 的降幅更小，说明 decoder 在弱骨干条件下能缓解语义不足。在资源受限场景中，这一点尤为关键。

## 4.3 结构贡献分析

在同一 backbone 下比较结构模块：PSPNet 与 FCN 的差异主要来自 PPM，DeepLabv3+ 与 PSPNet 的差异主要来自 decoder。强骨干条件下，PSPNet 与 FCN 差距几乎为零，PPM 的边际收益很小；弱骨干下，PPM 仅提升约 0.0083。相比之下，decoder 在弱骨干下带来 0.0906 的提升，说明低层特征融合更关键。总体上，PPM 更偏全局一致性，decoder 更偏边界与小目标细节。

## 4.4 训练曲线与收敛行为

训练曲线用于观察收敛速度与稳定性。ResNet-50 条件下，PSPNet 损失下降平稳，验证 mIoU 在中后期趋于收敛，反映强骨干与预训练带来更稳定的优化过程。

![PSPNet Loss (ResNet-50)](figures/segmentation/seg-v2-pspnet/camvid_loss.png)

图 4-1 ResNet-50 条件下 PSPNet 训练损失曲线

PSPNet 的 mIoU 后期趋于平台，说明强骨干下结构增益有限。

![PSPNet mIoU (ResNet-50)](figures/segmentation/seg-v2-pspnet/camvid_miou.png)

图 4-2 ResNet-50 条件下 PSPNet 验证集 mIoU 曲线

FCN 在强骨干下的收敛形态与 PSPNet 接近，但提升略小，缺少 PPM 时全局语义增益有限。

![FCN Loss (ResNet-50)](figures/segmentation/seg-v2-fcn/camvid_loss.png)

图 4-3 ResNet-50 条件下 FCN 训练损失曲线

![FCN mIoU (ResNet-50)](figures/segmentation/seg-v2-fcn/camvid_miou.png)

图 4-4 ResNet-50 条件下 FCN 验证集 mIoU 曲线

DeepLabv3+ 在强骨干下曲线与 PSPNet 接近，但后期仍保持轻微优势，decoder 对边界恢复在末期仍有效。

![DeepLabv3+ Loss (ResNet-50)](figures/segmentation/seg-v2-deeplabv3plus/camvid_loss.png)

图 4-5 ResNet-50 条件下 DeepLabv3+ 训练损失曲线

![DeepLabv3+ mIoU (ResNet-50)](figures/segmentation/seg-v2-deeplabv3plus/camvid_miou.png)

图 4-6 ResNet-50 条件下 DeepLabv3+ 验证集 mIoU 曲线

弱骨干条件下，PSPNet 的 mIoU 中期进入平台，骨干表达不足时结构增益难以持续。

![PSPNet Loss (ResNet-18)](figures/segmentation/seg-r18-pspnet/camvid_loss.png)

图 4-7 ResNet-18 条件下 PSPNet 训练损失曲线

![PSPNet mIoU (ResNet-18)](figures/segmentation/seg-r18-pspnet/camvid_miou.png)

图 4-8 ResNet-18 条件下 PSPNet 验证集 mIoU 曲线

FCN 在弱骨干下更早进入平台期，缺少细节恢复结构时性能上限较低。

![FCN Loss (ResNet-18)](figures/segmentation/seg-r18-fcn/camvid_loss.png)

图 4-9 ResNet-18 条件下 FCN 训练损失曲线

![FCN mIoU (ResNet-18)](figures/segmentation/seg-r18-fcn/camvid_miou.png)

图 4-10 ResNet-18 条件下 FCN 验证集 mIoU 曲线

DeepLabv3+ 在弱骨干下仍持续提升，decoder 的低层特征融合对弱骨干更关键。

![DeepLabv3+ Loss (ResNet-18)](figures/segmentation/seg-r18-deeplabv3plus/camvid_loss.png)

图 4-11 ResNet-18 条件下 DeepLabv3+ 训练损失曲线

![DeepLabv3+ mIoU (ResNet-18)](figures/segmentation/seg-r18-deeplabv3plus/camvid_miou.png)

图 4-12 ResNet-18 条件下 DeepLabv3+ 验证集 mIoU 曲线

## 4.5 可视化对比分析

可视化结果能直观反映差异。PSPNet 与 FCN 在道路、天空、建筑等大区域上稳定，但在行人、杆状物体与交通标志等小目标上易漏检或边界模糊。DeepLabv3+ 的边界连续性更好，小目标识别更完整，尤其在道路与人行道的交界处以及细长物体轮廓处更明显，这与 decoder 的低层特征融合一致。

为保证代表性，选取包含道路边界与小目标的样例进行展示，避免只展示易分割场景。

选取验证集样例 000，展示输入、标注及三种模型在 ResNet-50 条件下的预测，重点观察道路与建筑交界处的边界与小目标细节。

| 输入 | 标注 | PSPNet | FCN | DeepLabv3+ |
|---|---|---|---|---|
| ![](figures/segmentation/seg-v2-pspnet/eval_pspnet/vis/000_image.png) | ![](figures/segmentation/seg-v2-pspnet/eval_pspnet/vis/000_gt.png) | ![](figures/segmentation/seg-v2-pspnet/eval_pspnet/vis/000_pred.png) | ![](figures/segmentation/seg-v2-fcn/eval_fcn/vis/000_pred.png) | ![](figures/segmentation/seg-v2-deeplabv3plus/eval_deeplabv3plus/vis/000_pred.png) |

## 4.6 失败案例与误差模式

从逐类 IoU 看，难点集中在小目标与细长结构，如 Pole、SignSymbol 与 Pedestrian。成因包括类别不均衡与分辨率对小目标细节的压缩。弱骨干下 PSPNet 与 FCN 对 Pedestrian 识别不足，DeepLabv3+ 通过 decoder 有明显改善。另一类常见误差是边界模糊，尤其在 Road、Sidewalk 与 Fence 的交界处；光照变化与遮挡也会导致 Tree 与 Building、Sign 等类别混淆。整体来看，边界细节与小目标识别是主要瓶颈。

标注存在一定噪声与不确定性，细线结构或边界模糊区域在不同标注者之间可能有差异，这会降低这些区域的可达上限。尽管如此，带 decoder 的结构在边界与小目标上的优势依然明显。

## 4.7 结果可信度与局限性

训练设置已严格统一，但仍有改进空间。本报告主要基于单次训练结果，未做多随机种子重复实验，统计稳健性有限。小数据集上随机性带来的波动更明显，且 CamVid 规模有限，结论外推到更大数据集仍需验证。即便如此，结构差异的趋势一致，结论仍具参考价值。

## 4.8 结果解释与工程启示

定量与可视化结果给出清晰的工程启示：强骨干下结构复杂化收益有限，性能主要由骨干与预训练决定；弱骨干下结构贡献被放大，尤其是 decoder 的细节恢复作用。应用中若硬件允许使用强骨干，可选择更简化的头部结构；若必须轻量化，则优先考虑具备细节融合能力的结构。

## 4.9 逐类趋势与类别难度分析

虽然主要报告整体 mIoU，但类别难度差异明显。Road、Sky、Building 等大类通常 IoU 高、模型差距小；Pedestrian、Pole、SignSymbol 等小目标更依赖边界精度，结构差异更容易体现。PPM 有助于大区域一致性，但对细长结构提升有限；DeepLabv3+ 通过 decoder 恢复细节，难类别表现更好。因此评估时不应只看总体指标，还要关注难类别与边界质量。

## 4.10 可视化案例的结构性解读

在道路与人行道交界处，PSPNet 更容易形成平滑的大块区域，边界偏模糊；DeepLabv3+ 的边界更连贯且更接近标注。树木与建筑交界处，PSPNet 更易把相近纹理归为同类，DeepLabv3+ 的局部细节更好。对行人、骑行者等小目标，PSPNet 常出现断裂或漏检，DeepLabv3+ 更完整但仍受遮挡和尺度变化影响。这些现象与结构贡献分析一致。

## 4.11 结构选择的实践建议

从 CamVid 结果看，结构选择需结合骨干强度与任务需求。强骨干下，增加复杂模块的收益有限，可能带来训练成本与推理延迟；弱骨干下，结构贡献更明显，尤其是 decoder 对边界与小目标的修复，适合作为轻量化模型的补强方向。若应用强调边界质量，应优先选择具备细节融合能力的结构。
另一方面，PPM 更强调全局一致性，对大区域占比高、噪声较多的场景有价值。实践中可按数据集特性在 PPM 与 decoder 之间权衡，必要时考虑二者结合。

# 第五章 结论

本报告在 CamVid 上实现 PSPNet，并与 FCN、DeepLabv3+ 做统一设置对照。结果表明：强骨干下 PPM 增益很小；弱骨干下 decoder 对小目标与边界恢复效果显著。由此可见，资源受限时低层特征融合更关键；强骨干时结构差异被弱化。本报告给出完整的训练、评测与可视化流程，可直接复现实验并用于结构选择参考。

在方法上，统一设置下的多模型对照避免了把差异误归因于训练技巧或数据处理。定量结果与可视化结合，使结论既有数值依据也有直观证据。

后续可在更大数据集上验证结论，并用多随机种子实验与统计检验提升稳健性。也可尝试更高分辨率特征、更强 decoder 或边界相关损失，以进一步改善细长结构与小目标表现。

除精度外，还需要关注不同硬件平台上的推理效率与部署可行性，例如嵌入式或移动端的速度与功耗评估。针对小目标可尝试类别重加权或难样本挖掘，缓解类别不均衡。

工程落地中，标注质量与场景覆盖度同样重要，可结合更丰富的城市场景数据做交叉验证，提高鲁棒性。



# 参考文献

1. Zhao H, Shi J, Qi X, Wang X, Jia J. Pyramid Scene Parsing Network. In: CVPR. 2017.
2. Long J, Shelhamer E, Darrell T. Fully Convolutional Networks for Semantic Segmentation. In: CVPR. 2015.
3. Chen L C, Zhu Y, Papandreou G, Schroff F, Adam H. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabv3+). In: ECCV. 2018.
4. He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. In: CVPR. 2016.
5. Brostow G J, Shotton J, Fauqueur J, Cipolla R. Segmentation and Recognition Using Structure from Motion Point Clouds. In: ECCV. 2008.

# 附录

附录可包含开发日志、成员分工、关键配置与运行截图等材料。
