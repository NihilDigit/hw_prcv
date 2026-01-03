# 逐页内容包（PSPNet）

## 1 封面
- 标题：基于 PSPNet 的 CamVid 语义分割：实现与对照实验
- 副标题：程序演示 + 算法讲解 + 对比实验

## 2 目录 / 主线
- 程序演示：流程与模块
- 算法讲解：PSPNet/PPM
- 对比实验：强弱 backbone + 结构贡献

## 3 任务与数据集
- 语义分割：像素级分类
- CamVid：11 类，train/val/test=367/101/233
- 难点：小目标、细长结构、边界

## 4 评测指标
- IoU / mIoU 公式（见 equations）
- 为什么 mIoU：对类别不均衡敏感

## 5 算法总览（PSPNet）
- backbone → PPM → 融合 → 上采样
- 重点：全局上下文一致性

## 6 PPM 原理
- 1x1 / 2x2 / 3x3 / 6x6 池化
- 上采样后拼接
- 直觉：全局 + 局部上下文融合

## 7 对照模型差异
- FCN：无显式多尺度
- PSPNet：PPM 强化上下文
- DeepLabv3+：ASPP + decoder 强化边界

## 8 程序实现流程图
- 数据管线 → 模型构建 → 训练评测 → 可视化
- 强调可复现：统一设置

## 9 训练配置
- SGD + momentum 0.9
- Poly LR: lr = base_lr * (1 - iter/max_iter)^0.9
- 输入裁剪 360x480，50 epochs

## 10 超参对比（表 3-1）
- batch size=4 明显更好（0.7535）

## 11 R50 结果（表 4-1）
- PSPNet≈FCN（PPM 增益小）
- DeepLabv3+ 略高

## 12 R18 结果（表 4-2）
- 总体下降
- DeepLabv3+ 提升显著（decoder 作用更大）

## 13 曲线对比
- PSPNet R50 loss + mIoU
- R18 vs R50 曲线对比（稳定性与平台期）

## 14 可视化对比
- 输入 / GT / PSPNet / FCN / DeepLabv3+
- 关注边界与小目标

## 15 结论与工程启示
- 强骨干下结构增益小
- 弱骨干下 decoder 更关键
- 局限：单次训练、CamVid 小规模
