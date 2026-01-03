# PSPNet PPT 制作包（单文件汇总）

> 本文件整合 `pspnet_ppt/` 的全部内容，便于直接交给 PPT 设计师。

---

## 1) 制作包说明（README）

定位：以“程序演示 + 算法讲解 + 对比实验”为主线，内容来自 `report/pspnet_report.md`。

包含内容：
- `slides_outline.md`：15页结构与内容要点
- `slide_content.md`：逐页讲稿要点与视觉建议
- `data/*.csv`：表格数据（超参对比、R50/R18 结果）
- `figures/*.png`：训练曲线、mIoU曲线、样例可视化
- `notes/equations.md`：关键公式

备注：前端演示部分由你单独制作，本包为 PPT 结构与素材。

---

## 2) 15页结构（slides_outline.md）

# PSPNet 程序演示 + 算法讲解 + 对比实验（15页）

1. 封面
2. 目录 / 讲述主线
3. 任务与数据集（CamVid + 难点）
4. 评测指标（mIoU）
5. 算法总览：PSPNet 流程
6. PPM 原理（多尺度池化）
7. 对照模型差异：FCN / PSPNet / DeepLabv3+
8. 程序实现流程图（数据→模型→训练→评测→可视化）
9. 训练配置（Poly LR + 关键超参）
10. 超参对比（表 3-1）
11. 定量结果：ResNet-50 组对比（表 4-1）
12. 定量结果：ResNet-18 组对比（表 4-2）
13. 曲线对比（mIoU 收敛 & 代表性曲线）
14. 定性可视化对比（样例 000）
15. 结论与工程启示（含局限与改进）

---

## 3) 逐页内容（slide_content.md）

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

---

## 4) 关键公式（notes/equations.md）

1) IoU / mIoU
- IoU_c = TP_c / (TP_c + FP_c + FN_c)
- mIoU = mean over classes

2) Poly 学习率调度
- lr = base_lr * (1 - iter / max_iter) ^ 0.9

---

## 5) 数据表（CSV）

### 5.1 超参对比（table_hyperparams.csv）
```
setting,batch_size,lr,best_miou_val,best_epoch
A,8,0.01,0.5539,27
B,4,0.01,0.7535,48
```

### 5.2 ResNet-50 对照（table_r50_results.csv）
```
model,backbone,miou_val,best_epoch,run_id
PSPNet,ResNet-50,0.7534,48,seg-v2-pspnet
FCN,ResNet-50,0.7535,50,seg-v2-fcn
DeepLabv3+,ResNet-50,0.7603,48,seg-v2-deeplabv3plus
```

### 5.3 ResNet-18 对照（table_r18_results.csv）
```
model,backbone,miou_val,best_epoch,run_id
PSPNet,ResNet-18,0.6191,40,seg-r18-pspnet
FCN,ResNet-18,0.6108,38,seg-r18-fcn
DeepLabv3+,ResNet-18,0.7097,45,seg-r18-deeplabv3plus
```

---

## 6) 图像素材路径

训练/曲线：
- `pspnet_ppt/figures/pspnet_r50_loss.png`
- `pspnet_ppt/figures/pspnet_r50_miou.png`
- `pspnet_ppt/figures/fcn_r50_miou.png`
- `pspnet_ppt/figures/deeplab_r50_miou.png`
- `pspnet_ppt/figures/pspnet_r18_miou.png`
- `pspnet_ppt/figures/deeplab_r18_miou.png`

定性可视化（样例 000）：
- `pspnet_ppt/figures/sample_000_image.png`
- `pspnet_ppt/figures/sample_000_gt.png`
- `pspnet_ppt/figures/sample_000_pspnet.png`
- `pspnet_ppt/figures/sample_000_fcn.png`
- `pspnet_ppt/figures/sample_000_deeplab.png`

---

## 7) 设计简报（design_brief.md）

- 关键词：直观、工程感、流程清晰
- 颜色：蓝/青灰 + 橙色强调
- 版式：图表 > 文字，流程图居中
- 重点页：流程图（第8页）、对照表（第11/12页）、可视化对比（第14页）
