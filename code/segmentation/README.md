# CamVid 语义分割（PSPNet/FCN/DeepLabv3+）

## 数据目录

默认使用 `data/camvid/`，其结构应为：

```
data/camvid/
  train/      *.png
  trainannot/ *.png
  val/        *.png
  valannot/   *.png
  test/       *.png
  testannot/  *.png
```

## 训练

```bash
# PSPNet（默认 ResNet-50 backbone）
pixi run python code/segmentation/train_camvid.py --model pspnet --epochs 50 --batch-size 8

# PSPNet + ResNet-18（弱 backbone 对比；output_stride 自动切到 32）
pixi run python code/segmentation/train_camvid.py --model pspnet --backbone resnet18 --epochs 50 --batch-size 8

# FCN baseline（torchvision）
pixi run python code/segmentation/train_camvid.py --model fcn --epochs 50 --batch-size 4

# FCN + ResNet-18（本项目实现的轻量 FCN）
pixi run python code/segmentation/train_camvid.py --model fcn --backbone resnet18 --epochs 50 --batch-size 4

# DeepLabv3+（本项目实现）
pixi run python code/segmentation/train_camvid.py --model deeplabv3plus --epochs 50 --batch-size 4

# DeepLabv3+ + ResNet-18（output_stride 自动切到 32）
pixi run python code/segmentation/train_camvid.py --model deeplabv3plus --backbone resnet18 --epochs 50 --batch-size 4
```

训练日志与 checkpoint 默认输出到 `experiments/segmentation/<run-id>/`。

## 曲线

```bash
pixi run python code/segmentation/plot_training_curves.py --run-dir experiments/segmentation/<run-id>
```

默认保存到 `report/figures/segmentation/<run-id>/`。

## 评测 + 可视化

```bash
pixi run python code/segmentation/eval_camvid.py --model pspnet --split val \\
  --checkpoint experiments/segmentation/<run-id>/best.pth \\
  --out-dir report/figures/segmentation/<run-id>/eval_pspnet
```
