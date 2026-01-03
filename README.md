# prcv

Computer vision coursework project with two main tracks:
- ResNet CIFAR-10 reproduction and report
- PSPNet CamVid semantic segmentation implementation + comparisons

## Key outputs
- ResNet PPT: `resnet_ppt/resnet_ppt.pptx`
- PSPNet PPT package: `pspnet_ppt/` (includes `psp_ppt.pptx` + assets)
- Reports: `report/resnet_report.md`, `report/pspnet_report.md`

## Structure
- `src/` model code (ResNet, PSPNet, FCN, DeepLabv3+)
- `code/` training/eval scripts
- `data/` datasets (CamVid, CIFAR-10, etc.)
- `experiments/` checkpoints and logs
- `report/` figures + writeups
- `resnet_ppt/`, `pspnet_ppt/` PPT deliverables

## Quick start (PSPNet/CamVid)
See `code/segmentation/README.md` for full commands and dataset layout.

## Demo app (PSPNet/FCN/DeepLabv3+)
Interactive inference UI:

```bash
pixi run python app.py
```

Requirements:
- Weights in `experiments/segmentation/`:
  - `seg-v2-pspnet/best.pth`
  - `seg-r18-pspnet/best.pth`
  - `seg-v2-fcn/best.pth`
  - `seg-r18-fcn/best.pth`
  - `seg-v2-deeplabv3plus/best.pth`
  - `seg-r18-deeplabv3plus/best.pth`
- CamVid test images in `data/camvid/test/`

## Notes
- Environment is managed via `pixi.toml`.
- Large binaries (datasets, checkpoints) are not tracked in git.
