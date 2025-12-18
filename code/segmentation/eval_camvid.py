#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.camvid import CAMVID_11_CLASSES, NUM_CLASSES, class_id_to_rgb, get_dataloaders
from src.models.deeplabv3plus import DeepLabV3Plus, DeepLabV3PlusConfig
from src.models.fcn import FCNConfig, build_fcn
from src.models.pspnet import PSPNet, PSPNetConfig
from src.utils.segmentation_metrics import confusion_matrix, intersection_over_union


def _resolve_output_stride(backbone: str, output_stride: int | None) -> int:
    if output_stride is not None:
        return output_stride
    if backbone in ("resnet18", "resnet34"):
        return 32
    return 8


def build_model(
    name: str,
    *,
    num_classes: int,
    backbone: str,
    backbone_pretrained: bool,
    output_stride: int | None,
) -> nn.Module:
    resolved_output_stride = _resolve_output_stride(backbone, output_stride)
    if name == "pspnet":
        return PSPNet(
            PSPNetConfig(
                num_classes=num_classes,
                backbone=backbone,
                backbone_pretrained=backbone_pretrained,
                output_stride=resolved_output_stride,
            )
        )
    if name == "deeplabv3plus":
        return DeepLabV3Plus(
            DeepLabV3PlusConfig(
                num_classes=num_classes,
                backbone=backbone,
                backbone_pretrained=backbone_pretrained,
                output_stride=resolved_output_stride,
            )
        )
    if name == "fcn":
        return build_fcn(
            FCNConfig(num_classes=num_classes, backbone=backbone, backbone_pretrained=backbone_pretrained)
        )
    raise ValueError(f"Unknown model: {name}")


def forward_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    out = model(images)
    if isinstance(out, dict) and "out" in out:
        return out["out"]
    if isinstance(out, tuple):
        return out[0]
    return out


def save_triplet(out_dir: Path, idx: int, image: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    img_np = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    gt_rgb = class_id_to_rgb(target.detach().cpu().numpy().astype(np.uint8))
    pred_rgb = class_id_to_rgb(pred.detach().cpu().numpy().astype(np.uint8))

    Image.fromarray(img_np).save(out_dir / f"{idx:03d}_image.png")
    Image.fromarray(gt_rgb).save(out_dir / f"{idx:03d}_gt.png")
    Image.fromarray(pred_rgb).save(out_dir / f"{idx:03d}_pred.png")


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CamVid segmentation checkpoints (mIoU + visualizations).")
    parser.add_argument("--data-root", type=str, default="data/camvid")
    parser.add_argument("--model", type=str, choices=["pspnet", "fcn", "deeplabv3plus"], default="pspnet")
    parser.add_argument("--backbone", type=str, choices=["resnet18", "resnet34", "resnet50"], default="resnet50")
    parser.add_argument(
        "--backbone-pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use ImageNet pretrained backbone weights (usually unnecessary for eval).",
    )
    parser.add_argument(
        "--output-stride",
        type=int,
        choices=[8, 16, 32],
        default=None,
        help="Override backbone output stride (default: resnet50=8, resnet18/34=32).",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth).")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-h", type=int, default=360)
    parser.add_argument("--crop-w", type=int, default=480)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default=None, help="Outputs directory for metrics and images.")
    parser.add_argument("--max-vis", type=int, default=12, help="Max samples to visualize.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args.model,
        num_classes=NUM_CLASSES,
        backbone=args.backbone,
        backbone_pretrained=args.backbone_pretrained,
        output_stride=args.output_stride,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    crop_size = (args.crop_h, args.crop_w)
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root, batch_size=args.batch_size, crop_size=crop_size, num_workers=args.num_workers
    )
    loader = val_loader if args.split == "val" else test_loader

    out_dir = Path(args.out_dir) if args.out_dir else Path("report/figures/segmentation") / f"eval-{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
    total_loss = 0.0
    total_batches = 0

    vis_dir = out_dir / "vis"
    vis_count = 0

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=args.split)):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = forward_logits(model, images)
        loss = criterion(logits, targets)
        total_loss += float(loss.item())
        total_batches += 1

        preds = logits.argmax(dim=1)
        conf += confusion_matrix(preds, targets, num_classes=NUM_CLASSES, ignore_index=255)

        if vis_count < args.max_vis:
            for i in range(min(images.shape[0], args.max_vis - vis_count)):
                # input images are normalized; for quick view, just min-max to 0..255
                img = images[i].detach().cpu()
                img = (img - img.min()) / (img.max() - img.min() + 1e-6)
                save_triplet(vis_dir, vis_count, img, targets[i].detach().cpu(), preds[i].detach().cpu())
                vis_count += 1

    iou, miou = intersection_over_union(conf)
    metrics = {
        "split": args.split,
        "val_loss": total_loss / max(total_batches, 1),
        "miou": float(miou.item()),
        "per_class_iou": {CAMVID_11_CLASSES[i]: float(iou[i].item()) for i in range(NUM_CLASSES)},
        "checkpoint": args.checkpoint,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
