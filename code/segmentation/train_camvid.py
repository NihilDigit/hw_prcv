#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.camvid import NUM_CLASSES, get_dataloaders
from src.models.deeplabv3plus import DeepLabV3Plus, DeepLabV3PlusConfig
from src.models.fcn import FCNConfig, build_fcn
from src.models.pspnet import PSPNet, PSPNetConfig
from src.utils.segmentation_metrics import confusion_matrix, intersection_over_union
from src.utils.training import ExperimentLogger, compile_model, save_checkpoint, setup_torch_optimizations


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


def forward_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    out = model(images)
    if isinstance(out, dict) and "out" in out:
        return out["out"]
    return out


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, *, device: torch.device) -> dict:
    model.eval()
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
    total_loss = 0.0
    total_batches = 0
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = forward_logits(model, images)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = criterion(logits, targets)
        total_loss += float(loss.item())
        total_batches += 1

        preds = logits.argmax(dim=1)
        conf += confusion_matrix(preds, targets, num_classes=NUM_CLASSES, ignore_index=255)

    iou, miou = intersection_over_union(conf)
    return {
        "val_loss": total_loss / max(total_batches, 1),
        "miou": float(miou.item()),
        "per_class_iou": [float(x) for x in iou.tolist()],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train semantic segmentation models on CamVid (11 classes).")
    parser.add_argument("--data-root", type=str, default="data/camvid", help="CamVid root (train/, val/, test/...).")
    parser.add_argument("--model", type=str, choices=["pspnet", "fcn", "deeplabv3plus"], default="pspnet")
    parser.add_argument("--backbone", type=str, choices=["resnet18", "resnet34", "resnet50"], default="resnet50")
    parser.add_argument(
        "--backbone-pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet pretrained backbone weights.",
    )
    parser.add_argument(
        "--output-stride",
        type=int,
        choices=[8, 16, 32],
        default=None,
        help="Override backbone output stride (default: resnet50=8, resnet18/34=32).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--crop-h", type=int, default=360)
    parser.add_argument("--crop-w", type=int, default=480)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=str, default=None, help="Run directory name. Default: timestamp.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile().")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    setup_torch_optimizations()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = args.run_id or time.strftime("seg-%Y%m%d-%H%M%S")
    exp_dir = Path("experiments/segmentation") / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    crop_size = (args.crop_h, args.crop_w)
    train_loader, val_loader, _ = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        crop_size=crop_size,
        num_workers=args.num_workers,
    )

    model = build_model(
        args.model,
        num_classes=NUM_CLASSES,
        backbone=args.backbone,
        backbone_pretrained=args.backbone_pretrained,
        output_stride=args.output_stride,
    ).to(device)
    if not args.no_compile:
        model = compile_model(model)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    max_iter = args.epochs * len(train_loader)
    max_iter = max(max_iter, 1)

    power = 0.9

    def poly_lr(iter_idx: int) -> float:
        effective_iter = min(iter_idx + 1, max_iter)
        return (1.0 - effective_iter / max_iter) ** power

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    logger = ExperimentLogger(exp_dir)

    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        train_loss = 0.0
        num_batches = 0

        for images, targets in tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = forward_logits(model, images)
                if isinstance(out, tuple):
                    logits, aux_logits = out
                    loss = criterion(logits, targets) + 0.4 * criterion(aux_logits, targets)
                else:
                    logits = out
                    loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += float(loss.item())
            num_batches += 1

        lr = float(optimizer.param_groups[0]["lr"])

        metrics = evaluate(model, val_loader, device=device)
        is_best = metrics["miou"] > best_miou
        best_miou = max(best_miou, metrics["miou"])

        epoch_time = time.time() - start
        logger.log(
            epoch=epoch,
            train_loss=train_loss / max(num_batches, 1),
            val_loss=metrics["val_loss"],
            val_metric=metrics["miou"],
            lr=lr,
            epoch_time=epoch_time,
        )

        save_checkpoint(model, optimizer, epoch, metrics["miou"], exp_dir, is_best=is_best)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss / max(num_batches, 1):.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"mIoU={metrics['miou']:.4f} | "
            f"best={best_miou:.4f} | "
            f"time={epoch_time:.1f}s"
        )

    with open(exp_dir / "model_info.json", "w", encoding="utf-8") as f:
        if args.model == "pspnet":
            json.dump(
                {
                    "model": "pspnet",
                    "config": asdict(
                        PSPNetConfig(
                            num_classes=NUM_CLASSES,
                            backbone=args.backbone,
                            backbone_pretrained=args.backbone_pretrained,
                            output_stride=_resolve_output_stride(args.backbone, args.output_stride),
                        )
                    ),
                },
                f,
                indent=2,
            )
        elif args.model == "deeplabv3plus":
            json.dump(
                {
                    "model": "deeplabv3plus",
                    "config": asdict(
                        DeepLabV3PlusConfig(
                            num_classes=NUM_CLASSES,
                            backbone=args.backbone,
                            backbone_pretrained=args.backbone_pretrained,
                            output_stride=_resolve_output_stride(args.backbone, args.output_stride),
                        )
                    ),
                },
                f,
                indent=2,
            )
        else:
            json.dump(
                {
                    "model": "fcn",
                    "config": asdict(
                        FCNConfig(
                            num_classes=NUM_CLASSES,
                            backbone=args.backbone,
                            backbone_pretrained=args.backbone_pretrained,
                        )
                    ),
                },
                f,
                indent=2,
            )

    print(f"Done. Artifacts: {exp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
