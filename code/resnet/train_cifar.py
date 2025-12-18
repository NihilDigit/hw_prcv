#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.cpp_extension import CUDA_HOME

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_cifar10_dataloaders
from src.models import cifar_plain, cifar_resnet
from src.utils.training import setup_torch_optimizations


@dataclass(frozen=True)
class TrainConfig:
    model: str
    data_root: str
    epochs: int
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    shortcut_type: str
    compile: bool
    amp: bool
    num_workers: int
    seed: int


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    amp: bool,
    max_steps: int | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_acc += _accuracy(logits, labels)
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    return total_loss / max(1, steps), total_acc / max(1, steps)


@torch.no_grad()
def _eval(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    max_steps: int | None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        total_acc += _accuracy(logits, labels)
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    return total_loss / max(1, steps), total_acc / max(1, steps)


def _build_model(model_name: str, shortcut_type: str) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "plain20":
        return cifar_plain(depth=20)
    if model_name == "plain56":
        return cifar_plain(depth=56)
    if model_name == "resnet20":
        return cifar_resnet(depth=20, shortcut_type=shortcut_type)
    if model_name == "resnet56":
        return cifar_resnet(depth=56, shortcut_type=shortcut_type)
    raise ValueError(f"Unknown model: {model_name}")


def _cuda_headers_available() -> bool:
    candidates = []
    if CUDA_HOME:
        candidates.append(Path(CUDA_HOME) / "include" / "cuda.h")
    candidates.extend(
        [
            Path("/usr/local/cuda/include/cuda.h"),
            Path("/opt/cuda/include/cuda.h"),
            Path("/usr/include/cuda.h"),
        ]
    )
    return any(path.exists() for path in candidates)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 Plain/ResNet (20/56) for reproduction.")
    parser.add_argument("--model", required=True, choices=["plain20", "plain56", "resnet20", "resnet56"])
    parser.add_argument("--data-root", default="data/cifar10")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--shortcut-type", choices=["A", "B"], default="A")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile().")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP.")
    parser.add_argument("--max-steps-train", type=int, default=None, help="Debug: cap train steps per epoch.")
    parser.add_argument("--max-steps-test", type=int, default=None, help="Debug: cap test steps per epoch.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: experiments/resnet/<model>/<timestamp>).",
    )
    args = parser.parse_args()

    setup_torch_optimizations()
    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"
    compile_requested = (not args.no_compile) and hasattr(torch, "compile")
    compile_enabled = compile_requested
    if compile_enabled and device.type == "cuda" and not _cuda_headers_available():
        compile_enabled = False
        print("[WARN] CUDA headers (cuda.h) not found; disabling torch.compile(). Use --no-compile or install CUDA toolkit headers.")
    if compile_enabled and device.type == "cuda" and CUDA_HOME:
        # Triton looks for cuda.h in sys.prefix/targets/... by default; on Arch it's under /opt/cuda/include.
        os.environ.setdefault("TRITON_CUDART_PATH", str(Path(CUDA_HOME) / "include"))

    out_dir = args.out_dir
    if out_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_dir = Path("experiments") / "resnet" / args.model / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        model=args.model,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        shortcut_type=args.shortcut_type,
        compile=compile_enabled,
        amp=amp,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n")

    train_loader, test_loader = get_cifar10_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = _build_model(args.model, shortcut_type=args.shortcut_type).to(device)
    if compile_enabled:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            compile_enabled = False
            print(f"[WARN] torch.compile() failed, continuing without compile: {exc}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=False,
    )
    scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    metrics_path = out_dir / "metrics.json"
    history: list[dict] = []

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = _train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler if amp else None,
            amp=amp,
            max_steps=args.max_steps_train,
        )
        test_loss, test_acc = _eval(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            amp=amp,
            max_steps=args.max_steps_test,
        )

        scheduler.step()
        epoch_time = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": lr,
            "epoch_time_sec": epoch_time,
        }
        history.append(row)
        metrics_path.write_text(json.dumps(history, indent=2) + "\n")

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt = {
                "epoch": epoch,
                "model": args.model,
                "model_state_dict": model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_test_acc": best_acc,
                "config": asdict(config),
            }
            torch.save(ckpt, out_dir / "best.pth")

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} acc {test_acc:.4f} | "
            f"lr {lr:.5f} | {epoch_time:.1f}s"
        )

    print(f"Done. best_test_acc={best_acc:.4f} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
