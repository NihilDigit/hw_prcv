#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CIFAR-10 dataset via torchvision.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/cifar10"),
        help="Dataset root directory (default: data/cifar10).",
    )
    args = parser.parse_args()

    try:
        from torchvision.datasets import CIFAR10
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Failed to import torchvision CIFAR10: {exc}") from exc

    args.root.mkdir(parents=True, exist_ok=True)

    train = CIFAR10(root=str(args.root), train=True, download=True)
    test = CIFAR10(root=str(args.root), train=False, download=True)

    print(f"Downloaded CIFAR-10 to: {args.root.resolve()}")
    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

