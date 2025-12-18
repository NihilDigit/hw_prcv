#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick environment sanity checks (PyTorch/CUDA).")
    parser.add_argument("--verbose", action="store_true", help="Print additional details.")
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Failed to import torch: {exc}") from exc

    print(f"Python: {platform.python_version()} ({platform.platform()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if args.verbose:
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        print(f"GPU[{device_index}]: {torch.cuda.get_device_name(device_index)}")
        if args.verbose:
            print(f"GPU capability: {torch.cuda.get_device_capability(device_index)}")
            print(f"GPU count: {torch.cuda.device_count()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

