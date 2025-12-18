#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plotting import savefig, setup_style


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot CamVid training curves from experiments/*/training_log.json")
    parser.add_argument("--run-dir", type=str, required=True, help="experiments/segmentation/<run-id>")
    parser.add_argument("--out-dir", type=str, default=None, help="Default: report/figures/segmentation/<run-id>")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_path = run_dir / "training_log.json"
    history = json.loads(log_path.read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir) if args.out_dir else Path("report/figures/segmentation") / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style("science")

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs, history["train_loss"], label="train loss")
    if history.get("val_loss"):
        ax.plot(epochs[: len(history["val_loss"])], history["val_loss"], label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig(fig, "camvid_loss", output_dir=out_dir)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    if history.get("val_metric"):
        ax.plot(epochs[: len(history["val_metric"])], history["val_metric"], label="val mIoU")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig(fig, "camvid_miou", output_dir=out_dir)
    plt.close(fig)

    print(f"Saved figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
