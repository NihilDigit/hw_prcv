#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plotting import setup_style


def _load_metrics(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _series(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows]


def _epochs(rows: list[dict]) -> list[int]:
    return [int(r["epoch"]) for r in rows]


def _plot_curves(all_runs: dict[str, list[dict]], out_dir: Path) -> None:
    setup_style("science")

    # Train loss
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    for name, rows in all_runs.items():
        ax1.plot(_epochs(rows), _series(rows, "train_loss"), label=name)
    ax1.set_title("CIFAR-10 Train Loss (Plain vs ResNet)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.grid(True, alpha=0.2)
    ax1.legend()
    fig1.savefig(out_dir / "resnet_train_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # Test accuracy
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for name, rows in all_runs.items():
        ax2.plot(_epochs(rows), _series(rows, "test_acc"), label=name)
    ax2.set_title("CIFAR-10 Test Accuracy (Plain vs ResNet)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy")
    ax2.grid(True, alpha=0.2)
    ax2.legend()
    fig2.savefig(out_dir / "resnet_test_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot ResNet reproduction curves from a run directory.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing plain20/plain56/resnet20/resnet56 subdirs.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: report/figures/resnet/<run-id>).",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    run_id = run_dir.name

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("report") / "figures" / "resnet" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ["plain20", "plain56", "resnet20", "resnet56"]
    all_runs: dict[str, list[dict]] = {}
    for model in models:
        try:
            all_runs[model] = _load_metrics(run_dir / model)
        except FileNotFoundError:
            continue

    if not all_runs:
        raise SystemExit(f"No metrics.json found under: {run_dir}")

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "models": {},
    }
    for model, rows in all_runs.items():
        if not rows:
            continue
        last = rows[-1]
        best_test_acc = max(float(r["test_acc"]) for r in rows)
        summary["models"][model] = {
            "epochs": len(rows),
            "last": last,
            "best_test_acc": best_test_acc,
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot_curves(all_runs, out_dir)

    print(f"Wrote: {out_dir / 'resnet_train_loss.png'}")
    print(f"Wrote: {out_dir / 'resnet_test_accuracy.png'}")
    print(f"Wrote: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
