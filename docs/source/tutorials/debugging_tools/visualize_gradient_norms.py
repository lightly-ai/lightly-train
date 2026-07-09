"""
Gradient-Norm Visualization for the Debugging Tutorial

Parses ``train.log`` files written by ``lightly_train.train_image_classification``
and produces a side-by-side plot of gradient norms over training steps for
both the broken and the fixed run. Also prints a quick stability assessment.

Why parse ``train.log`` and not ``metrics.jsonl``?
    ``train_image_classification`` uses :class:`TaskLoggerArgs`, which only
    supports ``tensorboard``, ``mlflow`` and ``wandb`` — there is no JSONL
    output for fine-tuning tasks. The console ``train.log`` does contain a
    ``grad_norm: X.XXXX`` field on every ``Train Step`` line, so we parse
    that instead.

The relevant log line format is::

    [TIMESTAMP][INFO] Train Step  N/M | Epoch E | train_loss: X.XXXX | lr: X.XXXXXXXX | grad_norm: X.XXXX | Profiling [ ... ]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


# Regex for the train step lines printed by helpers.log_step():
#   Train Step  3/10 | Epoch 0 | train_loss: 2.3020 | lr: 0.00040000 | grad_norm: 4.5002 | Profiling [ ... ]
# Notes:
#   - The line is optionally prefixed by a "[timestamp][LEVEL] " header from
#     the logging system.
#   - The step number may have leading whitespace (" 3/10").
#   - The numeric fields use printf-style formatting that can emit "nan" or
#     "inf".
_TRAIN_STEP_RE = re.compile(
    r"Train\s+Step\s+(?P<step>\d+)/(?:\d+)\s*\|\s*"
    r"Epoch\s+(?P<epoch>\d+)\s*\|\s*"
    r"train_loss:\s*(?P<train_loss>[\-\+]?(?:nan|inf|\d+\.\d+|\d+))\s*\|\s*"
    r"lr:\s*(?P<lr>[\-\+]?(?:nan|inf|\d+\.\d+|\d+))\s*\|\s*"
    r"grad_norm:\s*(?P<grad_norm>[\-\+]?(?:nan|inf|\d+\.\d+|\d+))"
)


def _parse_float(text: str) -> float:
    """Parse ``nan``/``inf``/regular floats from the log formatter."""
    text = text.strip()
    if text == "nan":
        return float("nan")
    if text in ("inf", "+inf"):
        return float("inf")
    if text == "-inf":
        return float("-inf")
    return float(text)


def parse_train_log(log_path: Path) -> list[dict[str, float]]:
    """Extract per-step metrics from a ``train.log`` file.

    Returns a list of dicts with keys ``step``, ``epoch``, ``train_loss``,
    ``lr`` and ``grad_norm``. The list is in file order.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    rows: list[dict[str, float]] = []
    with log_path.open("r") as fh:
        for line in fh:
            match = _TRAIN_STEP_RE.search(line)
            if match is None:
                continue
            rows.append(
                {
                    "step": int(match.group("step")),
                    "epoch": int(match.group("epoch")),
                    "train_loss": _parse_float(match.group("train_loss")),
                    "lr": _parse_float(match.group("lr")),
                    "grad_norm": _parse_float(match.group("grad_norm")),
                }
            )
    print(f"Parsed {len(rows)} train-step lines from {log_path}")
    return rows


def extract_gradient_norms(
    rows: list[dict[str, float]],
) -> tuple[list[int], list[float]]:
    """Pull (step, grad_norm) pairs from a list of parsed log rows."""
    steps: list[int] = []
    norms: list[float] = []
    for row in rows:
        norm = row["grad_norm"]
        if not np.isfinite(norm):
            continue
        steps.append(row["step"])
        norms.append(norm)
    return steps, norms


def extract_train_loss(
    rows: list[dict[str, float]],
) -> tuple[list[int], list[float]]:
    """Pull (step, train_loss) pairs from a list of parsed log rows."""
    steps: list[int] = []
    losses: list[float] = []
    for row in rows:
        loss = row["train_loss"]
        if not np.isfinite(loss):
            continue
        steps.append(row["step"])
        losses.append(loss)
    return steps, losses


def assess_stability(steps: list[int], norms: list[float]) -> str:
    """Return a human-readable stability assessment for a gradient norm series."""
    if len(norms) < 3:
        return "Not enough data points for stability analysis."

    norms_arr = np.asarray(norms)
    max_norm = float(norms_arr.max())
    mean_norm = float(norms_arr.mean())
    std_norm = float(norms_arr.std())
    last_vs_first = norms_arr[-1] / max(norms_arr[0], 1e-12)

    lines: list[str] = [
        f"  - min:   {norms_arr.min():.4f}",
        f"  - max:   {max_norm:.4f}",
        f"  - mean:  {mean_norm:.4f}",
        f"  - std:   {std_norm:.4f}",
        f"  - ratio last/first: {last_vs_first:.2f}x",
    ]

    if last_vs_first > 100:
        lines.append("  - assessment: EXPLODING gradients (last/first > 100x)")
    elif last_vs_first > 5:
        lines.append("  - assessment: growing gradients (possible instability)")
    elif last_vs_first < 0.2:
        lines.append("  - assessment: vanishing gradients")
    else:
        lines.append("  - assessment: STABLE")
    return "\n".join(lines)


def plot_single_run(
    log_path: Path,
    title: str,
    save_path: Path | None = None,
) -> None:
    """Plot gradient norms and train loss for a single run."""
    rows = parse_train_log(log_path)
    grad_steps, grad_norms = extract_gradient_norms(rows)
    loss_steps, train_losses = extract_train_loss(rows)

    if not grad_steps:
        print(f"  [warning] no finite grad_norm entries in {log_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if grad_steps:
        axes[0].plot(grad_steps, grad_norms, marker="o", markersize=4, color="tab:blue")
        axes[0].set_yscale("log")
        axes[0].set_title(f"{title} - Gradient Norms (log scale)")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("grad_norm")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(
            0.5,
            0.5,
            "No finite grad_norm entries",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_title(f"{title} - Gradient Norms")

    if loss_steps:
        axes[1].plot(loss_steps, train_losses, marker="o", markersize=4, color="tab:orange")
        axes[1].set_title(f"{title} - Train Loss")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("train_loss")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No train_loss entries",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title(f"{title} - Train Loss")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved plot to {save_path}")
    else:
        plt.show()

    print(f"\nGradient norm stability ({title}):")
    print(assess_stability(grad_steps, grad_norms))


def plot_comparison(
    runs: dict[str, Path],
    save_path: Path | None = None,
) -> None:
    """Plot gradient norms from multiple runs on a single axis (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = 0
    for label, log_path in runs.items():
        rows = parse_train_log(log_path)
        grad_steps, grad_norms = extract_gradient_norms(rows)
        if not grad_steps:
            print(f"  [skip] {label}: no finite grad_norm entries")
            continue
        ax.plot(grad_steps, grad_norms, marker="o", markersize=4, label=label)
        plotted += 1

    if plotted == 0:
        print("No runs had any gradient_norm entries; nothing to plot.")
        plt.close(fig)
        return

    ax.set_yscale("log")
    ax.set_title("Gradient Norms: Broken vs Fixed Fine-Tuning")
    ax.set_xlabel("Step")
    ax.set_ylabel("grad_norm (log scale)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved comparison plot to {save_path}")
    else:
        plt.show()


def _expand_compare_args(values: Iterable[str]) -> dict[str, Path]:
    """Expand ``["label1=path1", "label2=path2"]`` into a dict."""
    out: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(
                f"--compare entries must be of the form label=path, got: {item!r}"
            )
        label, path_str = item.split("=", 1)
        out[label.strip()] = Path(path_str.strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize gradient norms from a LightlyTrain fine-tuning run."
    )
    parser.add_argument("log_path", nargs="?", help="Path to a train.log file.")
    parser.add_argument("--output", "-o", help="Where to save the plot (PNG).")
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare runs, e.g. --compare broken=out/debugging_broken_diagnostic/train.log fixed=out/debugging_fixed/train.log",
    )
    parser.add_argument("--title", default="Run", help="Title prefix for plots.")

    args = parser.parse_args()

    if args.compare:
        runs = _expand_compare_args(args.compare)
        plot_comparison(runs=runs, save_path=Path(args.output) if args.output else None)
        return

    if not args.log_path:
        parser.error("Provide either a log_path or --compare entries.")

    plot_single_run(
        log_path=Path(args.log_path),
        title=args.title,
        save_path=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()