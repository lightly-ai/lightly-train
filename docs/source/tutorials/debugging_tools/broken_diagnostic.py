"""
Broken Fine-Tuning WITHOUT DebugUnderflowOverflow (diagnostic only)

This script runs the same broken model as ``broken_finetuning.py`` but does
**not** enable ``debug_args.underflow_overflow``. The point is to capture the
gradient norm trajectory while training is unstable, so that the tutorial can
contrast it with the healthy fixed run.

With the unstable layer in place, gradient norms blow up within a handful of
steps. ``DebugUnderflowOverflow`` would normally abort on the first ``inf``/
``nan`` and prevent any metrics from being logged — that is why we keep it
disabled here.

Run with::

    python broken_diagnostic.py
"""

from __future__ import annotations

from pathlib import Path

import lightly_train

from broken_model import patched_resnet18
from setup_data import (
    CLASS_NAMES,
    build_classes_dict,
    build_synthetic_dataset,
)


OUT_DIR: str = "out/debugging_broken_diagnostic"
DATA_DIR: str = "datasets/debugging_tutorial"


def main() -> None:
    build_synthetic_dataset(
        data_dir=DATA_DIR,
        num_train_per_class=32,
        num_val_per_class=8,
        image_size=(96, 96),
        classes=CLASS_NAMES,
    )

    classes = build_classes_dict(CLASS_NAMES)

    with patched_resnet18(scale=0.8):
        lightly_train.train_image_classification(
            out=OUT_DIR,
            data={
                "train": Path(DATA_DIR) / "train",
                "val": Path(DATA_DIR) / "val",
                "classes": classes,
            },
            model="torchvision/resnet18",
            precision="16-mixed",
            accelerator="cpu",
            strategy="auto",
            steps=10,
            batch_size=8,
            num_workers=0,
            model_args={"lr": 1e-3},
            # Deliberately NO debug_args here — gradient norm logging is on by
            # default and we want the console log to fill up so we can
            # visualise the explosion.
            save_checkpoint_args={"save_last": False},
            overwrite=True,
        )

    log_path = Path(OUT_DIR) / "train.log"
    print(
        f"\nDiagnostic run complete. Console log: {log_path}\n"
        f"Compare with the fixed run:\n"
        f"  python visualize_gradient_norms.py --compare \\\n"
        f"    broken={log_path} \\\n"
        f"    fixed=out/debugging_fixed/train.log \\\n"
        f"    --output out/gradient_norm_comparison.png"
    )


if __name__ == "__main__":
    main()