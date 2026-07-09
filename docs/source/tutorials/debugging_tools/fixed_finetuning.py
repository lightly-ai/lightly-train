"""
Fixed (Baseline) Fine-Tuning Script for the Debugging Tutorial

Runs the same fine-tuning task as ``broken_finetuning.py`` but on an unmodified
torchvision ResNet18. The purpose is to provide a comparison point that shows
what "healthy" training looks like — stable gradient norms, decreasing loss —
so the user can clearly see the contrast with the broken run.

Run with::

    python fixed_finetuning.py
"""

from __future__ import annotations

from pathlib import Path

import lightly_train

from setup_data import (
    CLASS_NAMES,
    build_classes_dict,
    build_synthetic_dataset,
)


OUT_DIR: str = "out/debugging_fixed"
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

    lightly_train.train_image_classification(
        out=OUT_DIR,
        data={
            "train": Path(DATA_DIR) / "train",
            "val": Path(DATA_DIR) / "val",
            "classes": classes,
        },
        # Same model string as the broken run. The only difference is that
        # we are NOT inside ``patched_resnet18()``, so the standard ReLU is
        # used.
        model="torchvision/resnet18",
        precision="16-mixed",
        accelerator="cpu",
        strategy="auto",
        steps=10,
        batch_size=8,
        num_workers=0,
        model_args={"lr": 1e-3},
        # No debug_args: gradient norm logging is on by default and that is
        # all we need for the comparison.
        save_checkpoint_args={"save_last": False},
        overwrite=True,
    )

    metrics_path = Path(OUT_DIR) / "train.log"
    print(f"\nTraining complete. Console log: {metrics_path}")
    print(
        "Run `python visualize_gradient_norms.py "
        f"--title Fixed {metrics_path} --output out/fixed_gradient_norms.png` "
        "to inspect the gradient norm trend."
    )


if __name__ == "__main__":
    main()