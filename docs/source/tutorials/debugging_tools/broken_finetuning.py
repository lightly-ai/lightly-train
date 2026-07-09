"""
Broken Fine-Tuning Script for the Debugging Tutorial

Runs ``lightly_train.train_image_classification`` with ``UnstableReLU``
monkey-patched into ``torchvision.models.resnet18`` and with
``underflow_overflow`` debugging enabled. Training will abort as soon as the
first ``inf``/``nan`` propagates through the network, and LightlyTrain writes a
per-rank debug log to::

    <out>/debug/underflow_overflow_rank0.log

The log lists the most recent forward frames (model, weight, input, output
min/max) so the user can pinpoint which layer caused the overflow.

Run with::

    python broken_finetuning.py
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


# Where everything goes. Re-running with the same ``out`` will fail unless
# ``overwrite=True`` is set on the training call.
OUT_DIR: str = "out/debugging_broken"
DATA_DIR: str = "datasets/debugging_tutorial"


def main() -> None:
    # 1. Make sure the synthetic dataset exists. Cheap to rebuild, so we just
    #    regenerate every run.
    build_synthetic_dataset(
        data_dir=DATA_DIR,
        num_train_per_class=32,
        num_val_per_class=8,
        image_size=(96, 96),
        classes=CLASS_NAMES,
    )

    classes = build_classes_dict(CLASS_NAMES)

    # 2. Patch torchvision so the model LightlyTrain builds has UnstableReLU,
    #    then run training with underflow/overflow debugging enabled.
    #
    #    The training will abort as soon as the first inf/nan is detected —
    #    that is the whole point of the monitor.
    with patched_resnet18():
        lightly_train.train_image_classification(
            out=OUT_DIR,
            data={
                "train": Path(DATA_DIR) / "train",
                "val": Path(DATA_DIR) / "val",
                "classes": classes,
            },
            model="torchvision/resnet18",
            # fp16 overflows faster than bf16 (the default) and makes the
            # demo less reliant on a large learning rate.
            precision="16-mixed",
            # CPU keeps the tutorial portable.
            accelerator="cpu",
            strategy="auto",
            steps=10,
            batch_size=8,
            num_workers=0,
            # A slightly larger learning rate makes the overflow trigger
            # within the first handful of steps.
            model_args={"lr": 1e-3},
            # DebugUnderflowOverflow: register forward hooks on every module,
            # abort on first inf/nan and dump the last ``max_frames_to_save``
            # frames so the bad layer can be identified.
            debug_args={
                "underflow_overflow": {
                    "enabled": True,
                    "max_frames_to_save": 21,
                }
            },
            # We do not want to clutter the demo run with checkpointing.
            save_checkpoint_args={"save_last": False},
            overwrite=True,
        )

    print(
        "\nTraining aborted by DebugUnderflowOverflow (expected). "
        f"Inspect the log at: {Path(OUT_DIR) / 'debug' / 'underflow_overflow_rank0.log'}"
    )


if __name__ == "__main__":
    main()