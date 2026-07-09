"""
Dataset Setup for the Debugging Tutorial

This script generates a tiny synthetic image-classification dataset on disk
that LightlyTrain's ``train_image_classification`` can consume directly. We use
synthetic data instead of CIFAR-10 so the tutorial is fully reproducible
offline — no internet connection required.

The dataset layout follows the folder-of-folders convention expected by the
multiclass image-classification task:

    <data_dir>/
        train/
            class_0/img_0000.png, img_0001.png, ...
            class_1/img_0000.png, img_0001.png, ...
            ...
        val/
            class_0/img_0000.png, ...
            ...

The images themselves are random RGB noise. The exact pixel values do not
matter — we only need a few labeled samples so the fine-tuning pipeline runs
end-to-end and the gradient flow exercises the unstable layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


CLASS_NAMES: list[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

DEFAULT_NUM_TRAIN_PER_CLASS: int = 32
DEFAULT_NUM_VAL_PER_CLASS: int = 8
DEFAULT_IMAGE_SIZE: tuple[int, int] = (96, 96)


def _write_image(path: Path, height: int, width: int, seed: int) -> None:
    """Write a random RGB PNG to ``path``."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path)


def build_synthetic_dataset(
    data_dir: str | Path = "datasets/debugging_tutorial",
    num_train_per_class: int = DEFAULT_NUM_TRAIN_PER_CLASS,
    num_val_per_class: int = DEFAULT_NUM_VAL_PER_CLASS,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    classes: Iterable[str] = CLASS_NAMES,
) -> Path:
    """Build a synthetic image-classification dataset at ``data_dir``.

    Creates a ``train/`` and ``val/`` split, each containing one subdirectory
    per class. Each subdirectory is filled with random RGB PNGs.

    Args:
        data_dir:
            Root directory for the synthetic dataset.
        num_train_per_class:
            Number of training images per class.
        num_val_per_class:
            Number of validation images per class.
        image_size:
            ``(height, width)`` of each generated image.
        classes:
            Iterable of class names. The number of classes becomes the
            classification problem size.

    Returns:
        The resolved ``Path`` to ``data_dir``.
    """
    data_dir = Path(data_dir).resolve()
    height, width = image_size
    class_list = list(classes)

    print(f"Building synthetic dataset at: {data_dir}")
    print(f"  - classes: {len(class_list)} ({class_list[:3]}, ...)")
    print(f"  - train images per class: {num_train_per_class}")
    print(f"  - val images per class:   {num_val_per_class}")
    print(f"  - image size: {height}x{width}")

    for split, num_per_class in (("train", num_train_per_class), ("val", num_val_per_class)):
        split_idx = 0 if split == "train" else 1
        for class_idx, class_name in enumerate(class_list):
            split_dir = data_dir / split / class_name
            for img_idx in range(num_per_class):
                # Deterministic seed so re-runs produce identical data.
                # We avoid Python's hash() because it is randomised across
                # processes by PYTHONHASHSEED, which would make the dataset
                # different every time we run a fresh Python script.
                seed = split_idx * 1_000_000 + class_idx * 10_000 + img_idx
                _write_image(
                    path=split_dir / f"img_{img_idx:04d}.png",
                    height=height,
                    width=width,
                    seed=seed,
                )

    total = len(class_list) * (num_train_per_class + num_val_per_class)
    print(f"\nDataset built. Total images: {total}")
    print(f"Train dir: {data_dir / 'train'}")
    print(f"Val dir:   {data_dir / 'val'}")
    return data_dir


def build_classes_dict(classes: Iterable[str] = CLASS_NAMES) -> dict[int, str]:
    """Build the ``classes`` argument for ``train_image_classification``."""
    return {idx: name for idx, name in enumerate(classes)}


if __name__ == "__main__":
    build_synthetic_dataset()
    print(
        "\nNext step: pass this dataset to "
        "`lightly_train.train_image_classification(model='torchvision/resnet18', ...)`."
    )