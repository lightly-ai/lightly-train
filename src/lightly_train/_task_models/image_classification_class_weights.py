#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from lightly_train._data import label_helpers
from lightly_train._data.image_classification_dataset import (
    ImageClassificationDataArgs,
    ImageClassificationDatasetArgs,
)


def internal_ordered_class_ids(
    classes: dict[int, str],
    ignore_classes: set[int] | None,
) -> list[int]:
    """Return surviving user class IDs in internal-class order.

    The order matches
    ``label_helpers.get_class_id_to_internal_class_id_mapping`` and therefore
    the model head and dataset mapping. No sorting is applied.
    """
    mapping = label_helpers.get_class_id_to_internal_class_id_mapping(
        class_ids=classes.keys(),
        ignore_classes=ignore_classes,
    )
    return sorted(mapping, key=lambda class_id: mapping[class_id])


def count_training_labels(
    train_dataset_args: ImageClassificationDatasetArgs,
) -> tuple[list[int], int, list[int]]:
    """Count effective training labels in internal-class order.

    Uses ``list_image_info()`` so the counts match the examples LightlyTrain
    actually trains on (missing/unsupported images, ignored classes and empty
    remaining labels already filtered).

    Returns:
        (counts, total_images, ordered_user_class_ids) where ``counts`` is
        aligned with internal-class order.
    """
    mapping = label_helpers.get_class_id_to_internal_class_id_mapping(
        class_ids=train_dataset_args.classes.keys(),
        ignore_classes=train_dataset_args.ignore_classes,
    )
    ordered_user_ids = sorted(mapping, key=lambda class_id: mapping[class_id])
    counts_by_user: dict[int, int] = {class_id: 0 for class_id in ordered_user_ids}

    total = 0
    delimiter = train_dataset_args.label_delimiter
    for info in train_dataset_args.list_image_info():
        total += 1
        class_id_str = info.get("class_id", "")
        if not class_id_str:
            continue
        # Deduplicate within one image so a repeated label counts once.
        seen: set[int] = set()
        for part in class_id_str.split(delimiter):
            part = part.strip()
            if not part:
                continue
            try:
                class_id = int(part)
            except ValueError:
                continue
            if class_id in counts_by_user and class_id not in seen:
                seen.add(class_id)
                counts_by_user[class_id] += 1

    counts = [counts_by_user[class_id] for class_id in ordered_user_ids]
    return counts, total, ordered_user_ids


def compute_auto_multiclass_weights(counts: list[int]) -> list[float]:
    """Inverse class frequency rescaled to mean 1.0.

    Zero-example classes get a neutral weight of 1.0 so they stay finite and
    are not suppressed if they appear in validation. The mean over the full
    vector (including neutral entries) is 1.0.
    """
    num_classes = len(counts)
    if num_classes == 0:
        return []
    nonzero_inverse = [1.0 / c for c in counts if c > 0]
    if not nonzero_inverse:
        return [1.0] * num_classes
    mean_inverse = sum(nonzero_inverse) / len(nonzero_inverse)
    return [(1.0 / c) / mean_inverse if c > 0 else 1.0 for c in counts]


def compute_auto_multilabel_pos_weights(counts: list[int], total: int) -> list[float]:
    """Per-class ``neg / pos`` weights for ``BCEWithLogitsLoss``.

    A class with zero positive training examples gets a neutral ``pos_weight``
    of 1.0 (finite, does not suppress the class if it appears in validation).
    """
    if total <= 0:
        return [1.0] * len(counts)
    weights: list[float] = []
    for positive in counts:
        if positive <= 0:
            weights.append(1.0)
        else:
            weights.append((total - positive) / positive)
    return weights


def validate_manual_weights(
    manual: dict[str, float],
    classes: dict[int, str],
    ignore_classes: set[int] | None,
) -> list[float]:
    """Validate a manual name->weight mapping and return internal-order weights.

    The mapping must cover exactly the included (non-ignored) classes. Unknown
    names and missing classes both raise a clear ``ValueError``. Values must be
    finite and non-negative.
    """
    ignore = set() if ignore_classes is None else set(ignore_classes)
    included: dict[int, str] = {
        class_id: name for class_id, name in classes.items() if class_id not in ignore
    }
    included_names = set(included.values())
    # Detect ambiguous duplicate class names among included classes.
    if len(included_names) != len(included):
        raise ValueError(
            "Manual `class_weights` cannot be resolved because included class "
            "names are not unique. Class names must be unique to key weights "
            f"by name. Included classes: {sorted(included.values())}."
        )

    unknown = sorted(set(manual.keys()) - included_names)
    if unknown:
        raise ValueError(
            f"Unknown class name(s) in `class_weights`: {unknown}. "
            f"Expected only included class names: {sorted(included_names)}."
        )
    missing = sorted(included_names - set(manual.keys()))
    if missing:
        raise ValueError(
            f"`class_weights` is missing weight(s) for included class(es): "
            f"{missing}. Provide a weight for every included class: "
            f"{sorted(included_names)}."
        )
    for name, value in manual.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid weight for class '{name}': {value!r}. "
                "Weights must be finite non-negative numbers."
            )
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(
                f"Invalid weight for class '{name}': {value!r}. "
                "Weights must be finite non-negative numbers."
            )

    ordered_user_ids = internal_ordered_class_ids(
        classes=classes, ignore_classes=ignore_classes
    )
    user_to_name = {class_id: classes[class_id] for class_id in ordered_user_ids}
    return [float(manual[user_to_name[class_id]]) for class_id in ordered_user_ids]


def resolve_class_weights(
    class_weights: Literal["auto"] | dict[str, float] | None,
    data_args: ImageClassificationDataArgs,
) -> dict[str, float] | None:
    """Resolve ``class_weights`` to a serializable name->weight dict or None.

    - ``None`` preserves current unweighted behavior.
    - ``"auto"`` counts the training split via ``list_image_info()`` and
      applies the task-specific formula.
    - A manual dict is validated (unknown/missing names, finite values) and
      returned as plain floats.
    """
    if class_weights is None:
        return None
    if isinstance(class_weights, str):
        if class_weights != "auto":
            raise ValueError(
                f"Invalid `class_weights`: {class_weights!r}. "
                "Must be 'auto', a dict from class name to weight, or None."
            )
        train_dataset_args = data_args.get_train_args()
        counts, total, ordered_user_ids = count_training_labels(train_dataset_args)
        ordered_names = [data_args.classes[cid] for cid in ordered_user_ids]
        if data_args.classification_task == "multiclass":
            weights = compute_auto_multiclass_weights(counts)
        elif data_args.classification_task == "multilabel":
            weights = compute_auto_multilabel_pos_weights(counts, total)
        else:
            raise ValueError(
                f"Unsupported classification task: {data_args.classification_task}"
            )
        return {name: float(w) for name, w in zip(ordered_names, weights)}

    if isinstance(class_weights, dict):
        ordered = validate_manual_weights(
            manual=class_weights,
            classes=data_args.classes,
            ignore_classes=data_args.ignore_classes,
        )
        ordered_user_ids = internal_ordered_class_ids(
            classes=data_args.classes, ignore_classes=data_args.ignore_classes
        )
        ordered_names = [data_args.classes[cid] for cid in ordered_user_ids]
        return {name: float(w) for name, w in zip(ordered_names, ordered)}

    raise ValueError(
        f"Invalid `class_weights`: {class_weights!r}. "
        "Must be 'auto', a dict from class name to weight, or None."
    )


def resolved_to_tensor(
    resolved: dict[str, float] | None,
    data_args: ImageClassificationDataArgs,
) -> Tensor | None:
    """Convert a resolved name->weight dict to a tensor in internal order."""
    if resolved is None:
        return None
    ordered = validate_manual_weights(
        manual=resolved,
        classes=data_args.classes,
        ignore_classes=data_args.ignore_classes,
    )
    return torch.tensor(ordered, dtype=torch.float32)
