#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import math
from typing import Literal

import torch
from torch import Tensor

from lightly_train._data import label_helpers
from lightly_train._data.image_classification_dataset import (
    ImageClassificationDataArgs,
    ImageClassificationDataset,
)

logger = logging.getLogger(__name__)

# Largest pos_weight that "auto" can return. Very rare classes would otherwise get a
# huge weight, which inflates the loss and its gradients. Manual weights are not
# clamped.
MAX_AUTO_POS_WEIGHT = 100.0


def _ordered_class_ids(data_args: ImageClassificationDataArgs) -> list[int]:
    """Included class ids in internal class id order."""
    return label_helpers.internal_ordered_class_ids(
        class_ids=data_args.classes.keys(), ignore_classes=data_args.ignore_classes
    )


def validate_unique_class_names(data_args: ImageClassificationDataArgs) -> None:
    """Raise if two included classes share a name.

    Weights are keyed by class name, so duplicate names would silently drop entries.
    """
    names = list(data_args.included_classes.values())
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "`class_weights` requires unique class names because weights are keyed "
            f"by name, but these included class names are duplicated: {duplicates}. "
            "Rename the classes or drop the duplicates with `ignore_classes`."
        )


def compute_auto_multiclass_weights(counts: list[int]) -> list[float]:
    """Inverse class frequency rescaled to mean 1.0.

    Zero-example classes get a neutral weight of 1.0 so they stay finite and
    are not suppressed if they appear in validation. The mean over the full
    vector (including neutral entries) is 1.0.

    These weights are not clamped. The mean-1.0 rescaling already keeps them below
    roughly the number of classes.
    """
    num_classes = len(counts)
    if num_classes == 0:
        return []
    nonzero_inverse = [1.0 / c for c in counts if c > 0]
    if not nonzero_inverse:
        return [1.0] * num_classes
    mean_inverse = sum(nonzero_inverse) / len(nonzero_inverse)
    return [(1.0 / c) / mean_inverse if c > 0 else 1.0 for c in counts]


def compute_auto_multilabel_pos_weights(
    counts: list[int],
    total: int,
    class_names: list[str] | None = None,
) -> list[float]:
    """Per-class ``neg / pos`` weights for ``BCEWithLogitsLoss``.

    A class gets a neutral ``pos_weight`` of 1.0 if it has no positive training
    examples, or if it is in every training image. The second case would give
    ``neg / pos == 0.0``, which stops the class from getting any gradient.

    Weights are clamped to ``MAX_AUTO_POS_WEIGHT``. Clamped classes are logged.
    """
    if total <= 0:
        return [1.0] * len(counts)
    weights: list[float] = []
    clamped: list[str] = []
    for index, positive in enumerate(counts):
        negative = total - positive
        if positive <= 0 or negative <= 0:
            weights.append(1.0)
            continue
        weight = negative / positive
        if weight > MAX_AUTO_POS_WEIGHT:
            weight = MAX_AUTO_POS_WEIGHT
            name = (
                class_names[index]
                if class_names is not None and index < len(class_names)
                else str(index)
            )
            clamped.append(name)
        weights.append(weight)

    if clamped:
        logger.warning(
            f"Automatic `pos_weight` was clamped to {MAX_AUTO_POS_WEIGHT} for very "
            f"rare class(es): {sorted(clamped)}. Larger weights inflate the loss and "
            "its gradients and can make training unstable. Pass an explicit "
            "`class_weights` dict if you want larger weights for these classes."
        )
    return weights


def validate_manual_weights(
    manual: dict[str, float],
    data_args: ImageClassificationDataArgs,
) -> list[float]:
    """Validate a manual name->weight mapping and return internal-order weights.

    The mapping must cover exactly the included (non-ignored) classes. Unknown
    names and missing classes both raise a clear ``ValueError``. Values must be
    finite and non-negative.

    Expects unique class names. Call ``validate_unique_class_names`` first.
    """
    included_names = set(data_args.included_classes.values())

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

    ordered_class_ids = _ordered_class_ids(data_args)
    return [
        float(manual[data_args.classes[class_id]]) for class_id in ordered_class_ids
    ]


def resolve_class_weights(
    class_weights: Literal["auto"] | dict[str, float] | None,
    data_args: ImageClassificationDataArgs,
    train_dataset: ImageClassificationDataset | None = None,
) -> dict[str, float] | None:
    """Resolve ``class_weights`` to a serializable name->weight dict or None.

    - ``None`` preserves current unweighted behavior.
    - ``"auto"`` counts the training dataset and applies the task-specific formula.
      ``train_dataset`` is required in this case.
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
        if train_dataset is None:
            raise ValueError(
                "`class_weights='auto'` needs the training dataset to count classes, "
                "but no dataset was given."
            )
        validate_unique_class_names(data_args)
        counts = train_dataset.count_class_occurrences()
        ordered_names = [
            data_args.classes[class_id] for class_id in _ordered_class_ids(data_args)
        ]
        if data_args.classification_task == "multiclass":
            weights = compute_auto_multiclass_weights(counts)
        elif data_args.classification_task == "multilabel":
            weights = compute_auto_multilabel_pos_weights(
                counts, total=len(train_dataset), class_names=ordered_names
            )
        else:
            raise ValueError(
                f"Unsupported classification task: {data_args.classification_task}"
            )
        return {name: float(w) for name, w in zip(ordered_names, weights)}

    if isinstance(class_weights, dict):
        validate_unique_class_names(data_args)
        ordered = validate_manual_weights(manual=class_weights, data_args=data_args)
        ordered_names = [
            data_args.classes[class_id] for class_id in _ordered_class_ids(data_args)
        ]
        return {name: float(w) for name, w in zip(ordered_names, ordered)}

    raise ValueError(
        f"Invalid `class_weights`: {class_weights!r}. "
        "Must be 'auto', a dict from class name to weight, or None."
    )


def resolved_to_tensor(
    resolved: Literal["auto"] | dict[str, float] | None,
    data_args: ImageClassificationDataArgs,
) -> Tensor | None:
    """Turn an already resolved name->weight dict into a tensor in internal order.

    Expects the output of ``resolve_class_weights``.
    """
    if resolved is None:
        return None
    if isinstance(resolved, str):
        raise ValueError(
            f"`class_weights` must be resolved before building the loss, got "
            f"{resolved!r}. `resolve_auto` should have replaced it with a class name "
            "to weight mapping."
        )

    ordered_class_ids = _ordered_class_ids(data_args)
    ordered: list[float] = []
    for class_id in ordered_class_ids:
        name = data_args.classes[class_id]
        if name not in resolved:
            raise ValueError(
                f"Resolved `class_weights` is missing a weight for class '{name}'. "
                f"Expected weights for: "
                f"{[data_args.classes[cid] for cid in ordered_class_ids]}."
            )
        ordered.append(float(resolved[name]))
    return torch.tensor(ordered, dtype=torch.float32)
