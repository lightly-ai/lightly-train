#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Iterable


def get_class_id_to_internal_class_id_mapping(
    class_ids: Iterable[int], ignore_classes: set[int] | None
) -> dict[int, int]:
    """Returns mapping from class id to new class index.

    Skips ignored_classes. We use the class index internally as "internal class id" as
    some models require class ids to be in [0, num_classes - 1].
    """
    ignore_classes = ignore_classes or set()
    return {
        class_id: i
        for i, class_id in enumerate(
            class_id for class_id in class_ids if class_id not in ignore_classes
        )
    }


def internal_ordered_class_ids(
    class_ids: Iterable[int], ignore_classes: set[int] | None
) -> list[int]:
    """Returns the non-ignored class ids in internal class id order.

    The order matches `get_class_id_to_internal_class_id_mapping` and therefore the
    dataset mapping and the model head. No sorting is applied.
    """
    mapping = get_class_id_to_internal_class_id_mapping(
        class_ids=class_ids, ignore_classes=ignore_classes
    )
    return sorted(mapping, key=lambda class_id: mapping[class_id])
