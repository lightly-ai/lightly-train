#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import Field, model_validator
from typing_extensions import Self

from lightly_train._configs.config import PydanticConfig

# Visibility convention, shared by COCO and YOLO pose.
VISIBILITY_UNLABELED = 0
VISIBILITY_OCCLUDED = 1
VISIBILITY_VISIBLE = 2
VISIBILITIES = (VISIBILITY_UNLABELED, VISIBILITY_OCCLUDED, VISIBILITY_VISIBLE)


class KeypointSetArgs(PydanticConfig):
    """A keypoint set: how many keypoints, their names, how they relate.

    Only ``num_keypoints`` is required. No format carries every field: COCO declares
    names and a skeleton, YOLO pose the count and optionally flip pairs and names.
    Fields that neither the dataset nor the user provides stay ``None``. Nothing is
    defaulted here; each consumer decides what to do without a value.

    Attributes:
        num_keypoints:
            Keypoints per instance. Every instance has this many; unlabeled ones are
            marked via their visibility flag.
        names:
            Name of each keypoint, in keypoint order.
        sigmas:
            Per-keypoint OKS standard deviations. A property of the keypoint set, not
            of a dataset. No format stores them, so they come from the user only.
        flip_idx:
            Keypoint permutation for a horizontal flip. ``flip_idx[i]`` is the keypoint
            that takes position ``i``, so a left/right pair ``(1, 2)`` gives
            ``flip_idx[1] == 2`` and ``flip_idx[2] == 1``. Must be a permutation and its
            own inverse.
        skeleton:
            Connected keypoint index pairs, for visualization. Zero-indexed, unlike
            COCO's one-indexed ``skeleton``.
    """

    num_keypoints: int
    names: list[str] | None = None
    # strict=False: PydanticConfig sets strict=True, which rejects an int for a float
    # field. A YAML `sigmas: [1, 0.5]` would fail on its first element.
    sigmas: list[float] | None = Field(default=None, strict=False)
    flip_idx: list[int] | None = None
    # list, not tuple: strict=True does not coerce lists to tuples, and JSON and YAML
    # only give us lists.
    skeleton: list[list[int]] | None = None

    @model_validator(mode="after")
    def validate_keypoint_set(self) -> Self:
        if self.num_keypoints < 1:
            raise ValueError(
                f"Expected 'num_keypoints' to be at least 1, got {self.num_keypoints}."
            )

        for field_name in ("names", "sigmas", "flip_idx"):
            value = getattr(self, field_name)
            if value is not None and len(value) != self.num_keypoints:
                raise ValueError(
                    f"Expected '{field_name}' to have {self.num_keypoints} entries to "
                    f"match 'num_keypoints', got {len(value)}."
                )

        if self.sigmas is not None:
            non_positive = [sigma for sigma in self.sigmas if sigma <= 0]
            if non_positive:
                raise ValueError(
                    f"Expected all 'sigmas' to be positive, got {non_positive}."
                )

        if self.flip_idx is not None:
            # Must be a permutation, not just indices in range: anything else
            # duplicates one keypoint and drops another on every flip.
            if sorted(self.flip_idx) != list(range(self.num_keypoints)):
                raise ValueError(
                    f"Expected 'flip_idx' to be a permutation of "
                    f"[0, {self.num_keypoints - 1}], got {self.flip_idx}."
                )
            # Flipping twice must restore the original order, so the permutation must
            # be its own inverse.
            not_involutive = [
                i
                for i in range(self.num_keypoints)
                if self.flip_idx[self.flip_idx[i]] != i
            ]
            if not_involutive:
                raise ValueError(
                    f"Expected 'flip_idx' to be its own inverse, so that flipping twice "
                    f"restores the original keypoint order, but "
                    f"flip_idx[flip_idx[i]] != i for i in {not_involutive}. "
                    f"Got flip_idx={self.flip_idx}."
                )

        if self.skeleton is not None:
            for pair in self.skeleton:
                if len(pair) != 2:
                    raise ValueError(
                        f"Expected every 'skeleton' entry to connect exactly two "
                        f"keypoints, got {pair}."
                    )
                if any(not 0 <= i < self.num_keypoints for i in pair):
                    raise ValueError(
                        f"Expected every 'skeleton' index to be in "
                        f"[0, {self.num_keypoints - 1}], got {pair}. Note that "
                        f"'skeleton' is zero-indexed, while the COCO format uses "
                        f"one-indexed skeletons."
                    )
                if pair[0] == pair[1]:
                    raise ValueError(
                        f"Expected every 'skeleton' entry to connect two different "
                        f"keypoints, got {pair}."
                    )

        return self


def _check_agrees(
    field_name: str, from_dataset: Any, from_user: Any, dataset_description: str
) -> Any:
    """Returns the single value for a field the dataset and the user may both set.

    Neither source wins. If both give a value they must agree.
    """
    if from_dataset is None:
        return from_user
    if from_user is None:
        return from_dataset
    if from_dataset != from_user:
        raise ValueError(
            f"Conflicting values for '{field_name}': {dataset_description} says "
            f"{from_dataset!r}, while the 'keypoints' argument says {from_user!r}. "
            f"Remove one of them or make them agree."
        )
    return from_dataset


def resolve_keypoint_set_from_parts(
    num_keypoints: int,
    names: list[str] | None,
    flip_idx: list[int] | None,
    skeleton: list[list[int]] | None,
    keypoints: KeypointSetArgs | None,
    dataset_description: str,
) -> KeypointSetArgs:
    """Combines a keypoint set read from a dataset with one declared by the user.

    Args:
        num_keypoints:
            Keypoint count from the dataset.
        names:
            Keypoint names from the dataset, if declared.
        flip_idx:
            Flip permutation from the dataset, if declared.
        skeleton:
            Zero-indexed skeleton from the dataset, if declared.
        keypoints:
            Keypoint set declared by the user, if any.
        dataset_description:
            Where the dataset values came from, for error messages.
    """
    if keypoints is not None and keypoints.num_keypoints != num_keypoints:
        raise ValueError(
            f"Conflicting values for 'num_keypoints': {dataset_description} has "
            f"{num_keypoints} keypoints, while the 'keypoints' argument declares "
            f"{keypoints.num_keypoints}."
        )
    return KeypointSetArgs(
        num_keypoints=num_keypoints,
        names=_check_agrees(
            "names",
            names,
            None if keypoints is None else keypoints.names,
            dataset_description,
        ),
        sigmas=None if keypoints is None else keypoints.sigmas,
        flip_idx=_check_agrees(
            "flip_idx",
            flip_idx,
            None if keypoints is None else keypoints.flip_idx,
            dataset_description,
        ),
        skeleton=_check_agrees(
            "skeleton",
            skeleton,
            None if keypoints is None else keypoints.skeleton,
            dataset_description,
        ),
    )


def resolve_yolo_keypoint_set(
    kpt_shape: list[int],
    flip_idx: list[int] | None,
    kpt_names: dict[int, list[str]] | None,
    keypoints: KeypointSetArgs | None,
    included_class_ids: Iterable[int],
) -> KeypointSetArgs:
    """Resolves the keypoint set of a YOLO pose dataset.

    - ``kpt_shape`` gives the keypoint count.
    - ``flip_idx`` and ``kpt_names`` are optional.
    - Sigmas and skeleton are never declared by the format.

    ``kpt_names`` is per class, but one keypoint set is used for all classes, so all
    included classes must declare the same names.
    """
    num_keypoints, _ = validate_kpt_shape(kpt_shape)

    names = None
    if kpt_names is not None:
        included = set(included_class_ids)
        declared = {
            class_id: class_names
            for class_id, class_names in kpt_names.items()
            if class_id in included
        }
        distinct = {tuple(class_names) for class_names in declared.values()}
        if len(distinct) > 1:
            details = ", ".join(
                f"class {class_id}: {class_names}"
                for class_id, class_names in sorted(declared.items())
            )
            raise ValueError(
                f"Expected all classes to declare the same keypoint names in "
                f"'kpt_names', because a single keypoint set is used for all classes, "
                f"but got {details}. Remove the differing entries or exclude the "
                f"classes via 'ignore_classes'."
            )
        if distinct:
            names = list(next(iter(distinct)))

    return resolve_keypoint_set_from_parts(
        num_keypoints=num_keypoints,
        names=names,
        flip_idx=flip_idx,
        skeleton=None,
        keypoints=keypoints,
        dataset_description="the dataset config",
    )


def validate_kpt_shape(kpt_shape: list[int]) -> tuple[int, int]:
    """Validates a YOLO ``kpt_shape`` and returns it as (num_keypoints, num_dims)."""
    if len(kpt_shape) != 2:
        raise ValueError(
            f"Expected 'kpt_shape' to have two entries, [num_keypoints, num_dims], "
            f"got {kpt_shape}."
        )
    num_keypoints, num_dims = kpt_shape
    if num_keypoints < 1:
        raise ValueError(
            f"Expected the number of keypoints in 'kpt_shape' to be at least 1, got "
            f"{num_keypoints}."
        )
    if num_dims not in (2, 3):
        raise ValueError(
            f"Expected the number of dimensions in 'kpt_shape' to be 2 for (x, y) or "
            f"3 for (x, y, visibility), got {num_dims}."
        )
    return num_keypoints, num_dims


def resolve_coco_keypoint_set(
    categories: list[dict[str, Any]],
    included_class_ids: Iterable[int],
    keypoints: KeypointSetArgs | None,
) -> KeypointSetArgs:
    """Resolves the keypoint set of a COCO keypoint dataset.

    - Names and skeleton come from the ``categories`` entries.
    - Sigmas and flip pairs are never declared by the format; they come from
      ``keypoints``.

    One keypoint set is used for all classes, so all included categories that declare
    keypoints must declare the same ones. Categories without keypoints are ignored;
    their annotations are read as having no labeled keypoints.
    """
    included = set(included_class_ids)
    declared = {
        category["id"]: category
        for category in categories
        if category["id"] in included and category.get("keypoints")
    }

    distinct = {tuple(category["keypoints"]) for category in declared.values()}
    if len(distinct) > 1:
        details = ", ".join(
            f"category {category_id} ({declared[category_id].get('name')!r}): "
            f"{declared[category_id]['keypoints']}"
            for category_id in sorted(declared)
        )
        raise ValueError(
            f"Expected all categories to declare the same keypoints, because a single "
            f"keypoint set is used for all classes, but got {details}. Exclude the "
            f"differing categories via 'ignore_classes'."
        )

    if not distinct:
        if keypoints is None:
            raise ValueError(
                "No category in the annotations declares a 'keypoints' field, so the "
                "keypoint set cannot be determined from the dataset. Pass it explicitly "
                "via the 'keypoints' argument."
            )
        return keypoints

    names = list(next(iter(distinct)))
    num_keypoints = len(names)

    # All included categories declare the same keypoints, so any of them can supply
    # the skeleton. Take the first with a non-empty one.
    skeleton_one_indexed = next(
        (
            category["skeleton"]
            for _, category in sorted(declared.items())
            if category.get("skeleton")
        ),
        None,
    )
    skeleton = None
    if skeleton_one_indexed is not None:
        skeleton = []
        for pair in skeleton_one_indexed:
            if len(pair) != 2:
                raise ValueError(
                    f"Expected every 'skeleton' entry in the annotations to connect "
                    f"exactly two keypoints, got {pair}."
                )
            if any(not 1 <= i <= num_keypoints for i in pair):
                raise ValueError(
                    f"Expected every 'skeleton' index in the annotations to be in "
                    f"[1, {num_keypoints}], got {pair}. The COCO format uses "
                    f"one-indexed skeletons."
                )
            skeleton.append([pair[0] - 1, pair[1] - 1])

    return resolve_keypoint_set_from_parts(
        num_keypoints=num_keypoints,
        names=names,
        flip_idx=None,
        skeleton=skeleton,
        keypoints=keypoints,
        dataset_description="the annotations file",
    )


def bbox_from_keypoints(
    keypoints_xy: list[list[float]], visibility: list[int]
) -> list[float] | None:
    """Returns the tight bounding box around all labeled keypoints.

    Args:
        keypoints_xy:
            Keypoint coordinates as [[x, y], ...].
        visibility:
            Visibility flag per keypoint. Keypoints with visibility 0 are ignored.

    Returns:
        (x_center, y_center, width, height) in the keypoints' coordinate system, or
        None if no keypoint is labeled.
    """
    labeled = [
        point
        for point, vis in zip(keypoints_xy, visibility)
        if vis != VISIBILITY_UNLABELED
    ]
    if not labeled:
        return None
    xs = [point[0] for point in labeled]
    ys = [point[1] for point in labeled]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [
        (x_min + x_max) / 2.0,
        (y_min + y_max) / 2.0,
        x_max - x_min,
        y_max - y_min,
    ]
