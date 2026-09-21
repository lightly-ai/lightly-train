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

# A keypoint is either not labeled, labeled but not visible, or labeled and visible.
# This is the COCO convention and the YOLO pose convention alike.
VISIBILITY_UNLABELED = 0
VISIBILITY_OCCLUDED = 1
VISIBILITY_VISIBLE = 2
VISIBILITIES = (VISIBILITY_UNLABELED, VISIBILITY_OCCLUDED, VISIBILITY_VISIBLE)


class KeypointSetArgs(PydanticConfig):
    """Describes a keypoint set: how many keypoints, what they are called, and how
    they relate to each other.

    Every field except ``num_keypoints`` is optional because no dataset format carries
    all of them. The COCO format declares keypoint names and a skeleton but no OKS
    sigmas and no flip pairs; the YOLO pose format declares the keypoint count and
    optionally flip pairs and names, but no sigmas and no skeleton. Fields that neither
    the dataset nor the user provides stay ``None``. They are deliberately not defaulted
    here: whichever consumer needs one (the OKS metric needs ``sigmas``, the horizontal
    flip transform needs ``flip_idx``) decides what to do without a value.

    Attributes:
        num_keypoints:
            Number of keypoints per instance. Every instance has exactly this many
            keypoints; unlabeled ones are marked via their visibility flag.
        names:
            Name of each keypoint, in keypoint order.
        sigmas:
            Per-keypoint OKS standard deviations. These describe how much a keypoint's
            position varies between human annotators and are therefore a property of the
            keypoint set, not of any single dataset. No dataset format stores them, so
            they can only come from the user.
        flip_idx:
            Permutation applied to the keypoints when an image is flipped horizontally.
            ``flip_idx[i]`` is the keypoint that takes position ``i`` after the flip, so
            a left/right pair ``(1, 2)`` appears as ``flip_idx[1] == 2`` and
            ``flip_idx[2] == 1``. Must be a permutation and its own inverse.
        skeleton:
            Pairs of keypoint indices that are connected, used for visualization.
            Zero-indexed, unlike the COCO format's one-indexed ``skeleton`` field.
    """

    num_keypoints: int
    names: list[str] | None = None
    # strict=False because PydanticConfig sets strict=True, under which an int is not
    # accepted for a float field. A YAML list such as `sigmas: [1, 0.5]` would otherwise
    # be rejected because of its first element.
    sigmas: list[float] | None = Field(default=None, strict=False)
    flip_idx: list[int] | None = None
    # list[list[int]] rather than list[tuple[int, int]]: under strict=True a list is not
    # coerced to a tuple, and JSON and YAML only ever give us lists.
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
            # A permutation, not merely indices in range: a non-permutation would
            # silently duplicate one keypoint and drop another on every flip.
            if sorted(self.flip_idx) != list(range(self.num_keypoints)):
                raise ValueError(
                    f"Expected 'flip_idx' to be a permutation of "
                    f"[0, {self.num_keypoints - 1}], got {self.flip_idx}."
                )
            # Flipping twice must be the identity, so the permutation must be its own
            # inverse. A violation is always a mistake in the dataset config.
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
    """Returns the single value for a field that both the dataset and the user may set.

    Neither source takes precedence: if both provide a value they must agree, otherwise
    it is impossible to tell which one the user meant.
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
            Number of keypoints as determined by the dataset.
        names:
            Keypoint names from the dataset, if it declares any.
        flip_idx:
            Flip permutation from the dataset, if it declares one.
        skeleton:
            Zero-indexed skeleton from the dataset, if it declares one.
        keypoints:
            Keypoint set declared by the user, if any.
        dataset_description:
            Human readable description of where the dataset values came from, used in
            error messages.
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

    The dataset contributes the keypoint count via ``kpt_shape``, and optionally the
    flip permutation and the keypoint names. It never declares sigmas or a skeleton.

    ``kpt_names`` is declared per class, but a single keypoint set is used for all
    classes, so all included classes must declare the same names.
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

    The keypoint names and the skeleton are read from the ``categories`` entries. A
    single keypoint set is used for all classes, so all included categories that declare
    keypoints must declare the same ones. Categories without keypoints are ignored here;
    their annotations are read as having no labeled keypoints.

    The COCO format declares neither sigmas nor flip pairs, so those can only come from
    ``keypoints``.
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

    # All included categories declare the same keypoints, so any of them can supply the
    # skeleton. Prefer the first one that has a non-empty skeleton.
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
        The bounding box as (x_center, y_center, width, height) in the same coordinate
        system as the keypoints, or None if no keypoint is labeled.
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
