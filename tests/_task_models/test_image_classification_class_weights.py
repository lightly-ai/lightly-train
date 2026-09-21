#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch

from lightly_train._data.image_classification_dataset import (
    ImageClassificationDataArgs,
    ImageClassificationMulticlassDataArgs,
    ImageClassificationMultilabelDataArgs,
)
from lightly_train._task_models import image_classification_class_weights as cw

from .. import helpers


def _data_args(
    classes: dict[int, str], ignore: set[int] | None
) -> ImageClassificationMulticlassDataArgs:
    return ImageClassificationMulticlassDataArgs(
        train="train", val="val", classes=classes, ignore_classes=ignore
    )


def _resolve_auto(data_args: ImageClassificationDataArgs) -> dict[str, float] | None:
    return cw.resolve_class_weights(
        "auto",
        data_args,
        train_dataset=helpers.get_image_classification_train_dataset(data_args),
    )


def test_validate_unique_class_names__duplicates_raise() -> None:
    with pytest.raises(ValueError, match="requires unique class names"):
        cw.validate_unique_class_names(_data_args({0: "cat", 1: "cat"}, None))


def test_validate_unique_class_names__duplicate_ignored_is_fine() -> None:
    cw.validate_unique_class_names(_data_args({0: "cat", 1: "cat"}, {1}))


def test_compute_auto_multiclass_weights__inverse_frequency_mean_one() -> None:
    weights = cw.compute_auto_multiclass_weights([90, 10])
    assert math.isfinite(weights[0]) and math.isfinite(weights[1])
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    # Rare class gets higher weight; ratio equals inverse-count ratio.
    assert weights[1] / weights[0] == pytest.approx(9.0)
    # Balanced classes give uniform weights.
    balanced = cw.compute_auto_multiclass_weights([5, 5, 5])
    assert balanced == pytest.approx([1.0, 1.0, 1.0])


def test_compute_auto_multiclass_weights__zero_example_class_is_finite() -> None:
    weights = cw.compute_auto_multiclass_weights([5, 0])
    assert all(math.isfinite(w) for w in weights)
    assert weights[1] == pytest.approx(1.0)
    assert sum(weights) / len(weights) == pytest.approx(1.0)


def test_compute_auto_multilabel_pos_weights__zero_positive_is_neutral() -> None:
    weights = cw.compute_auto_multilabel_pos_weights([0, 2], total=4)
    assert all(math.isfinite(w) for w in weights)
    # Zero-positive class must not be suppressed with 0.0.
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx((4 - 2) / 2)


def test_compute_auto_multilabel_pos_weights__class_in_every_image_is_neutral() -> None:
    # "neg / pos" would be 0.0 here, which stops the class from getting a gradient.
    weights = cw.compute_auto_multilabel_pos_weights([4, 2], total=4)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(1.0)


def test_compute_auto_multilabel_pos_weights__zero_pos_weight_disables_class() -> None:
    # Why the guard above matters: pos_weight=0.0 removes the gradient completely.
    logits = torch.zeros(4, 2, requires_grad=True)
    targets = torch.ones(4, 2)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.0, 1.0]))(
        logits, targets
    )
    loss.backward()
    assert logits.grad is not None
    assert torch.all(logits.grad[:, 0] == 0.0)
    assert torch.all(logits.grad[:, 1] != 0.0)


def test_compute_auto_multilabel_pos_weights__exact_neg_over_pos() -> None:
    weights = cw.compute_auto_multilabel_pos_weights([1, 50_000], total=100_000)
    assert weights == pytest.approx([99_999.0, 1.0])


def test_validate_manual_weights__internal_order() -> None:
    classes = {3: "cat", 7: "car", 12: "dog"}
    ordered = cw.validate_manual_weights(
        {"dog": 3.0, "cat": 1.0, "car": 2.0}, data_args=_data_args(classes, None)
    )
    # Internal order follows user dict order: cat(3)->0, car(7)->1, dog(12)->2.
    assert ordered == [1.0, 2.0, 3.0]


def test_validate_manual_weights__unknown_class_name_raises() -> None:
    classes = {0: "cat", 1: "dog"}
    with pytest.raises(ValueError, match="Unknown class name"):
        cw.validate_manual_weights(
            {"cat": 1.0, "bird": 2.0}, data_args=_data_args(classes, None)
        )


def test_validate_manual_weights__missing_class_raises() -> None:
    classes = {0: "cat", 1: "dog"}
    with pytest.raises(ValueError, match="missing weight"):
        cw.validate_manual_weights({"cat": 1.0}, data_args=_data_args(classes, None))


def test_validate_manual_weights__ignored_class_names_rejected() -> None:
    classes = {0: "cat", 1: "car", 2: "dog"}
    with pytest.raises(ValueError, match="Unknown class name"):
        cw.validate_manual_weights(
            {"cat": 1.0, "car": 1.0, "dog": 1.0}, data_args=_data_args(classes, {1})
        )


def test_validate_manual_weights__ignore_middle_class_ordering() -> None:
    classes = {3: "cat", 7: "car", 12: "dog"}
    ordered = cw.validate_manual_weights(
        {"dog": 5.0, "cat": 2.0}, data_args=_data_args(classes, {7})
    )
    assert ordered == [2.0, 5.0]


def test_resolve_class_weights__none() -> None:
    assert cw.resolve_class_weights(None, _data_args({0: "cat"}, None)) is None


def test_resolve_class_weights__auto_without_dataset_raises() -> None:
    with pytest.raises(ValueError, match="needs the training dataset"):
        cw.resolve_class_weights("auto", _data_args({0: "cat", 1: "dog"}, None))


def test_resolve_class_weights__auto_with_duplicate_names_raises(
    tmp_path: Path,
) -> None:
    # Duplicate names are legal in the data args but cannot be keyed by name.
    classes = {0: "cat", 1: "cat"}
    train_dir = tmp_path / "train" / "cat"
    train_dir.mkdir(parents=True, exist_ok=True)
    helpers.create_images(train_dir, files=1)
    data_args = ImageClassificationMulticlassDataArgs(
        train=tmp_path / "train", val=tmp_path / "train", classes=classes
    )
    with pytest.raises(ValueError, match="requires unique class names"):
        _resolve_auto(data_args)


def test_resolve_class_weights__auto_uses_train_split_not_val(tmp_path: Path) -> None:
    # Train is imbalanced (3 cat, 1 dog); val is balanced differently.
    classes = {0: "cat", 1: "dog"}
    for split, counts in [
        ("train", {"cat": 3, "dog": 1}),
        ("val", {"cat": 1, "dog": 3}),
    ]:
        for cls_name, n in counts.items():
            d = tmp_path / split / cls_name
            d.mkdir(parents=True, exist_ok=True)
            helpers.create_images(d, files=n)
    data_args = ImageClassificationMulticlassDataArgs(
        train=tmp_path / "train", val=tmp_path / "val", classes=classes
    )
    resolved = _resolve_auto(data_args)
    assert resolved is not None
    # counts cat=3, dog=1 -> raw 1/3, 1/1, mean 2/3 -> weights 0.5, 1.5
    assert resolved["cat"] == pytest.approx(0.5)
    assert resolved["dog"] == pytest.approx(1.5)


def test_resolve_class_weights__auto_multilabel_neg_over_pos(tmp_path: Path) -> None:
    classes = {0: "cat", 1: "car", 2: "dog"}
    # 4 train images: cat in 3, car in 1, dog in 2.
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        helpers.create_image(train_dir / f"img{i}.png")
    val_dir = tmp_path / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        helpers.create_image(val_dir / f"img{i}.png")
    train_csv = tmp_path / "train.csv"
    with train_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label"])
        w.writeheader()
        w.writerow({"image_path": str(train_dir / "img0.png"), "label": "cat"})
        w.writerow({"image_path": str(train_dir / "img1.png"), "label": "cat,car"})
        w.writerow({"image_path": str(train_dir / "img2.png"), "label": "cat,dog"})
        w.writerow({"image_path": str(train_dir / "img3.png"), "label": "dog"})
    val_csv = tmp_path / "val.csv"
    with val_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label"])
        w.writeheader()
        w.writerow({"image_path": str(val_dir / "img0.png"), "label": "cat"})
        w.writerow({"image_path": str(val_dir / "img1.png"), "label": "dog"})
    data_args = ImageClassificationMultilabelDataArgs(
        train=train_csv, val=val_csv, classes=classes
    )
    resolved = _resolve_auto(data_args)
    assert resolved is not None
    # total=4: cat pos=3 -> (4-3)/3; car pos=1 -> 3/1; dog pos=2 -> 2/2.
    assert resolved["cat"] == pytest.approx((4 - 3) / 3)
    assert resolved["car"] == pytest.approx(3.0)
    assert resolved["dog"] == pytest.approx(1.0)
    # An image with several labels contributed to each class (img1 -> cat+car).


def test_resolve_class_weights__auto_zero_example_class_multiclass(
    tmp_path: Path,
) -> None:
    # Included class "dog" has no train images but exists in val.
    classes = {0: "cat", 1: "car", 2: "dog"}
    for cls_name, n in [("cat", 2), ("car", 2)]:
        d = tmp_path / "train" / cls_name
        d.mkdir(parents=True, exist_ok=True)
        helpers.create_images(d, files=n)
    for cls_name, n in [("cat", 1), ("car", 1), ("dog", 1)]:
        d = tmp_path / "val" / cls_name
        d.mkdir(parents=True, exist_ok=True)
        helpers.create_images(d, files=n)
    data_args = ImageClassificationMulticlassDataArgs(
        train=tmp_path / "train", val=tmp_path / "val", classes=classes
    )
    resolved = _resolve_auto(data_args)
    assert resolved is not None
    assert all(math.isfinite(v) for v in resolved.values())
    assert resolved["dog"] == pytest.approx(1.0)


def test_resolve_class_weights__auto_zero_positive_multilabel(tmp_path: Path) -> None:
    classes = {0: "cat", 1: "dog"}
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    helpers.create_image(train_dir / "img0.png")
    helpers.create_image(train_dir / "img1.png")
    val_dir = tmp_path / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    helpers.create_image(val_dir / "img0.png")
    train_csv = tmp_path / "train.csv"
    with train_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label"])
        w.writeheader()
        w.writerow({"image_path": str(train_dir / "img0.png"), "label": "cat"})
        w.writerow({"image_path": str(train_dir / "img1.png"), "label": "cat"})
    val_csv = tmp_path / "val.csv"
    with val_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label"])
        w.writeheader()
        w.writerow({"image_path": str(val_dir / "img0.png"), "label": "dog"})
    data_args = ImageClassificationMultilabelDataArgs(
        train=train_csv, val=val_csv, classes=classes
    )
    resolved = _resolve_auto(data_args)
    assert resolved is not None
    assert all(math.isfinite(v) for v in resolved.values())
    # "dog" unseen in train keeps neutral 1.0 so val positives are not suppressed.
    assert resolved["dog"] == pytest.approx(1.0)
    # "cat" is in every train image, so it stays neutral instead of dropping to 0.0.
    assert resolved["cat"] == pytest.approx(1.0)


def test_resolved_to_tensor__internal_order() -> None:
    classes = {3: "cat", 7: "car", 12: "dog"}
    tensor = cw.resolved_to_tensor({"cat": 2.0, "dog": 5.0}, _data_args(classes, {7}))
    assert tensor is not None
    assert torch.allclose(tensor, torch.tensor([2.0, 5.0]))


def test_resolved_to_tensor__none() -> None:
    assert cw.resolved_to_tensor(None, _data_args({0: "cat"}, None)) is None


def test_resolved_to_tensor__unresolved_auto_raises() -> None:
    with pytest.raises(ValueError, match="must be resolved"):
        cw.resolved_to_tensor("auto", _data_args({0: "cat", 1: "dog"}, None))


def test_resolved_to_tensor__missing_class_raises() -> None:
    with pytest.raises(ValueError, match="missing a weight"):
        cw.resolved_to_tensor({"cat": 1.0}, _data_args({0: "cat", 1: "dog"}, None))
