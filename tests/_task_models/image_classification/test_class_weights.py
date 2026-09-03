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
    ImageClassificationMulticlassDataArgs,
    ImageClassificationMultilabelDataArgs,
)
from lightly_train._metrics.classification.task_metric import (
    MulticlassClassificationTaskMetricArgs,
    MultilabelClassificationTaskMetricArgs,
)
from lightly_train._task_models import image_classification_class_weights as cw
from lightly_train._task_models.image_classification.train_model import (
    ImageClassificationTrain,
    ImageClassificationTrainArgs,
)
from lightly_train._task_models.image_classification.transforms import (
    ImageClassificationTrainTransformArgs,
    ImageClassificationValTransformArgs,
)
from lightly_train._task_models.image_classification_multihead.train_model import (
    ImageClassificationMultiheadTrain,
    ImageClassificationMultiheadTrainArgs,
)
from lightly_train._task_models.image_classification_multihead.transforms import (
    ImageClassificationMultiheadTrainTransformArgs,
    ImageClassificationMultiheadValTransformArgs,
)

from ... import helpers


def _transform_args() -> tuple[
    ImageClassificationTrainTransformArgs, ImageClassificationValTransformArgs
]:
    train_args = ImageClassificationTrainTransformArgs()
    train_args.resolve_auto(model_init_args={})
    val_args = ImageClassificationValTransformArgs()
    val_args.resolve_auto(model_init_args={})
    return train_args, val_args


def _multihead_transform_args() -> tuple[
    ImageClassificationMultiheadTrainTransformArgs,
    ImageClassificationMultiheadValTransformArgs,
]:
    train_args = ImageClassificationMultiheadTrainTransformArgs()
    train_args.resolve_auto(model_init_args={})
    val_args = ImageClassificationMultiheadValTransformArgs()
    val_args.resolve_auto(model_init_args={})
    return train_args, val_args


def _multiclass_metric_args() -> MulticlassClassificationTaskMetricArgs:
    return MulticlassClassificationTaskMetricArgs(
        accuracy=None,
        f1=None,
        precision=None,
        recall=None,
        watch_metric="val_loss",
    )


def _multilabel_metric_args() -> MultilabelClassificationTaskMetricArgs:
    return MultilabelClassificationTaskMetricArgs(
        accuracy=None,
        f1=None,
        precision=None,
        recall=None,
        auroc=None,
        average_precision=None,
        hamming_distance=None,
        watch_metric="val_loss",
    )


def _make_train_model(
    data_args: ImageClassificationMulticlassDataArgs
    | ImageClassificationMultilabelDataArgs,
    model_args: ImageClassificationTrainArgs,
) -> ImageClassificationTrain:
    train_t, val_t = _transform_args()
    model_args.resolve_auto(
        total_steps=10,
        gradient_accumulation_steps=1,
        train_num_batches=4,
        model_name="dinov2/_vittest14",
        model_init_args={},
        data_args=data_args,  # type: ignore[arg-type]
    )
    metric_args = (
        _multilabel_metric_args()
        if data_args.classification_task == "multilabel"
        else _multiclass_metric_args()
    )
    return ImageClassificationTrain(
        model_name="dinov2/_vittest14",
        model_args=model_args,
        data_args=data_args,  # type: ignore[arg-type]
        train_transform_args=train_t,
        val_transform_args=val_t,
        load_weights=False,
        metric_args=metric_args,
        gradient_accumulation_steps=1,
    )


def _make_multihead_model(
    data_args: ImageClassificationMulticlassDataArgs
    | ImageClassificationMultilabelDataArgs,
    model_args: ImageClassificationMultiheadTrainArgs,
) -> ImageClassificationMultiheadTrain:
    train_t, val_t = _multihead_transform_args()
    model_args.resolve_auto(
        total_steps=10,
        gradient_accumulation_steps=1,
        train_num_batches=4,
        model_name="dinov2/_vittest14",
        model_init_args={},
        data_args=data_args,  # type: ignore[arg-type]
    )
    metric_args = (
        _multilabel_metric_args()
        if data_args.classification_task == "multilabel"
        else _multiclass_metric_args()
    )
    return ImageClassificationMultiheadTrain(
        model_name="dinov2/_vittest14",
        model_args=model_args,
        data_args=data_args,  # type: ignore[arg-type]
        train_transform_args=train_t,
        val_transform_args=val_t,
        load_weights=False,
        metric_args=metric_args,
        gradient_accumulation_steps=1,
    )


def test_default_none_preserves_unweighted_multiclass(tmp_path: Path) -> None:
    classes = {0: "cat", 1: "dog"}
    helpers.create_multiclass_image_classification_dataset(
        tmp_path=tmp_path, class_names=["cat", "dog"], num_files_per_class=1
    )
    data_args = ImageClassificationMulticlassDataArgs(
        train=tmp_path / "train", val=tmp_path / "val", classes=classes
    )
    assert ImageClassificationTrainArgs().class_weights is None
    assert ImageClassificationMultiheadTrainArgs().class_weights is None

    model = _make_train_model(data_args, ImageClassificationTrainArgs())
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)
    assert model.criterion.weight is None
    # label_smoothing default untouched
    assert model.criterion.label_smoothing == 0.0


def test_default_none_preserves_unweighted_multilabel(tmp_path: Path) -> None:
    classes = {0: "cat", 1: "dog"}
    helpers.create_multilabel_image_classification_dataset(
        tmp_path=tmp_path, classes=classes, num_files=2
    )
    data_args = ImageClassificationMultilabelDataArgs(
        train=tmp_path / "train.csv", val=tmp_path / "val.csv", classes=classes
    )
    model = _make_train_model(data_args, ImageClassificationTrainArgs())
    assert isinstance(model.criterion, torch.nn.BCEWithLogitsLoss)
    assert model.criterion.pos_weight is None


def test_manual_multiclass_weights_order() -> None:
    classes = {3: "cat", 7: "car", 12: "dog"}
    ordered = cw.validate_manual_weights(
        {"dog": 3.0, "cat": 1.0, "car": 2.0},
        classes=classes,
        ignore_classes=None,
    )
    # Internal order follows user dict order: cat(3)->0, car(7)->1, dog(12)->2.
    assert ordered == [1.0, 2.0, 3.0]


def test_manual_multilabel_pos_weight_used(tmp_path: Path) -> None:
    classes = {0: "cat", 1: "dog"}
    helpers.create_multilabel_image_classification_dataset(
        tmp_path=tmp_path, classes=classes, num_files=2
    )
    data_args = ImageClassificationMultilabelDataArgs(
        train=tmp_path / "train.csv", val=tmp_path / "val.csv", classes=classes
    )
    model = _make_train_model(
        data_args, ImageClassificationTrainArgs(class_weights={"cat": 1.0, "dog": 4.0})
    )
    assert isinstance(model.criterion, torch.nn.BCEWithLogitsLoss)
    assert model.criterion.pos_weight is not None
    assert torch.allclose(model.criterion.pos_weight.cpu(), torch.tensor([1.0, 4.0]))


def test_manual_multiclass_weights_used(tmp_path: Path) -> None:
    classes = {0: "cat", 1: "dog"}
    helpers.create_multiclass_image_classification_dataset(
        tmp_path=tmp_path, class_names=["cat", "dog"], num_files_per_class=1
    )
    data_args = ImageClassificationMulticlassDataArgs(
        train=tmp_path / "train", val=tmp_path / "val", classes=classes
    )
    model = _make_train_model(
        data_args, ImageClassificationTrainArgs(class_weights={"cat": 1.0, "dog": 3.5})
    )
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)
    assert model.criterion.weight is not None
    assert torch.allclose(model.criterion.weight.cpu(), torch.tensor([1.0, 3.5]))
    # label_smoothing preserved
    model2 = _make_train_model(
        data_args,
        ImageClassificationTrainArgs(
            class_weights={"cat": 1.0, "dog": 3.5}, label_smoothing=0.1
        ),
    )
    assert isinstance(model2.criterion, torch.nn.CrossEntropyLoss)
    assert model2.criterion.label_smoothing == 0.1
    assert model2.criterion.weight is not None
    assert torch.allclose(model2.criterion.weight.cpu(), torch.tensor([1.0, 3.5]))


def test_auto_multiclass_inverse_frequency_mean_one() -> None:
    weights = cw.compute_auto_multiclass_weights([90, 10])
    assert math.isfinite(weights[0]) and math.isfinite(weights[1])
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    # Rare class gets higher weight; ratio equals inverse-count ratio.
    assert weights[1] / weights[0] == pytest.approx(9.0)
    # Balanced classes give uniform weights.
    balanced = cw.compute_auto_multiclass_weights([5, 5, 5])
    assert balanced == pytest.approx([1.0, 1.0, 1.0])


def test_auto_multiclass_uses_train_split_not_val(tmp_path: Path) -> None:
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
    resolved = cw.resolve_class_weights("auto", data_args)
    assert resolved is not None
    # counts cat=3, dog=1 -> raw 1/3, 1/1, mean 2/3 -> weights 0.5, 1.5
    assert resolved["cat"] == pytest.approx(0.5)
    assert resolved["dog"] == pytest.approx(1.5)


def test_auto_multilabel_neg_over_pos(tmp_path: Path) -> None:
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
    resolved = cw.resolve_class_weights("auto", data_args)
    assert resolved is not None
    # total=4: cat pos=3 -> (4-3)/3; car pos=1 -> 3/1; dog pos=2 -> 2/2.
    assert resolved["cat"] == pytest.approx((4 - 3) / 3)
    assert resolved["car"] == pytest.approx(3.0)
    assert resolved["dog"] == pytest.approx(1.0)
    # An image with several labels contributed to each class (img1 -> cat+car).


def test_zero_example_class_finite_multiclass() -> None:
    weights = cw.compute_auto_multiclass_weights([5, 0])
    assert all(math.isfinite(w) for w in weights)
    assert weights[1] == pytest.approx(1.0)
    assert sum(weights) / len(weights) == pytest.approx(1.0)


def test_zero_positive_multilabel_neutral() -> None:
    weights = cw.compute_auto_multilabel_pos_weights([0, 2], total=4)
    assert all(math.isfinite(w) for w in weights)
    # Zero-positive class must not be suppressed with 0.0.
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx((4 - 2) / 2)


def test_zero_example_class_end_to_end_multiclass(tmp_path: Path) -> None:
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
    resolved = cw.resolve_class_weights("auto", data_args)
    assert resolved is not None
    assert all(math.isfinite(v) for v in resolved.values())
    assert resolved["dog"] == pytest.approx(1.0)


def test_zero_positive_end_to_end_multilabel(tmp_path: Path) -> None:
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
    resolved = cw.resolve_class_weights("auto", data_args)
    assert resolved is not None
    assert all(math.isfinite(v) for v in resolved.values())
    # "dog" unseen in train keeps neutral 1.0 so val positives are not suppressed.
    assert resolved["dog"] == pytest.approx(1.0)


def test_unknown_class_name_raises() -> None:
    classes = {0: "cat", 1: "dog"}
    with pytest.raises(ValueError, match="Unknown class name"):
        cw.validate_manual_weights(
            {"cat": 1.0, "bird": 2.0}, classes=classes, ignore_classes=None
        )


def test_missing_class_raises() -> None:
    classes = {0: "cat", 1: "dog"}
    with pytest.raises(ValueError, match="missing weight"):
        cw.validate_manual_weights({"cat": 1.0}, classes=classes, ignore_classes=None)


def test_ignored_class_names_rejected() -> None:
    classes = {0: "cat", 1: "car", 2: "dog"}
    with pytest.raises(ValueError, match="Unknown class name"):
        cw.validate_manual_weights(
            {"cat": 1.0, "car": 1.0, "dog": 1.0},
            classes=classes,
            ignore_classes={1},
        )


def test_ignore_middle_class_ordering() -> None:
    # Original: 3->cat, 7->car, 12->dog; ignore 7 -> internal 0->cat, 1->dog.
    classes = {3: "cat", 7: "car", 12: "dog"}
    ordered_ids = cw.internal_ordered_class_ids(classes, {7})
    assert ordered_ids == [3, 12]
    ordered = cw.validate_manual_weights(
        {"dog": 5.0, "cat": 2.0}, classes=classes, ignore_classes={7}
    )
    assert ordered == [2.0, 5.0]
    tensor = cw.resolved_to_tensor({"cat": 2.0, "dog": 5.0}, _data_args(classes, {7}))
    assert tensor is not None
    assert torch.allclose(tensor, torch.tensor([2.0, 5.0]))


def _data_args(
    classes: dict[int, str], ignore: set[int] | None
) -> ImageClassificationMulticlassDataArgs:
    return ImageClassificationMulticlassDataArgs(
        train="train", val="val", classes=classes, ignore_classes=ignore
    )


def test_normal_train_uses_feature_multiclass(tmp_path: Path) -> None:
    classes = {3: "cat", 7: "car", 12: "dog"}
    for name in ["cat", "dog"]:
        # Only surviving classes need folders; ignored "car" folder is skipped.
        d = tmp_path / "train" / name
        d.mkdir(parents=True, exist_ok=True)
        helpers.create_images(d, files=1)
        d = tmp_path / "val" / name
        d.mkdir(parents=True, exist_ok=True)
        helpers.create_images(d, files=1)
    data_args = ImageClassificationMulticlassDataArgs(
        train=tmp_path / "train",
        val=tmp_path / "val",
        classes=classes,
        ignore_classes={7},
    )
    model = _make_train_model(
        data_args, ImageClassificationTrainArgs(class_weights={"cat": 2.0, "dog": 5.0})
    )
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)
    assert model.criterion.weight is not None
    assert torch.allclose(model.criterion.weight.cpu(), torch.tensor([2.0, 5.0]))


def test_multihead_uses_feature_multiclass(tmp_path: Path) -> None:
    classes = {3: "cat", 7: "car", 12: "dog"}
    for name in ["cat", "dog"]:
        d = tmp_path / "train" / name
        d.mkdir(parents=True, exist_ok=True)
        helpers.create_images(d, files=1)
        d = tmp_path / "val" / name
        d.mkdir(parents=True, exist_ok=True)
        helpers.create_images(d, files=1)
    data_args = ImageClassificationMulticlassDataArgs(
        train=tmp_path / "train",
        val=tmp_path / "val",
        classes=classes,
        ignore_classes={7},
    )
    model = _make_multihead_model(
        data_args,
        ImageClassificationMultiheadTrainArgs(
            lr=[0.001, 0.01], class_weights={"cat": 2.0, "dog": 5.0}
        ),
    )
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)
    assert model.criterion.weight is not None
    assert torch.allclose(model.criterion.weight.cpu(), torch.tensor([2.0, 5.0]))


def test_multihead_uses_feature_multilabel(tmp_path: Path) -> None:
    classes = {0: "cat", 1: "dog"}
    helpers.create_multilabel_image_classification_dataset(
        tmp_path=tmp_path, classes=classes, num_files=2
    )
    data_args = ImageClassificationMultilabelDataArgs(
        train=tmp_path / "train.csv", val=tmp_path / "val.csv", classes=classes
    )
    model = _make_multihead_model(
        data_args,
        ImageClassificationMultiheadTrainArgs(class_weights={"cat": 1.0, "dog": 2.0}),
    )
    assert isinstance(model.criterion, torch.nn.BCEWithLogitsLoss)
    assert model.criterion.pos_weight is not None
    assert torch.allclose(model.criterion.pos_weight.cpu(), torch.tensor([1.0, 2.0]))


def test_multihead_auto_matches_single_head(tmp_path: Path) -> None:
    classes = {0: "cat", 1: "dog", 2: "bird"}
    for cls_name, n in [("cat", 4), ("dog", 1), ("bird", 1)]:
        d = tmp_path / "train" / cls_name
        d.mkdir(parents=True, exist_ok=True)
        helpers.create_images(d, files=n)
        d = tmp_path / "val" / cls_name
        d.mkdir(parents=True, exist_ok=True)
        helpers.create_images(d, files=1)
    data_args = ImageClassificationMulticlassDataArgs(
        train=tmp_path / "train", val=tmp_path / "val", classes=classes
    )
    single = _make_train_model(
        data_args, ImageClassificationTrainArgs(class_weights="auto")
    )
    multi = _make_multihead_model(
        data_args, ImageClassificationMultiheadTrainArgs(class_weights="auto")
    )
    assert isinstance(single.criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(multi.criterion, torch.nn.CrossEntropyLoss)
    assert single.criterion.weight is not None
    assert multi.criterion.weight is not None
    assert torch.allclose(single.criterion.weight.cpu(), multi.criterion.weight.cpu())
