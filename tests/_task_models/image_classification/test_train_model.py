#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lightly_train._data.image_classification_dataset import (
    ImageClassificationDataArgs,
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

from ... import helpers


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
    data_args: ImageClassificationDataArgs,
    model_args: ImageClassificationTrainArgs,
) -> ImageClassificationTrain:
    train_t = ImageClassificationTrainTransformArgs()
    train_t.resolve_auto(model_init_args={})
    val_t = ImageClassificationValTransformArgs()
    val_t.resolve_auto(model_init_args={})
    model_args.resolve_auto(
        total_steps=10,
        gradient_accumulation_steps=1,
        train_num_batches=4,
        model_name="dinov2/_vittest14",
        model_init_args={},
        data_args=data_args,
        train_dataset=helpers.get_image_classification_train_dataset(data_args),
    )
    metric_args = (
        _multilabel_metric_args()
        if data_args.classification_task == "multilabel"
        else _multiclass_metric_args()
    )
    return ImageClassificationTrain(
        model_name="dinov2/_vittest14",
        model_args=model_args,
        data_args=data_args,
        train_transform_args=train_t,
        val_transform_args=val_t,
        load_weights=False,
        metric_args=metric_args,
        gradient_accumulation_steps=1,
    )


def _multiclass_data_args(tmp_path: Path) -> ImageClassificationMulticlassDataArgs:
    helpers.create_multiclass_image_classification_dataset(
        tmp_path=tmp_path, class_names=["cat", "dog"], num_files_per_class=1
    )
    return ImageClassificationMulticlassDataArgs(
        train=tmp_path / "train", val=tmp_path / "val", classes={0: "cat", 1: "dog"}
    )


def _multilabel_data_args(tmp_path: Path) -> ImageClassificationMultilabelDataArgs:
    classes = {0: "cat", 1: "dog"}
    helpers.create_multilabel_image_classification_dataset(
        tmp_path=tmp_path, classes=classes, num_files=2
    )
    return ImageClassificationMultilabelDataArgs(
        train=tmp_path / "train.csv", val=tmp_path / "val.csv", classes=classes
    )


def test_class_weights__default_none_multiclass(tmp_path: Path) -> None:
    data_args = _multiclass_data_args(tmp_path)
    assert ImageClassificationTrainArgs().class_weights is None

    model = _make_train_model(data_args, ImageClassificationTrainArgs())
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)
    assert model.criterion.weight is None
    # label_smoothing default untouched
    assert model.criterion.label_smoothing == 0.0


def test_class_weights__default_none_multilabel(tmp_path: Path) -> None:
    data_args = _multilabel_data_args(tmp_path)
    model = _make_train_model(data_args, ImageClassificationTrainArgs())
    assert isinstance(model.criterion, torch.nn.BCEWithLogitsLoss)
    assert model.criterion.pos_weight is None


def test_class_weights__manual_multiclass(tmp_path: Path) -> None:
    data_args = _multiclass_data_args(tmp_path)
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


def test_class_weights__manual_multilabel(tmp_path: Path) -> None:
    data_args = _multilabel_data_args(tmp_path)
    model = _make_train_model(
        data_args, ImageClassificationTrainArgs(class_weights={"cat": 1.0, "dog": 4.0})
    )
    assert isinstance(model.criterion, cw.NormalizedBCEWithLogitsLoss)
    assert model.criterion.pos_weight is not None
    assert torch.allclose(model.criterion.pos_weight.cpu(), torch.tensor([1.0, 4.0]))


def test_class_weights__ignored_class(tmp_path: Path) -> None:
    classes = {3: "cat", 7: "car", 12: "dog"}
    for name in ["cat", "dog"]:
        # Only surviving classes need folders; ignored "car" folder is skipped.
        for split in ["train", "val"]:
            d = tmp_path / split / name
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


@pytest.mark.parametrize("classification_task", ["multiclass", "multilabel"])
def test_class_weights__validation_criterion_is_unweighted(
    tmp_path: Path, classification_task: str
) -> None:
    if classification_task == "multiclass":
        data_args: ImageClassificationDataArgs = _multiclass_data_args(tmp_path)
        weights = {"cat": 1.0, "dog": 3.5}
    else:
        data_args = _multilabel_data_args(tmp_path)
        weights = {"cat": 1.0, "dog": 4.0}

    model = _make_train_model(
        data_args, ImageClassificationTrainArgs(class_weights=weights)
    )
    if classification_task == "multiclass":
        assert isinstance(model.val_criterion, torch.nn.CrossEntropyLoss)
        assert model.val_criterion.weight is None
    else:
        assert isinstance(model.val_criterion, torch.nn.BCEWithLogitsLoss)
        assert model.val_criterion.pos_weight is None


def test_class_weights__auto_multiclass(tmp_path: Path) -> None:
    classes = {0: "cat", 1: "dog"}
    for split, counts in [
        ("train", {"cat": 3, "dog": 1}),
        ("val", {"cat": 1, "dog": 1}),
    ]:
        for cls_name, n in counts.items():
            d = tmp_path / split / cls_name
            d.mkdir(parents=True, exist_ok=True)
            helpers.create_images(d, files=n)
    data_args = ImageClassificationMulticlassDataArgs(
        train=tmp_path / "train", val=tmp_path / "val", classes=classes
    )
    model = _make_train_model(
        data_args, ImageClassificationTrainArgs(class_weights="auto")
    )
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)
    assert model.criterion.weight is not None
    assert torch.allclose(model.criterion.weight.cpu(), torch.tensor([0.5, 1.5]))


def test_class_weights__weighted_cross_entropy_keeps_scale() -> None:
    # CrossEntropyLoss divides by the sum of the sample weights, so the reported
    # loss stays on the same scale as an unweighted run. Documented in
    # docs/source/image_classification.md.
    torch.manual_seed(0)
    logits = torch.randn(8, 3)
    targets = torch.tensor([0, 0, 0, 0, 0, 0, 1, 2])
    weights = torch.tensor([1.0, 10.0, 10.0])
    unweighted = torch.nn.CrossEntropyLoss()(logits, targets)
    weighted = torch.nn.CrossEntropyLoss(weight=weights)(logits, targets)
    assert weighted < 2 * unweighted


def test_class_weights__normalized_multilabel_bce_keeps_scale_and_relative_weight() -> (
    None
):
    # At zero logits, every unweighted BCE term is equal. Normalization should keep
    # the overall loss on that same scale while retaining the positive-term ratio.
    targets = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    unweighted_logits = torch.zeros_like(targets)
    unweighted = torch.nn.BCEWithLogitsLoss()(unweighted_logits, targets)
    assert cw.NormalizedBCEWithLogitsLoss()(
        unweighted_logits, targets
    ) == pytest.approx(unweighted.item())
    normalized = cw.NormalizedBCEWithLogitsLoss(pos_weight=torch.tensor([2.0, 4.0]))
    assert normalized.reduction == "none"
    assert normalized(unweighted_logits, targets) == pytest.approx(unweighted.item())

    logits = torch.zeros_like(targets, requires_grad=True)
    loss = normalized(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1, 0].abs() / logits.grad[0, 0].abs() == pytest.approx(2.0)
    assert logits.grad[1, 1].abs() / logits.grad[0, 1].abs() == pytest.approx(4.0)

    zero_weight_logits = torch.zeros_like(targets, requires_grad=True)
    zero_weight_loss = cw.NormalizedBCEWithLogitsLoss(pos_weight=torch.zeros(2))(
        zero_weight_logits, torch.ones_like(targets)
    )
    assert torch.isfinite(zero_weight_loss)
    zero_weight_loss.backward()
    assert zero_weight_logits.grad is not None
    assert torch.isfinite(zero_weight_logits.grad).all()
