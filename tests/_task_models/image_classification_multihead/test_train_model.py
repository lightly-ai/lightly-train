#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

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
from lightly_train._task_models.image_classification_multihead.train_model import (
    ImageClassificationMultiheadTrain,
    ImageClassificationMultiheadTrainArgs,
)
from lightly_train._task_models.image_classification_multihead.transforms import (
    ImageClassificationMultiheadTrainTransformArgs,
    ImageClassificationMultiheadValTransformArgs,
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


def _make_multihead_model(
    data_args: ImageClassificationDataArgs,
    model_args: ImageClassificationMultiheadTrainArgs,
) -> ImageClassificationMultiheadTrain:
    train_t = ImageClassificationMultiheadTrainTransformArgs()
    train_t.resolve_auto(model_init_args={})
    val_t = ImageClassificationMultiheadValTransformArgs()
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
    return ImageClassificationMultiheadTrain(
        model_name="dinov2/_vittest14",
        model_args=model_args,
        data_args=data_args,
        train_transform_args=train_t,
        val_transform_args=val_t,
        load_weights=False,
        metric_args=metric_args,
        gradient_accumulation_steps=1,
    )


def _make_single_head_model(
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
    return ImageClassificationTrain(
        model_name="dinov2/_vittest14",
        model_args=model_args,
        data_args=data_args,
        train_transform_args=train_t,
        val_transform_args=val_t,
        load_weights=False,
        metric_args=_multiclass_metric_args(),
        gradient_accumulation_steps=1,
    )


def test_class_weights__default_none(tmp_path: Path) -> None:
    helpers.create_multiclass_image_classification_dataset(
        tmp_path=tmp_path, class_names=["cat", "dog"], num_files_per_class=1
    )
    data_args = ImageClassificationMulticlassDataArgs(
        train=tmp_path / "train", val=tmp_path / "val", classes={0: "cat", 1: "dog"}
    )
    assert ImageClassificationMultiheadTrainArgs().class_weights is None

    model = _make_multihead_model(data_args, ImageClassificationMultiheadTrainArgs())
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)
    assert model.criterion.weight is None


def test_class_weights__manual_multiclass(tmp_path: Path) -> None:
    classes = {3: "cat", 7: "car", 12: "dog"}
    for name in ["cat", "dog"]:
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
    model = _make_multihead_model(
        data_args,
        ImageClassificationMultiheadTrainArgs(
            lr=[0.001, 0.01], class_weights={"cat": 2.0, "dog": 5.0}
        ),
    )
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)
    assert model.criterion.weight is not None
    assert torch.allclose(model.criterion.weight.cpu(), torch.tensor([2.0, 5.0]))


def test_class_weights__manual_multilabel(tmp_path: Path) -> None:
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
    assert isinstance(model.criterion, cw.NormalizedBCEWithLogitsLoss)
    assert model.criterion.pos_weight is not None
    assert torch.allclose(model.criterion.pos_weight.cpu(), torch.tensor([1.0, 2.0]))


def test_class_weights__validation_criterion_is_unweighted(tmp_path: Path) -> None:
    classes = {0: "cat", 1: "dog"}
    helpers.create_multiclass_image_classification_dataset(
        tmp_path=tmp_path, class_names=["cat", "dog"], num_files_per_class=1
    )
    data_args = ImageClassificationMulticlassDataArgs(
        train=tmp_path / "train", val=tmp_path / "val", classes=classes
    )
    model = _make_multihead_model(
        data_args,
        ImageClassificationMultiheadTrainArgs(class_weights={"cat": 1.0, "dog": 3.5}),
    )
    assert isinstance(model.val_criterion, torch.nn.CrossEntropyLoss)
    assert model.val_criterion.weight is None


def test_class_weights__validation_multilabel_criterion_is_unweighted(
    tmp_path: Path,
) -> None:
    classes = {0: "cat", 1: "dog"}
    helpers.create_multilabel_image_classification_dataset(
        tmp_path=tmp_path, classes=classes, num_files=2
    )
    data_args = ImageClassificationMultilabelDataArgs(
        train=tmp_path / "train.csv", val=tmp_path / "val.csv", classes=classes
    )
    model = _make_multihead_model(
        data_args,
        ImageClassificationMultiheadTrainArgs(class_weights={"cat": 1.0, "dog": 3.5}),
    )
    assert isinstance(model.val_criterion, torch.nn.BCEWithLogitsLoss)
    assert model.val_criterion.pos_weight is None


def test_class_weights__auto_matches_single_head(tmp_path: Path) -> None:
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
    single = _make_single_head_model(
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
