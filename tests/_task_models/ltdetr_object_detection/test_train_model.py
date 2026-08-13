#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Dict, List, cast

import pytest
import torch
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor

if RequirementCache("torchmetrics<1.5"):
    # Skip test if torchmetrics version is too old. This can happen if SuperGradients
    # is installed which requires torchmetrics==0.8
    pytest.skip("Old torchmetrics version", allow_module_level=True)

from lightly_train._data import label_helpers
from lightly_train._metrics.mean_average_precision import MeanAveragePrecision
from lightly_train._pre_post_processing.object_detection import (
    ObjectDetectionBatchOutput,
    decode_object_detection_output,
)
from lightly_train._task_models.ltdetr_object_detection.task_model import (
    LTDETRObjectDetection,
)


def test_decode_predictions_for_metrics__non_contiguous_class_ids() -> None:
    """Validation metrics must decode predictions into the internal class id space.

    Ground truth labels in ``batch["classes"]`` are remapped to internal contiguous
    ids by ``YOLOObjectDetectionDatasetArgs.list_image_info``, and
    ``MeanAveragePrecision`` indexes ``class_names`` by the internal id too.
    ``decode_object_detection_output`` must therefore never remap predictions to
    user-facing class ids: whenever ``data.names`` is non-contiguous, or
    ``ignore_classes`` is used, remapping would put predictions and targets in
    different label spaces, silently scoring a perfect detection as mAP 0 and making
    classwise metrics raise IndexError.
    """
    # A non-contiguous user class map, as `data.names` would provide.
    classes = {0: "person", 5: "bus"}
    model = LTDETRObjectDetection(
        model_name="dinov3/vitt16-notpretrained-ltdetr",
        classes=classes,
        image_size=(256, 256),
        load_weights=False,
    )

    # The dataset remaps ground truth labels to internal contiguous ids, so a "bus"
    # annotation reaches the training step as label 1, not 5.
    class_id_to_internal_class_id = (
        label_helpers.get_class_id_to_internal_class_id_mapping(
            class_ids=classes.keys(), ignore_classes=None
        )
    )
    gt_label = class_id_to_internal_class_id[5]
    assert gt_label == 1
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([gt_label]),
        }
    ]

    # One query predicts "bus" with a box exactly matching the ground truth; the
    # remaining queries are suppressed. The decoder takes a fixed top-k over
    # (query, class) pairs, so the full query count has to be present.
    num_queries = model.num_top_queries
    logits = torch.full((1, num_queries, len(classes)), -20.0)
    logits[0, 0, 1] = 10.0
    boxes = torch.zeros(1, num_queries, 4)
    boxes[0, 0] = torch.tensor([0.075, 0.075, 0.1, 0.1])
    outputs = ObjectDetectionBatchOutput(logits=logits, boxes=boxes)
    results = decode_object_detection_output(
        raw=outputs,
        target_sizes=torch.tensor([[400, 400]]),
        num_top_queries=num_queries,
    ).to_torchmetrics_list()

    # The decoded label must stay in the internal id space the targets use.
    assert int(results[0]["labels"][0]) == gt_label

    metric = MeanAveragePrecision(
        prefix="val", class_names=list(classes.values()), class_metrics=True
    )
    metric.update(cast(List[Dict[str, Tensor]], results), targets)
    computed = metric.compute()

    # A perfect prediction must score a perfect mAP. If predictions were remapped to
    # user-facing ids this would be 0.0 (predictions carry id 5, targets carry
    # internal id 1) and class_metrics=True would additionally raise IndexError,
    # since class_names is indexed by the internal id and has only len(classes)
    # entries.
    assert float(computed["val/map"]) == 1.0
    assert float(computed["val_classwise/map_bus"]) == 1.0
