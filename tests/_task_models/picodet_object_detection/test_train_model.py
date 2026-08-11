#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch

from lightly_train._data import label_helpers
from lightly_train._metrics.mean_average_precision import MeanAveragePrecision
from lightly_train._pre_post_processing.object_detection import (
    ObjectDetectionBatchOutput,
    decode_object_detection_output,
)


def test_decode_predictions_for_metrics__non_contiguous_class_ids() -> None:
    """Validation metrics decode predictions into the internal class id space.

    Ground truth labels are remapped to internal contiguous ids by the dataset, and
    the metric indexes class_names by that internal id, so
    ``decode_object_detection_output`` must never remap predictions to user-facing
    class ids.
    """
    # A non-contiguous user class map, as `data.names` would provide.
    classes = {0: "person", 5: "bus"}
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

    # One anchor predicts "bus" with a box exactly matching the ground truth.
    logits = torch.tensor([[[-20.0, 10.0], [-20.0, -20.0]]])
    boxes = torch.tensor([[[0.075, 0.075, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0]]])

    results = decode_object_detection_output(
        raw=ObjectDetectionBatchOutput(logits=logits, boxes=boxes),
        target_sizes=torch.tensor([[400, 400]]),
        num_top_queries=2,
    ).to_torchmetrics_list()

    assert int(results[0]["labels"][0]) == gt_label

    metric = MeanAveragePrecision(
        prefix="val", class_names=list(classes.values()), class_metrics=True
    )
    metric.update(results, targets)
    computed = metric.compute()

    # A perfect prediction scores a perfect mAP, and classwise metrics resolve the
    # class name through the internal id without going out of range.
    assert float(computed["val/map"]) == 1.0
    assert float(computed["val_classwise/map_bus"]) == 1.0
