#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest
import torch

from lightly_train._metrics.detection.task_metric import (
    ObjectDetectionTaskMetric,
    ObjectDetectionTaskMetricArgs,
)
from lightly_train._task_models.ltdetr_object_detection.task_model import (
    LTDETRObjectDetection,
)
from lightly_train._task_models.ltdetr_object_detection.train_model import (
    _decode_predictions_for_metrics,
)


@pytest.mark.parametrize(
    "class_ids",
    [
        [0, 1, 2],
        [1, 2, 3],  # COCO-style ids starting at 1.
        [0, 2, 5],  # Gapped ids.
    ],
)
def test_decode_predictions_for_metrics__perfect_predictions_give_full_map(
    class_ids: list[int],
) -> None:
    model = LTDETRObjectDetection(
        model_name="dinov3/vitt16-notpretrained-ltdetr",
        classes={class_id: f"class_{class_id}" for class_id in class_ids},
        image_size=(64, 64),
        load_weights=False,
    )
    num_classes = len(class_ids)
    num_queries = model.postprocessor.num_top_queries
    image_size = 100

    # One ground truth box per class. Targets hold internal class ids, as the
    # dataset emits them.
    internal_labels = torch.arange(num_classes)
    boxes_cxcywh = torch.tensor(
        [[0.2 + 0.25 * i, 0.5, 0.1, 0.1] for i in range(num_classes)]
    )
    boxes_xyxy = torch.cat(
        [
            boxes_cxcywh[:, :2] - boxes_cxcywh[:, 2:] / 2,
            boxes_cxcywh[:, :2] + boxes_cxcywh[:, 2:] / 2,
        ],
        dim=1,
    )
    targets = [{"boxes": boxes_xyxy * image_size, "labels": internal_labels}]

    # Query i predicts ground truth box i with its class and high confidence.
    pred_logits = torch.full((1, num_queries, num_classes), -10.0)
    pred_logits[0, internal_labels, internal_labels] = 10.0
    pred_boxes = torch.full((1, num_queries, 4), 0.05)
    pred_boxes[0, :num_classes] = boxes_cxcywh

    results = _decode_predictions_for_metrics(
        model=model,
        outputs={"pred_logits": pred_logits, "pred_boxes": pred_boxes},
        orig_target_sizes=torch.tensor([[image_size, image_size]]),
    )

    metric = ObjectDetectionTaskMetric(
        task_metric_args=ObjectDetectionTaskMetricArgs(),
        split="val",
        class_names=list(model.classes.values()),
        box_format="xyxy",
        loss_names=[],
    )
    metric.update_with_predictions(results, targets)
    assert metric.compute_aggregated_values().watch_metric_value == pytest.approx(1.0)
