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
from lightly_train._task_models.ltdetr_object_detection.task_model import (
    LTDETRObjectDetection,
)
from lightly_train._task_models.ltdetr_object_detection.train_model import (
    _decode_predictions_for_metrics,
)


def test_decode_predictions_for_metrics__non_contiguous_class_ids() -> None:
    """Validation metrics must decode predictions into the internal class id space.

    KNOWN FAILURE. ``_decode_predictions_for_metrics`` passes
    ``model.internal_class_to_class`` into ``decode_object_detection_output``, which
    maps predictions from internal ids to user-facing class ids. Ground truth labels
    in ``batch["classes"]`` have already been remapped to internal contiguous ids by
    ``YOLOObjectDetectionDatasetArgs.list_image_info``, and
    ``MeanAveragePrecision`` indexes ``class_names`` by the internal id too.

    So whenever ``data.names`` is non-contiguous, or ``ignore_classes`` is used,
    predictions and targets end up in different label spaces: a perfect detection
    scores mAP 0, and classwise metrics raise IndexError. The bug is invisible for
    the usual ``0..N-1`` class maps because the mapping is then the identity.

    Fix: pass an identity mapping in
    ``lightly_train/_task_models/ltdetr_object_detection/train_model.py``
    (``_decode_predictions_for_metrics``), as PicoDet does with its
    ``metric_class_mapping`` buffer. Then delete this comment.
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
    outputs = {"pred_logits": logits, "pred_boxes": boxes}
    results = _decode_predictions_for_metrics(
        model=model,
        outputs=outputs,
        orig_target_sizes=torch.tensor([[400, 400]]),
    )

    # The decoded label must stay in the internal id space the targets use.
    assert int(results[0]["labels"][0]) == gt_label

    metric = MeanAveragePrecision(
        prefix="val", class_names=list(classes.values()), class_metrics=False
    )
    metric.update(results, targets)

    # A perfect prediction must score a perfect mAP. With the bug present this is
    # 0.0, because predictions carry user id 5 while targets carry internal id 1.
    # With class_metrics=True, compute() additionally raises IndexError because
    # class_names is indexed by the internal id and has only len(classes) entries.
    assert float(metric.compute()["val/map"]) == 1.0
