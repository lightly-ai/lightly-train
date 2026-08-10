#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Run object detection inference with a compiled MIGraphX LT-DETR engine and
save a plot of the predicted bounding boxes.

The exported engine only returns raw class logits and normalized ``cxcywh``
boxes: image preprocessing, top-k selection, score thresholding, and box
rescaling all happen here, outside the graph.

Example:
    python predict.py \\
        --engine /workspace/dinov3-vitt16-ltdetr-coco.mxr \\
        --image /workspace/image.jpg \\
        --out /workspace/prediction.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import migraphx  # type: ignore[import-untyped]
import numpy as np
import torch
from torch import Tensor
from torchvision.io import read_image
from torchvision.ops import box_convert
from torchvision.transforms.v2 import functional as transforms_functional
from torchvision.utils import draw_bounding_boxes

import lightly_train

# Hardcoded ImageNet normalization statistics, matching the values LightlyTrain
# uses by default for object detection models.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Upper bound on the number of (query, class) pairs considered during decoding.
# Predictions are always filtered by --threshold afterwards, so this only needs
# to be at least as large as the number of detections one expects in an image.
NUM_TOP_QUERIES = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", required=True, type=Path, help="Path to the compiled .mxr engine."
    )
    parser.add_argument(
        "--image", required=True, type=Path, help="Path to the input image."
    )
    parser.add_argument(
        "--checkpoint",
        default="dinov3/vitt16-ltdetr-coco",
        help="LightlyTrain checkpoint name or path the engine was exported "
        "from. Only used to look up class names for the predicted labels.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("prediction.jpg"),
        help="Path to save the annotated plot to.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Minimum confidence score for a detection to be kept.",
    )
    return parser.parse_args()


def get_engine_input_size(program: migraphx.program) -> tuple[int, int]:
    """Read the static (height, width) the engine was exported with."""
    shape = program.get_parameter_shapes()["images"]
    _, _, height, width = shape.lens()
    return height, width


def preprocess(image_path: Path, height: int, width: int) -> tuple[Tensor, np.ndarray]:
    """Load and preprocess an image for the engine's fixed input size.

    Returns the original uint8 image (for drawing boxes on) and the
    preprocessed NCHW float32 batch (for feeding to the engine).
    """
    image = read_image(str(image_path))
    x = transforms_functional.to_dtype(image, dtype=torch.float32, scale=True)
    x = transforms_functional.resize(x, [height, width])
    x = transforms_functional.normalize(
        x, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)
    )
    batch = x.unsqueeze(0).contiguous().numpy().astype(np.float32)
    return image, batch


def decode(
    logits: np.ndarray,
    boxes: np.ndarray,
    orig_width: int,
    orig_height: int,
    threshold: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode raw engine outputs into (boxes_xyxy, scores, labels).

    Mirrors ``decode_object_detection_output`` in
    ``lightly_train._pre_post_processing.object_detection``.
    """
    logits_t = torch.from_numpy(logits)[0]  # (num_queries, num_classes)
    boxes_t = torch.from_numpy(boxes)[0]  # (num_queries, 4), normalized cxcywh

    scores_all = logits_t.sigmoid()
    num_classes = scores_all.shape[-1]
    num_top_queries = min(NUM_TOP_QUERIES, scores_all.numel())
    scores, index = scores_all.flatten().topk(num_top_queries)
    labels = index % num_classes
    query_index = index // num_classes

    boxes_xyxy = box_convert(boxes_t, in_fmt="cxcywh", out_fmt="xyxy")
    boxes_xyxy = boxes_xyxy[query_index]
    scale = torch.tensor([orig_width, orig_height, orig_width, orig_height])
    boxes_xyxy = boxes_xyxy * scale

    keep = scores > threshold
    return boxes_xyxy[keep], scores[keep], labels[keep]


def main() -> None:
    args = parse_args()

    program = migraphx.load(str(args.engine))
    height, width = get_engine_input_size(program)

    image, batch = preprocess(args.image, height, width)
    orig_height, orig_width = image.shape[-2:]

    outputs = program.run({"images": migraphx.argument(batch)})
    logits, boxes = (np.array(output) for output in outputs)

    boxes_xyxy, scores, labels = decode(
        logits, boxes, orig_width, orig_height, threshold=args.threshold
    )
    print(f"Found {len(scores)} objects above threshold {args.threshold}.")

    classes = lightly_train.load_model(args.checkpoint, device="cpu").classes
    label_strings = [
        f"{classes[label.item()]} {score:.2f}" for label, score in zip(labels, scores)
    ]

    image_with_boxes = draw_bounding_boxes(
        image, boxes=boxes_xyxy, labels=label_strings
    )
    plt.imshow(image_with_boxes.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
