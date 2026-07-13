#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Lightweight ONNX Runtime inference for exported LTDETR object detectors.

This mirrors the inference path of
``lightly_train._task_models.ltdetr_object_detection.LTDETRObjectDetection``
(``predict`` / ``predict_batch``) using only numpy, Pillow and onnxruntime.

The exported ``.onnx`` is self-describing: the class names, normalization
statistics and input size are read from the model's metadata and input shape, so
no configuration beyond the file path is required.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from lightlytrain_deploy_py.pre_post_processing import (
    ImageInput,
    ObjectDetectionMetadata,
    ObjectDetectionPostprocessor,
    ObjectDetectionPreprocessor,
)

PathLike = Union[str, "os.PathLike[str]"]


class LTDETRObjectDetectionONNX:
    """Run inference on an exported LTDETR object detection ONNX model.

    Args:
        onnx_path:
            Path to the exported ``.onnx`` model.
        num_top_queries:
            Number of top (query, class) candidates to keep before thresholding.
            Defaults to 300, which matches every non-test LTDETR config. It is not
            stored in the ONNX metadata, so set it explicitly if your model uses a
            different value.
        providers:
            ONNX Runtime execution providers. If None, onnxruntime's default order
            is used (which picks up a GPU provider when ``onnxruntime-gpu`` is
            installed).
    """

    def __init__(
        self,
        onnx_path: PathLike,
        *,
        num_top_queries: int = 300,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            os.fspath(onnx_path),
            providers=list(providers) if providers is not None else None,
        )

        # Read the self-describing metadata embedded during export.
        metadata = self.session.get_modelmeta().custom_metadata_map
        # JSON object keys are strings; class ids are integers.
        self.classes: Dict[int, str] = {
            int(class_id): name
            for class_id, name in json.loads(metadata["classes"]).items()
        }
        image_normalize = (
            json.loads(metadata["image_normalize"])
            if "image_normalize" in metadata
            else None
        )
        self.model_name: Optional[str] = metadata.get("model_name")

        # Input tensor: (batch, channels, height, width) with static C/H/W.
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        _, channels, height, width = model_input.shape
        self.image_size: Tuple[int, int] = (int(height), int(width))
        expected_input_channels = int(channels)

        # Output tensors, matched by name to stay robust to output ordering.
        self.output_names = [output.name for output in self.session.get_outputs()]

        # Internally the model uses contiguous class ids 0..N-1; map back to the
        # user-facing class ids (the keys of ``classes``).
        internal_class_to_class = np.asarray(list(self.classes.keys()), dtype=np.int64)
        self.included_classes: Dict[int, str] = {
            internal_class_id: class_name
            for internal_class_id, class_name in enumerate(self.classes.values())
        }

        self.preprocessor = ObjectDetectionPreprocessor(
            image_size=self.image_size,
            image_normalize=image_normalize,
            expected_input_channels=expected_input_channels,
        )
        self.postprocessor = ObjectDetectionPostprocessor(
            num_classes=len(self.classes),
            num_top_queries=num_top_queries,
            internal_class_to_class=internal_class_to_class,
        )

    def predict(
        self, image: ImageInput, threshold: float = 0.6
    ) -> Dict[str, np.ndarray]:
        """Run inference on a single image and return its predictions.

        Args:
            image:
                Input image as a filesystem path or a PIL image.
            threshold:
                Score threshold to filter low-confidence predictions. Predictions
                with scores <= threshold are discarded.

        Returns:
            A dict with ``"labels"`` (N,), ``"bboxes"`` (N, 4) as ``xyxy`` pixel
            coordinates, and ``"scores"`` (N,).
        """
        x, metadata = self.preprocessor.preprocess_image(image)
        batch = self.preprocessor.preprocess_batch(x[None, ...])
        logits, boxes = self._run(batch)
        return self.postprocessor.postprocess(logits, boxes, [metadata], threshold)[0]

    def predict_batch(
        self, images: Sequence[ImageInput], threshold: float = 0.6
    ) -> List[Dict[str, np.ndarray]]:
        """Run inference on a batch of images and return per-image predictions.

        Args:
            images:
                Sequence of input images, each a filesystem path or a PIL image.
            threshold:
                Score threshold to filter low-confidence predictions. Predictions
                with scores <= threshold are discarded.

        Returns:
            A list with one prediction dict per input image (see ``predict``).
        """
        if len(images) == 0:
            raise ValueError("images must contain at least one image.")
        tensors: List[np.ndarray] = []
        metadata: List[ObjectDetectionMetadata] = []
        for image in images:
            x, meta = self.preprocessor.preprocess_image(image)
            tensors.append(x)
            metadata.append(meta)
        batch = self.preprocessor.preprocess_batch(np.stack(tensors, axis=0))
        logits, boxes = self._run(batch)
        return self.postprocessor.postprocess(logits, boxes, metadata, threshold)

    def _run(self, batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run the ONNX session and return ``(logits, boxes)`` by output name."""
        outputs = self.session.run(
            self.output_names,
            {self.input_name: batch.astype(np.float32)},
        )
        named = dict(zip(self.output_names, outputs))
        return named["logits"], named["boxes"]
