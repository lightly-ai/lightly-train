# lightlytrain-deploy-py

Lightweight inference for models exported from
[LightlyTrain](https://github.com/lightly-ai/lightly-train).

This package runs exported ONNX object detection models with only **Pillow**,
**NumPy** and **ONNX Runtime** — no torch, torchvision, or the full LightlyTrain
stack. It mirrors the `predict` / `predict_batch` inference path of
`LTDETRObjectDetection`.

## Installation

Pick the ONNX Runtime build that matches your hardware:

```bash
# CPU
pip install "lightlytrain-deploy-py[cpu]"
# GPU (CUDA)
pip install "lightlytrain-deploy-py[gpu]"
```

## Usage

The exported `.onnx` is self-describing: class names, normalization statistics
and input size are read from the model file, so only the path is required.

```python
from lightlytrain_deploy_py import LTDETRObjectDetectionONNX

model = LTDETRObjectDetectionONNX("model.onnx")

# Single image (path or PIL image).
prediction = model.predict("image.jpg", threshold=0.6)
print(prediction["labels"])  # (N,)   class ids
print(prediction["bboxes"])  # (N, 4) xyxy pixel coordinates
print(prediction["scores"])  # (N,)   confidence scores

# Batch of images.
predictions = model.predict_batch(["a.jpg", "b.jpg"], threshold=0.6)
```

### Notes

- `num_top_queries` (the number of top query/class candidates kept before
  thresholding) is not stored in the ONNX metadata. It defaults to `300`, which
  matches every non-test LightlyTrain config. Override it in the constructor if
  your model differs:

  ```python
  model = LTDETRObjectDetectionONNX("model.onnx", num_top_queries=20)
  ```

- Pass `providers=[...]` to control the ONNX Runtime execution providers (e.g.
  `["CUDAExecutionProvider", "CPUExecutionProvider"]`).
