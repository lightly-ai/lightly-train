# NVIDIA GPU Inference

This page collects deployment recipes for models exported by LightlyTrain on NVIDIA GPUs.
Choose a runtime first, then use the matching Python export notebook or the native C++ recipe.

## ONNX Runtime

### Python

Export an LT-DETR model to ONNX with the [object-detection export notebook](https://github.com/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection_export.ipynb).
The notebook records the model input size, normalization, and classes needed at inference time.

### C++

The [ONNX Runtime C++ recipe](https://github.com/lightly-ai/lightly-train/tree/main/examples/cpp/onnxruntime) runs the exported model with ONNX Runtime's CUDA execution provider.
Build it through the top-level [C++ inference README](https://github.com/lightly-ai/lightly-train/tree/main/examples/cpp), which documents the required CUDA, OpenCV, and ONNX Runtime SDK setup.

## TensorRT

### Python

Use the same [object-detection export notebook](https://github.com/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection_export.ipynb) to export a TensorRT engine. Keep the exported model's image size, normalization, and class list with the engine because TensorRT engines do not carry that metadata.

### C++

The [TensorRT C++ recipe](https://github.com/lightly-ai/lightly-train/tree/main/examples/cpp/tensorrt) executes a serialized engine with GPU-resident inputs and outputs.
Follow the [C++ inference README](https://github.com/lightly-ai/lightly-train/tree/main/examples/cpp) for CMake configuration, TensorRT version requirements, and a correctness smoke check against `model.predict(...)`.

## Verify an export

For either runtime, compare the native recipe's labels, scores, and boxes with `model.predict(...)` on the same image and threshold. Small floating-point differences from preprocessing are expected; changes in predicted classes or box geometry indicate an export or preprocessing mismatch.