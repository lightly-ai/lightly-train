# C++ inference recipes for LT-DETR object detection

These are **recipes**, not general-purpose CLI tools: each program is a
single-purpose example for one model/image, with paths, normalization, image
size, and class names hardcoded as constants at the top of `main.cpp`. Copy
the folder, edit the constants, rebuild. This mirrors
[`examples/notebooks/object_detection_export.ipynb`](../notebooks/object_detection_export.ipynb),
just in C++.

Two recipes are provided:

- `onnxruntime/` -- runs an exported ONNX model with ONNX Runtime's CUDA
  execution provider.
- `tensorrt/` -- runs a serialized TensorRT engine.

Both allocate their model input directly on the GPU (zero-copy) instead of
relying on the runtime's implicit host↔device copies.

## 1. Export a model

On a machine with `lightly-train` installed:

```python
import lightly_train

model = lightly_train.load_model("ltdetrv2-s-coco")
print(model.image_size, model.image_normalize, model.classes)

model.export_onnx("model.onnx")
model.export_tensorrt("model.trt")  # requires the `tensorrt` package, see below
```

The printed `image_size` / `image_normalize` / `classes` must match the
`kModelHeight`/`kModelWidth`/`kNormalize`/`kClassNames` constants at the top
of `onnxruntime/main.cpp` and `tensorrt/main.cpp` -- update them if you
export a different checkpoint. TensorRT engines carry no metadata (unlike
ONNX's embedded `image_normalize`/`classes`), so this hardcoding is required
for the TensorRT recipe in particular.

## 2. Prerequisites

- CUDA toolkit (matching the driver on your machine).
- OpenCV development package (e.g. `apt install libopencv-dev` on Ubuntu).
- For the ONNX Runtime recipe: the **prebuilt GPU tarball** from the
  [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases)
  (e.g. `onnxruntime-linux-x64-gpu-<version>.tgz`) -- the pip `onnxruntime-gpu`
  wheel does not reliably ship the C++ headers/import library needed here.
- For the TensorRT recipe: TensorRT matching the version pinned in the
  notebook (`tensorrt-cu12==10.13.3.9`), either via the pip wheel's
  `site-packages/tensorrt*` directories or a native TensorRT install.

## 3. Build

```bash
cmake -S examples/cpp -B examples/cpp/build \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-<version> \
  -DTensorRT_ROOT=/path/to/TensorRT-10.13.3.9  # or the pip tensorrt package's install dir
cmake --build examples/cpp/build -j
```

If TensorRT was installed via `pip install tensorrt-cu12`, point
`-DTensorRT_ROOT` at its package directory, e.g.:

```bash
-DTensorRT_ROOT=$(python -c "import tensorrt, os; print(os.path.dirname(tensorrt.__file__))")
```

## 4. Run

```bash
./examples/cpp/build/onnxruntime/od_infer_onnxruntime
./examples/cpp/build/tensorrt/od_infer_tensorrt
```

Each program reads `image.jpg` (edit `kImagePath` for a different file) and
writes an annotated `output.jpg`, plus prints detected labels/scores/boxes to
stdout.

## Correctness check

Both recipes reproduce the same postprocessing as
`model.predict(image, threshold=0.6)`: sigmoid over raw logits, top-K
selection, box format conversion, and rescaling to the original image size
(see `common/detection_utils.hpp`). As a smoke test, compare a recipe's
printed labels/scores/boxes against `model.predict(...)`'s `"labels"`,
`"scores"`, `"bboxes"` for the same image and threshold -- they should match
modulo floating-point/interpolation tolerance (OpenCV's resize vs.
torchvision's).

There is no automated test harness for these recipes (this repo's `make test`
only covers the Python package) -- verification is manual, on a CUDA-capable
machine with TensorRT installed.
