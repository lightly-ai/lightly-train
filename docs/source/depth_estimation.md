(depth-estimation-doc)=

# Depth Estimation (NEW)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/depth_estimation.ipynb)

```{note}
LightlyTrain supports depth estimation inference with
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and
[Depth Anything V3](https://github.com/ByteDance-Seed/Depth-Anything-3) models.
Training support will be released soon!
```

LightlyTrain ports the Depth Anything V2 (DAv2) and V3 (DAv3) monocular depth estimation
models. Both come in two flavors:

- **Relative depth** predicts an unscaled depth map: it captures the ordering of the
  scene (what is closer and what is farther) but the values have no physical unit.
- **Metric depth** predicts depth in meters, suitable for 3D reconstruction and any
  application that needs absolute scale.

```{warning}
The meaning of the predicted values is not the same across models. For
**DAv2 relative** models, larger values are *nearer*. For **DAv3 relative** and for all
**metric** models, larger values are *farther*.
```

(depth-estimation-models)=

## Models

All models use a ViT backbone. The model name prefix indicates which one: `dinov2/`
models use a DINOv2 backbone and `dinov3/` models a DINOv3 backbone.

### Depth Anything V3

| Model                            | Type     |     Backbone     |
| -------------------------------- | -------- | :--------------: |
| `dinov3/dav3-relative-tiny`      | Relative | DINOv3 ViT-T/16  |
| `dinov3/dav3-relative-tiny-plus` | Relative | DINOv3 ViT-T+/16 |
| `dinov2/dav3-relative-small`     | Relative | DINOv2 ViT-S/14  |
| `dinov2/dav3-relative-large`     | Relative | DINOv2 ViT-L/14  |
| `dinov3/dav3-metric-tiny`        | Metric   | DINOv3 ViT-T/16  |
| `dinov3/dav3-metric-tiny-plus`   | Metric   | DINOv3 ViT-T+/16 |
| `dinov2/dav3-metric-small`       | Metric   | DINOv2 ViT-S/14  |
| `dinov2/dav3-metric-large`       | Metric   | DINOv2 ViT-L/14  |

The large models are ported from the official
[Depth Anything V3](https://github.com/ByteDance-Seed/Depth-Anything-3) release. The
tiny, tiny-plus, and small models are trained by LightlyTrain by distilling the
corresponding large model, giving much faster inference at a small accuracy cost (see
the [benchmarks](#depth-estimation-benchmarks) below).

### Depth Anything V2

| Model                                 | Type     | Backbone |
| ------------------------------------- | -------- | :------: |
| `dinov2/dav2-relative-small`          | Relative | ViT-S/14 |
| `dinov2/dav2-relative-base`\*         | Relative | ViT-B/14 |
| `dinov2/dav2-relative-large`\*        | Relative | ViT-L/14 |
| `dinov2/dav2-metric-small-hypersim`   | Metric   | ViT-S/14 |
| `dinov2/dav2-metric-base-hypersim`\*  | Metric   | ViT-B/14 |
| `dinov2/dav2-metric-large-hypersim`\* | Metric   | ViT-L/14 |
| `dinov2/dav2-metric-small-vkitti`\*   | Metric   | ViT-S/14 |
| `dinov2/dav2-metric-base-vkitti`\*    | Metric   | ViT-B/14 |
| `dinov2/dav2-metric-large-vkitti`\*   | Metric   | ViT-L/14 |

\* Not hosted by LightlyTrain. See the note below.

```{note}
All Depth Anything V3 models are hosted by LightlyTrain and downloaded automatically by
`load_model`. For Depth Anything V2, only the two small Apache-2.0 models
(`dinov2/dav2-relative-small` and `dinov2/dav2-metric-small-hypersim`) are hosted. The
remaining DAv2 models (marked with \* above) — the ViT-B/ViT-L variants and all VKITTI
variants — are released under non-commercial licenses
([CC-BY-NC-4.0](https://github.com/DepthAnything/Depth-Anything-V2) for the relative
base/large and the Hypersim metric variants,
[CC-BY-NC-SA-3.0](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
for the VKITTI metric variants), so LightlyTrain does not host them. You can convert
them from the official weights yourself, see
[Using Non-Hosted Depth Anything V2 Checkpoints](#depth-estimation-convert).
```

### Which model should I use?

- **Do you need depth in meters?** If yes, pick a **metric** model. If you only need the
  relative ordering of the scene (closer vs. farther), pick a **relative** model, it is
  simpler to use and needs no camera information.
- **Which size?** The tiny and tiny-plus models are the fastest and light enough for CPU
  and edge deployment, small is a good middle ground, and large is the most accurate.
  Start with large and step down if you need more speed, see the
  [benchmarks](#depth-estimation-benchmarks) below.
- **DAv3 or DAv2?** Use DAv3: it is the recent model, generally the most accurate, and
  comes in sizes from ViT-T to ViT-L. This is especially true for zero-shot metric depth
  estimation: the DAv2 metric models are only trained on in-domain data (indoor or
  outdoor driving scenes), while the DAv3 metric models generalize across domains.
- **Which DAv2 metric model?** The metric DAv2 models are trained per domain: use a
  `hypersim` model for **indoor** scenes (depth up to 20 m) and a `vkitti` model for
  **outdoor** driving scenes (depth up to 80 m).

(depth-estimation-benchmarks)=

### Benchmarks

Depth accuracy is evaluated zero-shot on the NYUv2 test split (654 images) with the
eigen crop and a depth range of 0.1 m to 10 m. NYUv2 was not used during training.
**Metric** models are scored directly against the ground-truth depth:

| Model                          | Params (M) |  δ1   | AbsRel | RMSE  |
| ------------------------------ | :--------: | :---: | :----: | :---: |
| `dinov2/dav3-metric-large`     |   334.2M   | 0.950 | 0.078  | 0.339 |
| `dinov2/dav3-metric-small`     |   24.7M    | 0.912 | 0.099  | 0.377 |
| `dinov3/dav3-metric-tiny-plus` |    7.9M    | 0.846 | 0.123  | 0.457 |
| `dinov3/dav3-metric-tiny`      |    6.2M    | 0.818 | 0.131  | 0.506 |

**Relative** models are scored after a per-image least-squares scale-and-shift alignment
to the ground truth, so the numbers are affine-invariant and not directly comparable to
the metric table:

| Model                            | Params (M) |  δ1   | AbsRel |
| -------------------------------- | :--------: | :---: | :----: |
| `dinov2/dav3-relative-large`     |   334.2M   | 0.928 | 0.084  |
| `dinov2/dav3-relative-small`     |   24.7M    | 0.909 | 0.101  |
| `dinov3/dav3-relative-tiny-plus` |    7.9M    | 0.874 | 0.120  |
| `dinov3/dav3-relative-tiny`      |    6.2M    | 0.882 | 0.118  |

All models are evaluated with the aspect-preserving `lower_bound_resize` method.

Inference time of the distilled relative models, measured with FP16 TensorRT engines on
an NVIDIA T4 GPU:

| Model                            | Input Size | Params (M) | Avg inference time |
| -------------------------------- | :--------: | :--------: | :----------------: |
| `dinov3/dav3-relative-tiny`      |  576×576   |    6.2M    |      5.27 ms       |
| `dinov3/dav3-relative-tiny-plus` |  576×576   |    7.9M    |      5.49 ms       |
| `dinov2/dav3-relative-small`     |  504×504   |   24.7M    |      9.17 ms       |

(depth-estimation-relative)=

## Quick Start

Load a model and call `predict` on an image. The image can be a file path, a URL, a PIL
image, or a `(C, H, W)` tensor. The result is a single `(H, W)` tensor with the same
height and width as the input image.

```python
import lightly_train

# Load a model hosted by LightlyTrain (downloaded and cached automatically).
model = lightly_train.load_model("dinov2/dav3-relative-small")

# Predict a relative-depth map. Returns a (H, W) tensor matching the input resolution.
depth = model.predict("image.jpg")
```

```{tip}
By default `load_model` runs on a GPU (`"cuda"` or `"mps"`) if one is available and falls
back to CPU otherwise. Pass `device=` to choose explicitly, e.g.
`lightly_train.load_model("dinov2/dav3-relative-small", device="cuda")`. The tiny and
small models are light enough for CPU inference; the ViT-L models are sizable, so a GPU
is recommended for them.
```

### Visualize the Result

The depth map is a plain tensor, so you can colorize and display it with `matplotlib`:

```python
import matplotlib.pyplot as plt
from PIL import Image

import lightly_train

model = lightly_train.load_model("dinov2/dav3-relative-small")
depth = model.predict("image.jpg")

# Colorize the depth map and save it next to the input image.
image = Image.open("image.jpg")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(image)
axes[0].set_title("Input")
axes[0].axis("off")
depth_vis = axes[1].imshow(depth.cpu(), cmap="Spectral_r")
axes[1].set_title("Relative depth (larger = farther)")
axes[1].axis("off")
fig.colorbar(depth_vis, ax=axes[1], fraction=0.046, pad=0.04)
fig.savefig("depth.png")
```

(depth-estimation-metric)=

## Predict Metric Depth

Metric models return depth in **meters**, with larger values corresponding to farther
scene content.

### Depth Anything V3

DAv3 metric models require the camera intrinsics of the input image, which set the
absolute scale of the prediction. Pass a `(3, 3)` intrinsics matrix in the original
image's pixel coordinates via the `intrinsics` argument:

```python
import math

import torch
from PIL import Image

import lightly_train

model = lightly_train.load_model("dinov2/dav3-metric-large")

# Approximate intrinsics from an assumed 60° horizontal field of view.
image = Image.open("image.jpg")
width, height = image.size
focal_px = (width / 2) / math.tan(math.radians(60.0) / 2)
intrinsics = torch.tensor(
    [
        [focal_px, 0.0, width / 2],
        [0.0, focal_px, height / 2],
        [0.0, 0.0, 1.0],
    ]
)

depth_m = model.predict("image.jpg", intrinsics=intrinsics)
print(f"depth range: {depth_m.min():.2f} m – {depth_m.max():.2f} m")
```

If you do not know the true intrinsics, an approximation from the field of view (as
above) still gives a reasonable scale.

### Depth Anything V2

DAv2 metric models are trained for a fixed domain and do **not** take camera intrinsics.
Choose the model that matches your scene:

- `hypersim` models are trained on indoor scenes (depth up to 20 m).
- `vkitti` models are trained on outdoor driving scenes (depth up to 80 m).

```python
import lightly_train

model = lightly_train.load_model("dinov2/dav2-metric-small-hypersim")
depth_m = model.predict("image.jpg")  # Metric depth in meters.
```

(depth-estimation-batch)=

## Batch Inference

Use `predict_batch` to run inference on several images at once. It returns a list of
`(H, W)` tensors, one per image, each resized back to its original resolution.

```python
import lightly_train

model = lightly_train.load_model("dinov2/dav3-relative-small")
depths = model.predict_batch(["image1.jpg", "image2.jpg"])
```

For DAv3 metric models, pass one intrinsics matrix per image:

```python skip_ruff
model = lightly_train.load_model("dinov2/dav3-metric-large")
depths = model.predict_batch(
    ["image1.jpg", "image2.jpg"],
    intrinsics=[intrinsics1, intrinsics2],
)
```

```{note}
By default (`process_res_method="square_resize"`) every image is resized to the same
square processing resolution, so batches of differently sized images are handled without
any cropping. The aspect-preserving methods (`"upper_bound_resize"` and
`"lower_bound_resize"`) can yield different processed sizes across a batch; those images
are then center-cropped to the smallest processed size before inference, so their depth
maps are slightly stretched when resized back to the original resolution. For
pixel-perfect results with an aspect-preserving method, call `predict` on each image
individually.
```

(depth-estimation-convert)=

## Using Non-Hosted Depth Anything V2 Checkpoints

To use a DAv2 model that LightlyTrain does not host, convert the official Depth Anything
V2 weights into a LightlyTrain checkpoint. You are responsible for complying with each
model's license terms.

1. Download the official weights for the model you want from the corresponding
   [Depth Anything V2 Hugging Face repository](https://huggingface.co/collections/depth-anything/depth-anything-v2-666b22412f18a6dbfde23a93)
   (for example `depth_anything_v2_metric_vkitti_vitl.pth`).

1. Convert them into a LightlyTrain checkpoint:

   ```bash
   python -m lightly_train._task_models.depth_estimation_components.convert_checkpoint_dav2 \
       --model-name dinov2/dav2-metric-large-vkitti \
       --weights path/to/depth_anything_v2_metric_vkitti_vitl.pth \
       --out ckpt/dav2_metric_vkitti_large.pt
   ```

1. Load the converted checkpoint like any other model:

   ```python
   import lightly_train

   model = lightly_train.load_model("ckpt/dav2_metric_vkitti_large.pt")
   depth_m = model.predict("image.jpg")
   ```

```{tip}
Converting the Apache-2.0 models (`dinov2/dav2-relative-small` and
`dinov2/dav2-metric-small-hypersim`) is not necessary, they are hosted by LightlyTrain
and downloaded automatically. For these, the converter can fetch the official weights
from Hugging Face directly, so the `--weights` argument can be omitted.
```

(depth-estimation-onnx)=

## Exporting a Checkpoint to ONNX

[Open Neural Network Exchange (ONNX)](https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange)
is a standard format for representing machine learning models in a framework independent
manner. In particular, it is useful for deploying our models on edge devices where
PyTorch is not available.

```{note}
The ONNX graph contains only the model forward pass. It outputs the raw depth map at the
model's processing resolution (plus a sky map for Depth Anything V3 models, which have a
sky head). Preprocessing the input image and postprocessing the output (sky filling,
metric scaling, and resizing back to the original image resolution) are not part of the
graph and must be applied by the caller.
```

### Requirements

Exporting to ONNX requires some additional packages to be installed. Namely

- [onnx](https://pypi.org/project/onnx/)
- [onnxruntime](https://pypi.org/project/onnxruntime/) if `verify` is set to `True`.
- [onnxslim](https://pypi.org/project/onnxslim/) if `simplify` is set to `True`.

You can install them with:

```bash
pip install "lightly-train[onnx,onnxruntime,onnxslim]"
```

The following example shows how to export a model to ONNX.

```python
import lightly_train

# Load a model.
model = lightly_train.load_model("dinov2/dav3-relative-small")

# Export to ONNX.
model.export_onnx(
    out="model.onnx",
    # precision="fp16", # Export model with FP16 weights for smaller size and faster inference.
)
```

See {py:meth}`~.DepthAnythingDepthEstimation.export_onnx` for all available options when
exporting to ONNX.

The following notebook shows how to export a model to ONNX in Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/depth_estimation_export.ipynb)

(depth-estimation-tensorrt)=

## Exporting a Checkpoint to TensorRT

TensorRT engines are built from an ONNX representation of the model. The
`export_tensorrt` method internally exports the model to ONNX (see the ONNX export
section above) before building a [TensorRT](https://developer.nvidia.com/tensorrt)
engine for fast GPU inference.

### Requirements

TensorRT is not part of LightlyTrain’s dependencies and must be installed separately.
Installation depends on your OS, Python version, GPU, and NVIDIA driver/CUDA setup. See
the
[TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)
for more details.

On CUDA 12.x systems you can often install the Python package via:

```bash
pip install tensorrt-cu12
```

```python
import lightly_train

# Load a model.
model = lightly_train.load_model("dinov2/dav3-relative-small")

# Export to TensorRT from an ONNX file.
model.export_tensorrt(
    out="model.trt", # TensorRT engine destination.
    # precision="fp16", # Export model with FP16 weights for smaller size and faster inference.
)
```

See {py:meth}`~.DepthAnythingDepthEstimation.export_tensorrt` for all available options
when exporting to TensorRT.

### Combining Conversion and Export

ONNX export and TensorRT engine building can be combined in a single call: building a
TensorRT engine requires an ONNX model, and `export_tensorrt` exports it for you. You
can fix the export resolution via `onnx_args` (`height` and `width`, both multiples of
the backbone patch size: 14 for `dinov2/` models, 16 for `dinov3/` models) and pick the
precision at the same time:

```python skip_ruff
import lightly_train

# A hosted model name, or a path to a converted DAv2 checkpoint.
model = lightly_train.load_model("dinov2/dav3-relative-small")


model.export_tensorrt(
    out="model.trt",
    onnx_args={"height": 504, "width": 504},
    precision="fp16",
)
```
