(depth-estimation)=

# Depth Estimation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/depth_estimation.ipynb)

```{note}
LightlyTrain currently supports **inference only** for depth estimation with
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and
[Depth Anything V3](https://github.com/ByteDance-Seed/Depth-Anything-3) models.
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

All models use a DINOv2 ViT backbone.

### Depth Anything V3

| Model                        | Type     | Backbone |
| ---------------------------- | -------- | :------: |
| `dinov2/dav3-relative-large` | Relative | ViT-L/14 |
| `dinov2/dav3-metric-large`   | Metric   | ViT-L/14 |

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
- **DAv3 or DAv2?** DAv3 is the recent model and generally the most accurate. Choose
  DAv2 if you need a smaller and faster ViT-S or ViT-B model, or a model under a
  permissive license for commercial use, the two hosted DAv2 small models are
  Apache-2.0.
- **Which DAv2 metric model?** The metric DAv2 models are trained per domain: use a
  `hypersim` model for **indoor** scenes (depth up to 20 m) and a `vkitti` model for
  **outdoor** driving scenes (depth up to 80 m).

(depth-estimation-relative)=

## Quick Start

Load a model and call `predict` on an image. The image can be a file path, a URL, a PIL
image, or a `(C, H, W)` tensor. The result is a single `(H, W)` tensor with the same
height and width as the input image.

```python
import lightly_train

# Load a model hosted by LightlyTrain (downloaded and cached automatically).
model = lightly_train.load_model("dinov2/dav3-relative-large")

# Predict a relative-depth map. Returns a (H, W) tensor matching the input resolution.
depth = model.predict("image.jpg")
```

```{tip}
By default `load_model` runs on a GPU (`"cuda"` or `"mps"`) if one is available and falls
back to CPU otherwise. Pass `device=` to choose explicitly, e.g.
`lightly_train.load_model("dinov2/dav3-relative-large", device="cuda")`. The default ViT-L
models are sizable, so a GPU is recommended.
```

### Visualize the Result

The depth map is a plain tensor, so you can colorize and display it with `matplotlib`:

```python
import matplotlib.pyplot as plt
from PIL import Image

import lightly_train

model = lightly_train.load_model("dinov2/dav3-relative-large")
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

model = lightly_train.load_model("dinov2/dav3-relative-large")
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
Images with different aspect ratios are center-cropped to the smallest processed size in
the batch before inference, so their depth maps are slightly stretched when resized back
to the original resolution. For pixel-perfect results on images of different sizes, call
`predict` on each image individually.
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
