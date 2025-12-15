(panoptic-segmentation)=

# Panoptic Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_panoptic_segmentation.ipynb)

```{note}
🔥 LightlyTrain now supports training **DINOv3**-based panoptic segmentation models
with the [EoMT architecture](https://arxiv.org/abs/2503.19108) by Kerssies et al.!
```

(panoptic-segmentation-benchmark-results)=

## Benchmark Results

Below we provide the models and report the validation panoptic quality (PQ) and
inference latency of different DINOv3 models fine-tuned on COCO with LightlyTrain.
You can check [here](panoptic-segmentation-train) how to use these models for further
fine-tuning.

You can also explore running inference and training these models using our Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_panoptic_segmentation.ipynb)

### COCO

| Implementation | Model | Val PQ | Avg. Latency (ms) | Params (M) | Input Size |
|----------------|----------------|-------------|----------|-----------|------------|
| LightlyTrain | dinov3/vits16-eomt-panoptic-coco | 46.8 | 21.2 | 23.4 | 640×640 |
| LightlyTrain | dinov3/vitb16-eomt-panoptic-coco | 53.2 | 39.4 | 92.5 | 640×640 |
| LightlyTrain | dinov3/vitl16-eomt-panoptic-coco | 57.0 | 80.1 | 315.1 | 640×640 |
| LightlyTrain | dinov3/vitl16-eomt-panoptic-coco-1280 | **59.0** | 500.1 | 315.1 | 1280×1280 |
| EoMT (CVPR 2025 paper, current SOTA) | dinov3/vitl16-eomt-panoptic-coco-1280 | 58.9 | - | 315.1 | 1280×1280 |

Training follows the protocol in the original [EoMT paper](https://arxiv.org/abs/2503.19108).
Small and base models are trained for 180K steps (24 epochs) and large models for
90K steps (12 epochs) on the COCO dataset with batch size `16` and learning rate `2e-4`.
The average latency values were measured with model compilation using `torch.compile`
on a single NVIDIA T4 GPU with FP16 precision.

(panoptic-segmentation-train)=

## Train a Panoptic Segmentation Model

Training a panoptic segmentation model with LightlyTrain is straightforward and
only requires a few lines of code. See [data](#panoptic-segmentation-data)
for more details on how to prepare your dataset.

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_panoptic_segmentation(
        out="out/my_experiment",
        model="dinov3/vitl16-eomt-panoptic-coco", 
        data={
            "train": {
                "images": "images/train",   # Path to train images
                "masks": "annotations/train", # Path to train mask images
                "annotations": "annotations/train.json", # Path to train COCO-style annotations
            },
            "val": {
                "images": "images/val", # Path to val images
                "masks": "annotations/val", # Path to val mask images
                "annotations": "annotations/val.json", # Path to val COCO-style annotations
            },
        },
    )
```

During training, the best and last model weights are exported to
`out/my_experiment/exported_models/`, unless disabled in `save_checkpoint_args`:

- best (highest validation PQ): `exported_best.pt`
- last: `exported_last.pt`

You can use these weights to continue fine-tuning on another dataset by loading the
weights with `model="<checkpoint path>"`:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_panoptic_segmentation(
        out="out/my_experiment",
        model="out/my_experiment/exported_models/exported_best.pt",  # Continue training from the best model
        data={...},
    )
```

(panoptic-segmentation-inference)=

### Load the Trained Model from Checkpoint and Predict

After the training completes, you can load the best model checkpoints for inference like
this:

```python
import lightly_train

model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
results = model.predict("image.jpg")
results["masks"]    # Masks with (class_label, segment_id) for each pixel, tensor of
                    # shape (height, width, 2). Height and width correspond to the
                    # original image size.
results["segment_ids"]    # Segment ids, tensor of shape (num_segments,).
results["scores"]   # Confidence scores, tensor of shape (num_segments,)
```

Or use one of the pretrained models directly from LightlyTrain:

```python
import lightly_train

model = lightly_train.load_model("dinov3/vitl16-eomt-panoptic-coco")
results = model.predict("image.jpg")
```

### Visualize the Predictions

You can visualize the predicted masks like this:

```python skip_ruff
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks

image = read_image("image.jpg")
masks = torch.stack([masks[..., 1] == -1] + [masks[..., 1] == segment_id for segment_id in segment_ids])
colors = [(0, 0, 0)] + [[int(color * 255) for color in plt.cm.tab20c(i / len(segment_ids))[:3]] for i in range(len(segment_ids))]
image_with_masks = draw_segmentation_masks(image, masks, colors=colors, alpha=1.0)
plt.imshow(image_with_masks.permute(1, 2, 0))
```

<!--

# Figure created with

import lightly_train
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
from pathlib import Path

image_path = "/datasets/coco/images/val2017/000000070254.jpg"
model = lightly_train.load_model("251209_dinov3_vitl16_eomt_panoptic_coco_1280/exported_models/exported_best.pt")
results = model.predict(image_path)
masks = results["masks"]
segment_ids = results["segment_ids"]

image = read_image(image_path)
masks = torch.stack([masks[..., 1] == -1] + [masks[..., 1] == segment_id for segment_id in segment_ids])
colors = [(0, 0, 0)] + [[int(color * 255) for color in plt.cm.hsv(i / len(segment_ids))[:3]] for i in range(len(segment_ids))]
image_with_masks = draw_segmentation_masks(image, masks, colors=colors, alpha=1.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image.permute(1, 2, 0))
ax2.imshow(image_with_masks.permute(1, 2, 0))
ax1.axis("off")
ax2.axis("off")
fig.savefig(f"panoptic_segmentation_result_{Path(image_path).stem}_.png", bbox_inches="tight")
-->

```{figure} /_static/images/panoptic_segmentation/train.jpg
```

(panoptic-segmentation-data)=

## Data

Lightly**Train** supports panoptic segmentation datasets in COCO format.
Every image must have a corresponding mask image that encodes the segmentation class
and segment ID for each pixel. The dataset must also include COCO-style JSON annotation
files that define the thing and stuff classes and list the individual segments for each
image. See the [COCO Panoptic Segmentation format](https://cocodataset.org/#format-data)
for more details.

The following image formats are supported:

- jpg
- jpeg
- png
- ppm
- bmp
- pgm
- tif
- tiff
- webp

Your dataset directory must be organized like this:

```text
my_data_dir/
├── images
│   ├── train
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── annotations
    ├── train
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    ├── train.json
    ├── val
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    └── val.json
```

The directories can have any name, as long as the paths are correctly specified in the
`data` argument.

See the [Colab notebook](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_panoptic_segmentation.ipynb)
for an example dataset and how to set up the data for training.

(panoptic-segmentation-model)=

## Model

The `model` argument defines the model used for panoptic segmentation training. The
following models are available:

### DINOv3 Models

- `dinov3/vits16-eomt-panoptic-coco` (fine-tuned on COCO)
- `dinov3/vitb16-eomt-panoptic-coco` (fine-tuned on COCO)
- `dinov3/vitl16-eomt-panoptic-coco` (fine-tuned on COCO)
- `dinov3/vitl16-eomt-panoptic-coco-1280` (fine-tuned on COCO with 1280x1280 input size)
- `dinov3/vitt16-eomt`
- `dinov3/vitt16plus-eomt`
- `dinov3/vits16-eomt`
- `dinov3/vits16plus-eomt`
- `dinov3/vitb16-eomt`
- `dinov3/vitl16-eomt`
- `dinov3/vitl16plus-eomt`
- `dinov3/vith16plus-eomt`
- `dinov3/vit7b16-eomt`

All models are [pretrained by Meta](https://github.com/facebookresearch/dinov3/tree/main?tab=readme-ov-file#pretrained-models)
and fine-tuned by Lightly, except the `vitt` models which are pretrained by Lightly.

(panoptic-segmentation-logging)=

## Logging

Logging is configured with the `logger_args` argument. The following loggers are
supported:

- [`mlflow`](panoptic-segmentation-mlflow): Logs training metrics to MLflow (disabled by
  default, requires MLflow to be installed)
- [`tensorboard`](panoptic-segmentation-tensorboard): Logs training metrics to TensorBoard
  (enabled by default, requires TensorBoard to be installed)
- [`wandb`](panoptic-segmentation-wandb): Logs training metrics to Weights & Biases (disabled by
  default, requires wandb to be installed)

(panoptic-segmentation-mlflow)=

### MLflow

```{important}
MLflow must be installed with `pip install "lightly-train[mlflow]"`.
```

The mlflow logger can be configured with the following arguments:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_panoptic_segmentation(
        out="out/my_experiment",
        model="dinov3/vitl16-eomt-panoptic-coco",
        data={
            # ...
        },
        logger_args={
            "mlflow": {
                "experiment_name": "my_experiment",
                "run_name": "my_run",
                "tracking_uri": "tracking_uri",
            },
        },
    )
```

(panoptic-segmentation-tensorboard)=

### TensorBoard

TensorBoard logs are automatically saved to the output directory. Run TensorBoard in
a new terminal to visualize the training progress:

```bash
tensorboard --logdir out/my_experiment
```

Disable the TensorBoard logger with:

```python
logger_args={"tensorboard": None}
```

(panoptic-segmentation-wandb)=

### Weights & Biases

```{important}
Weights & Biases must be installed with `pip install "lightly-train[wandb]"`.
```

The Weights & Biases logger can be configured with the following arguments:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_panoptic_segmentation(
        out="out/my_experiment",
        model="dinov3/vitl16-eomt-panoptic-coco",
        data={
            # ...
        },
        logger_args={
            "wandb": {
                "project": "my_project",
                "name": "my_experiment",
                "log_model": False,        # Set to True to upload model checkpoints
            },
        },
    )
```

(panoptic-segmentation-resume-training)=

## Resume Training

There are two distinct ways to continue training, depending on your intention.

### Resume Interrupted Training

Use `resume_interrupted=True` to **resume a previously interrupted or crashed training run**.
This will pick up exactly where the training left off.

- You **must use the same `out` directory** as the original run.
- You **must not change any training parameters** (e.g., learning rate, batch size, data, etc.).
- This is intended for continuing the **same** run without modification.

This will utilize the `.ckpt` checkpoint file `out/my_experiment/checkpoints/last.ckpt`
to restore the entire training state, including model weights, optimizer state, and epoch count.

### Load Weights for a New Run

As stated above, you can specify `model="<checkpoint path">` to further fine-tune a
model from a previous run.

- You are free to **change training parameters**.
- This is useful for continuing training with a different setup.

We recommend using the exported best model weights from `out/my_experiment/exported_models/exported_best.pt`
for this purpose, though a `.ckpt` file can also be loaded.

(panoptic-segmentation-transform-args)=

## Default Image Transform Arguments

The following are the default train transform arguments. The validation arguments are
automatically inferred from the train arguments.

You can configure the image size and normalization like this:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_panoptic_segmentation(
        out="out/my_experiment",
        model="dinov3/vitl16-eomt-panoptic-coco",
        data={
            # ...
        }
        transform_args={
            "image_size": (1280, 1280),     # (height, width)
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    )
```

`````{dropdown} EoMT Panoptic Segmentation DINOv3 Default Transform Arguments
````{dropdown} Train
```{include} _auto/dinov3eomtpanopticsegmentationtrain_train_transform_args.md
```
````
````{dropdown} Val
```{include} _auto/dinov3eomtpanopticsegmentationtrain_val_transform_args.md
```
````
`````

In case you need different parameters for training and validation, you can pass an
optional `val` dictionary to `transform_args` to override the validation parameters:

```python
transform_args={
    "image_size": (640, 640), # (height, width)
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "val": {    # Override validation parameters
        "image_size": (512, 512), # (height, width)
    }
}
```
