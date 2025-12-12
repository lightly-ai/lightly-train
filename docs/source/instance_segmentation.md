(instance-segmentation)=

# Instance Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_instance_segmentation.ipynb)

```{note}
🔥 LightlyTrain now supports training **DINOv3**-based instance segmentation models
with the [EoMT architecture](https://arxiv.org/abs/2503.19108) by Kerssies et al.!
```

(instance-segmentation-benchmark-results)=

## Benchmark Results

Below we provide the models and report the validation mAP and inference latency
of different DINOv3 models fine-tuned on COCO with LightlyTrain. You can check
[here](instance-segmentation-train) how to use these models for further fine-tuning.

You can also explore running inference and training these models using our Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_instance_segmentation.ipynb)

### COCO

| Implementation | Model | Val mAP mask | Avg. Latency (ms) | Params (M) | Input Size |
|----------------|----------------|-------------|----------|-----------|------------|
| LightlyTrain | dinov3/vits16-eomt-inst-coco | 32.6 | 19.4 | 21.6 | 640×640 |
| LightlyTrain | dinov3/vitb16-eomt-inst-coco | 40.3 | 39.7 | 85.7 | 640×640 |
| LightlyTrain | dinov3/vitl16-eomt-inst-coco | **46.2** | 80.0 | 303.2 | 640×640 |
| Original EoMT | dinov3/vitl16-eomt-inst-coco | 45.9 | - | 303.2 | 640×640 |

Training follows the protocol in the original [EoMT paper](https://arxiv.org/abs/2503.19108).
Models are trained for 90K steps (~12 epochs) on the COCO dataset with batch size `16`
and learning rate `2e-4`. The average latency values were measured with model compilation
using `torch.compile` on a single NVIDIA T4 GPU with FP16 precision.

(instance-segmentation-train)=

## Train an Instance Segmentation Model

Training an instance segmentation model with LightlyTrain is straightforward and
only requires a few lines of code. See [data](#instance-segmentation-data)
for more details on how to prepare your dataset.

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_instance_segmentation(
        out="out/my_experiment",
        model="dinov3/vitl16-eomt-inst-coco", 
        data={
            "path": "my_data_dir",      # Path to dataset directory
            "train": "images/train",    # Path to training images
            "val": "images/val",        # Path to validation images
            "names": {                  # Classes in the dataset                    
                0: "background",
                1: "car",
                2: "bicycle",
                # ...
            },
        },
    )
```

During training, the best and last model weights are exported to
`out/my_experiment/exported_models/`, unless disabled in `save_checkpoint_args`:

- best (highest validation mask mAP): `exported_best.pt`
- last: `exported_last.pt`

You can use these weights to continue fine-tuning on another dataset by loading the
weights with `model="<checkpoint path>"`:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_instance_segmentation(
        out="out/my_experiment",
        model="out/my_experiment/exported_models/exported_best.pt",  # Continue training from the best model
        data={...},
    )
```

(instance-segmentation-inference)=

### Load the Trained Model from Checkpoint and Predict

After the training completes, you can load the best model checkpoints for inference like
this:

```python
import lightly_train

model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
results = model.predict("image.jpg")
results["labels"]   # Class labels, tensor of shape (num_instances,)
results["masks"]    # Binary masks, tensor of shape (num_instances, height, width).
                    # Height and width correspond to the original image size.
results["scores"]   # Confidence scores, tensor of shape (num_instances,)
```

Or use one of the pretrained models directly from LightlyTrain:

```python
import lightly_train

model = lightly_train.load_model("dinov3/vitl16-eomt-inst-coco")
results = model.predict("image.jpg")
```

### Visualize the Predictions

You can visualize the predicted masks like this:

```python skip_ruff
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks

image = read_image("image.jpg")
image_with_masks = draw_segmentation_masks(image, results["masks"], alpha=0.6)
plt.imshow(image_with_masks.permute(1, 2, 0))
```

<!--

# Figure created with

import lightly_train
import matplotlib.pyplot as plt
from torchvision.io import decode_image
from torchvision.utils import draw_segmentation_masks
import urllib.request

model = lightly_train.load_model("251107_dinov3_vitb16_eomt_inst_coco.pt")
img = "http://images.cocodataset.org/val2017/000000039769.jpg"
results = model.predict(img)
masks = results["masks"]
scores = results["scores"]

urllib.request.urlretrieve(img, "/tmp/image.jpg")
image = decode_image("/tmp/image.jpg")
image_with_masks = draw_segmentation_masks(image, masks, alpha=1.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image.permute(1, 2, 0))
ax2.imshow(image_with_masks.permute(1, 2, 0))
ax1.axis("off")
ax2.axis("off")
fig.savefig("out/preds/inst_seg.jpg", bbox_inches="tight")
fig.show()
-->

```{figure} /_static/images/instance_segmentation/cats.jpg
```

(instance-segmentation-data)=

## Data

Lightly**Train** supports instance segmentation datasets in YOLO format.
Every image must have a corresponding annotation file that contains for every object in
the image a line with the class ID and (x1, y1, x2, y2, ...) polygon coordinates in
normalized format.

```text
0 0.782016 0.986521 0.937078 0.874167 0.957297 0.782021 0.950562 0.739333
1 0.557859 0.143813 0.487078 0.0314583 0.859547 0.00897917 0.985953 0.130333 0.984266 0.184271
```

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
└── labels
    ├── train
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val
        ├── image1.txt
        ├── image2.txt
        └── ...
```

Alternatively, the train/val splits can also be at the top level:

```text
my_data_dir/
├── train
│   ├── images
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── val
    ├── images
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── labels
        ├── image1.txt
        ├── image2.txt
        └── ...
```

The `data` argument in `train_instance_segmentation` must point to the dataset
directory and specify the paths to the training and validation images relative to
the dataset directory. For example:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_instance_segmentation(
        out="out/my_experiment",
        model="dinov3/vitl16-eomt-inst-coco", 
        data={
            "path": "my_data_dir",      # Path to dataset directory
            "train": "images/train",    # Path to training images
            "val": "images/val",        # Path to validation images
            "names": {                  # Classes in the dataset                    
                0: "background",        # Classes must match those in the annotation files
                1: "car",
                2: "bicycle",
                # ...
            },
        },
    )
```

(instance-segmentation-model)=

## Model

The `model` argument defines the model used for instance segmentation training. The
following models are available:

### DINOv3 Models

- `dinov3/vits16-eomt`
- `dinov3/vits16plus-eomt`
- `dinov3/vitb16-eomt`
- `dinov3/vitl16-eomt`
- `dinov3/vitl16plus-eomt`
- `dinov3/vith16plus-eomt`
- `dinov3/vit7b16-eomt`
- `dinov3/vits16-eomt-inst-coco` (fine-tuned on COCO)
- `dinov3/vitb16-eomt-inst-coco` (fine-tuned on COCO)
- `dinov3/vitl16-eomt-inst-coco` (fine-tuned on COCO)

All models are [pretrained by Meta](https://github.com/facebookresearch/dinov3/tree/main?tab=readme-ov-file#pretrained-models)
and fine-tuned by Lightly.

(instance-segmentation-logging)=

## Logging

Logging is configured with the `logger_args` argument. The following loggers are
supported:

- [`mlflow`](instance-segmentation-mlflow): Logs training metrics to MLflow (disabled by
  default, requires MLflow to be installed)
- [`tensorboard`](instance-segmentation-tensorboard): Logs training metrics to TensorBoard
  (enabled by default, requires TensorBoard to be installed)
- [`wandb`](instance-segmentation-wandb): Logs training metrics to Weights & Biases (disabled by
  default, requires wandb to be installed)

(instance-segmentation-mlflow)=

### MLflow

```{important}
MLflow must be installed with `pip install "lightly-train[mlflow]"`.
```

The mlflow logger can be configured with the following arguments:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_instance_segmentation(
        out="out/my_experiment",
        model="dinov3/vitl16-eomt-inst-coco",
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

(instance-segmentation-tensorboard)=

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

(instance-segmentation-wandb)=

### Weights & Biases

```{important}
Weights & Biases must be installed with `pip install "lightly-train[wandb]"`.
```

The Weights & Biases logger can be configured with the following arguments:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_instance_segmentation(
        out="out/my_experiment",
        model="dinov3/vitl16-eomt-inst-coco",
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

(instance-segmentation-resume-training)=

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

(instance-segmentation-transform-args)=

## Default Image Transform Arguments

The following are the default train transform arguments. The validation arguments are
automatically inferred from the train arguments.

You can configure the image size and normalization like this:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_instance_segmentation(
        out="out/my_experiment",
        model="dinov3/vitl16-eomt-inst-coco",
        data={
            # ...
        }
        transform_args={
            "image_size": (640, 640),     # (height, width)
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    )
```

`````{dropdown} EoMT Instance Segmentation DINOv3 Default Transform Arguments
````{dropdown} Train
```{include} _auto/dinov3eomtinstancesegmentationtrain_train_transform_args.md
```
````
````{dropdown} Val
```{include} _auto/dinov3eomtinstancesegmentationtrain_val_transform_args.md
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
