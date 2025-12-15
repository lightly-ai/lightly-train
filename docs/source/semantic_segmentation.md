(semantic-segmentation)=

# Semantic Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_semantic_segmentation.ipynb)

```{note}
🔥 **New**: LightlyTrain now supports training **[DINOv3](#-use-eomt-with-dinov3-)** and DINOv2 models for semantic segmentation with the `train_semantic_segmentation` function! The method is based on the
state-of-the-art segmentation model [EoMT](https://arxiv.org/abs/2503.19108) by
Kerssies et al. and reaches 59.1% mIoU with DINOv3 weights and 58.4% mIoU with DINOv2 weights on the ADE20k dataset.
```

(semantic-segmentation-benchmark-results)=

## Benchmark Results

Below we provide the model checkpoints and report the validation mIoUs and inference latency of three different DINOv3 models fine-tuned on various datasets with LightlyTrain. We also made the comparison to the results obtained in the original EoMT paper, if available. You can check [here](semantic-segmentation-eomt-dinov3-model-weights) for how to use these model checkpoints for further fine-tuning.

The experiments, unless stated otherwise, generally follow the protocol in the original EoMT paper, using a batch size of `16` and a learning rate of `1e-4`. The average latency values were measured with model compilation using `torch.compile` on a single NVIDIA T4 GPU with FP16 precision.

You can also explore inferencing with these model weights using our Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_semantic_segmentation.ipynb)

### COCO-Stuff

| Implementation | Model | Val mIoU | Avg. Latency (ms) | Params (M) | Input Size |
|----------------|----------------------------|----------|----------|-----------|------------|
| LightlyTrain | dinov3/vits16-eomt-coco | 46.5 | 11.3 | 21.6 | 512×512 |
| LightlyTrain | dinov3/vitb16-eomt-coco | 52.0 | 23.1 | 85.7 | 512×512 |
| LightlyTrain | dinov3/vitl16-eomt-coco | **54.4** | 49.0 | 303.2 | 512×512 |

We trained with 12 epochs (~88k steps) on the COCO-Stuff dataset with `num_queries=200` for EoMT.

### Cityscapes

| Implementation | Model | Val mIoU | Avg. Latency (ms) | Params (M) | Input Size |
|----------------|--------------------------------------|----------|----------|-----------|------------|
| LightlyTrain | dinov3/vits16-eomt-cityscapes | 78.6 | 53.8 | 21.6 | 1024×1024 |
| LightlyTrain | dinov3/vitb16-eomt-cityscapes | 81.0 | 114.9 | 85.7 | 1024×1024 |
| LightlyTrain | dinov3/vitl16-eomt-cityscapes | **84.4** | 256.4 | 303.2 | 1024×1024 |
| Original EoMT | dinov2/vitl16-eomt | 84.2 | - | 319 | 1024×1024 |

We trained with 107 epochs (~20k steps) on the Cityscapes dataset with `num_queries=200` for EoMT.

## Semantic Segmentation with EoMT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_semantic_segmentation.ipynb)

Training a semantic segmentation model with LightlyTrain is straightforward and
only requires a few lines of code. See [data](#semantic-segmentation-data)
for more details on how to prepare your dataset.

### Train a Semantic Segmentation Model

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out/my_experiment",
        model="dinov2/vitl14-eomt", 
        data={
            "train": {
                "images": "my_data_dir/train/images",   # Path to training images
                "masks": "my_data_dir/train/masks",     # Path to training masks
            },
            "val": {
                "images": "my_data_dir/val/images",     # Path to validation images
                "masks": "my_data_dir/val/masks",       # Path to validation masks
            },
            "classes": {                                # Classes in the dataset                    
                0: "background",
                1: "car",
                2: "bicycle",
                # ...
            },
            # Optional, classes that are in the dataset but should be ignored during
            # training.
            "ignore_classes": [0], 
        },
    )
```

During training, both the

- best (with highest validation mIoU) and
- last (last validation round as determined by `save_checkpoint_args.save_every_num_steps`)

model weights are exported to `out/my_experiment/exported_models/`, unless disabled in `save_checkpoint_args`. You can use these weights to continue fine-tuning on another task by loading the weights via the `checkpoint` parameter:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out/my_experiment",
        model="out/my_experiment/exported_models/exported_best.pt",  # Continue training from the best model
        data={...},
    )
```

```{note}
Check [here](semantic-segmentation-eomt-dinov3-model-weights) for how to use the LightlyTrain model checkpoints for further fine-tuning.
```

### Load the Trained Model from Checkpoint and Predict

After the training completes, you can load the best model checkpoints for inference like this:

```python
import lightly_train

model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
masks = model.predict("path/to/image.jpg")
# Masks is a tensor of shape (height, width) with class labels as values.
# It has the same height and width as the input image.
```

### Visualize the Result

After making the predictions with the model weights, you can visualize the predicted masks like this:

```python skip_ruff
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks

image = read_image("path/to/image.jpg")
masks = torch.stack([masks == class_id for class_id in masks.unique()])
image_with_masks = draw_segmentation_masks(image, masks, alpha=0.6)
plt.imshow(image_with_masks.permute(1, 2, 0))
```

The predicted masks have shape `(height, width)` and each value corresponds to a class
ID as defined in the `classes` dictionary in the dataset.

(semantic-segmentation-eomt-dinov3)=

## 🔥 Use EoMT with DINOv3 🔥

To fine-tune EoMT from DINOv3, you have to set `model` to one of the [DINOv3 models](#dinov3-models).

```{note}
DINOv3 models are released under the [DINOv3 license](https://github.com/lightly-ai/lightly-train/blob/main/licences/DINOv3_LICENSE.md).
```

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out/my_experiment",
        model="dinov3/vits16-eomt",
        data={
            "train": {
                "images": "my_data_dir/train/images",   # Path to training images
                "masks": "my_data_dir/train/masks",     # Path to training masks
            },
            "val": {
                "images": "my_data_dir/val/images",     # Path to validation images
                "masks": "my_data_dir/val/masks",       # Path to validation masks
            },
            "classes": {                                # Classes in the dataset                    
                0: "background",
                1: "car",
                2: "bicycle",
                # ...
            },
            # Optional, classes that are in the dataset but should be ignored during
            # training.
            "ignore_classes": [0], 
        },
    )
```

(semantic-segmentation-eomt-dinov3-model-weights)=

### Use the LightlyTrain Model Checkpoints

Now you can also start with the DINOv3 model checkpoints that LightlyTrain provides.
The models are listed [here](#semantic-segmentation-benchmark-results) in the
"Model" column of the tables.

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out/my_experiment",
        model="dinov3/vits16-eomt-coco", # Use the COCO-Stuff model checkpoint for further fine-tuning
        data={...},
    )
```

(semantic-segmentation-output)=

## Out

The `out` argument specifies the output directory where all training logs, model exports,
and checkpoints are saved. It looks like this after training:

```text
out/my_experiment
├── checkpoints
│   └── last.ckpt                                       # Last checkpoint
├── exported_models
|   └── exported_last.pt                                # Last model exported (unless disabled)
|   └── exported_best.pt                                # Best model exported (unless disabled)
├── events.out.tfevents.1721899772.host.1839736.0       # TensorBoard logs
└── train.log                                           # Training logs
```

The final model checkpoint is saved to `out/my_experiment/checkpoints/last.ckpt`. The last and best model weights are exported to `out/my_experiment/exported_models/` unless disabled in `save_checkpoint_args`.

```{tip}
Create a new output directory for each experiment to keep training logs, model exports,
and checkpoints organized.
```

(semantic-segmentation-data)=

## Data

Lightly**Train** supports training semantic segmentation models with images and masks.
Every image must have a corresponding mask whose filename either matches that of the image (under a different directory) or follows a specific template pattern. The masks must be PNG images in either grayscale integer format, where each pixel value corresponds to a class ID, or multi-channel (e.g., RGB) format.

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
- dcm (DICOM)

For more details on LightlyTrain's support for data input, please check the [Data Input](#data-input) page.

The following mask formats are supported:

- png

### Specify Mask Filepaths

We support two ways of specifying the mask filepaths in relation to the image filepaths:

1. Using the same filename as the image but under a different directory.
1. Using a template against the image filepath.

#### Using the Same Filename as the Image

We support loading masks that share the same filenames as their corresponding images under different directories. Here is an example of such a directory structure with training and validation images and masks:

```bash
my_data_dir
├── train
│   ├── images
│   │   ├── image0.jpg
│   │   └── image1.jpg
│   └── masks
│       ├── image0.png
│       └── image1.png
└── val
    ├── images
    |  ├── image2.jpg
    |  └── image3.jpg
    └── masks
       ├── image2.png
       └── image3.png
```

To train with this directory structure, set the `data` argument like this:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out/my_experiment",
        model="dinov2/vitl14-eomt",
        data={
            "train": {
                "images": "my_data_dir/train/images",   # Path to training images
                "masks": "my_data_dir/train/masks",     # Path to training masks
            },
            "val": {
                "images": "my_data_dir/val/images",     # Path to validation images
                "masks": "my_data_dir/val/masks",       # Path to validation masks
            },
            "classes": {                                # Classes in the dataset                    
                0: "background",
                1: "car",
                2: "bicycle",
                # ...
            },
            # Optional, classes that are in the dataset but should be ignored during
            # training.
            "ignore_classes": [0], 
        },
    )
```

The classes in the dataset must be specified in the `classes` dictionary. The keys
are the class IDs and the values are the class names. The class IDs must be identical to
the values in the mask images. All possible class IDs must be specified, otherwise
Lightly**Train** will raise an error if an unknown class ID is encountered. If you would
like to ignore some classes during training, you specify their class IDs in the
`ignore_classes` argument. The trained model will then not predict these classes.

#### Using a Template against the Image Filepath

We also support loading masks that follow a certain template against the corresponding image filepath. For example, if you have the following directory structure:

```bash
my_data_dir
├── train
│   ├── images
│   │   ├── image0.jpg
│   │   └── image1.jpg
│   └── masks
│       ├── image0_mask.png
│       └── image1_mask.png
└── val
    ├── images
    |  ├── image2.jpg
    |  └── image3.jpg
    └── masks
       ├── image2_mask.png
       └── image3_mask.png
```

you could set the `data` argument like this:

```python
import lightly_train

if __name__ == "__main__":
    mask_file = "{image_path.parent.parent}/masks/{image_path.stem}_mask.png"
    lightly_train.train_semantic_segmentation(
        out="out/my_experiment",
        model="dinov2/vitl14-eomt",
        data={
            "train": {
                "images": "my_data_dir/train/images",
                "masks": mask_file,  # This will match masks with the same filename stem as the training image but with a `_mask` suffix in "my_data_dir/train/masks" in PNG format
            },
            "val": {
                "images": "my_data_dir/val/images", 
                "masks": mask_file,  # This will match masks with the same filename stem as the training image but with a `_mask` suffix in "my_data_dir/val/masks" in PNG format
            },
            "classes": {             # Classes in the dataset                    
                0: "background",
                1: "car",
                2: "bicycle",
                # ...
            },
            # Optional, classes that are in the dataset but should be ignored during
            # training.
            "ignore_classes": [0], 
        },
    )
```

The template string always uses `image_path` (of type `pathlib.Path`) to refer to the filepath of the corresponding image. Only this parameter is allowed in the template string, which is used to calculate the mask filepath.

### Specify Training Classes

We support two mask formats:

- Single-channel integer masks, where each integer value determines a label
- Multi-channel masks (e.g., RGB masks), where each pixel value determines a label

Use the `classes` dict in the `data` dict to map class IDs to labels. In this document, a **class ID** is a key in the `classes` dictionary and a **label** is its value.

#### Using Integer Masks

For single-channel integer masks (each pixel value is a label), the default is a direct mapping from class IDs to label names:

```
    "classes": {                   
        0: "background",
        1: "car",
        2: "bicycle",
        # ...
    },
```

Alternatively, to merge multiple labels into one class during training, use a dictionary like the following:

```
    "classes": {
        0: "background",
        1: {"name": "vehicle", "labels": [1, 2]}, # Merge label 1 and 2 into "vehicle" with class ID 1
        # ...
    },
```

Or:

```
    "classes": {
        0: {"name": "background", "labels": [0]},
        1: {"name": "vehicle", "labels": [1, 2]},
        # ...
    },
```

It is fine if original labels coincide with class IDs, as in the example. Only the class IDs are used for the internal classes for training and prediction masks. Note that each label can map to only **one** class ID.

#### Using Multi-channel Masks

For multi-channel masks, specify pixel values as lists of integer tuples (type `list[tuple[int, ...]]`) in the `"labels"` field:

```
    "classes": {
        0: {"name": "unlabeled", "labels": [(0, 0, 0), (255, 255, 255)]}, # (0,0,0) and (255,255,255) will be mapped to class "unlabeled" with class ID 0
        1: {"name": "road", "labels": [(128, 128, 128)]},
    },
```

These pixel values are converted to class IDs internally during training. Predictions are single-channel masks with those class IDs. Again, each label can map to only **one** class ID, and you cannot mix integer and tuple-valued labels in a single `classes` dictionary.

(semantic-segmentation-model)=

## Model

The `model` argument defines the model used for semantic segmentation training. The
following models are available:

### DINOv3 Models

- `dinov3/vits16-eomt`
- `dinov3/vits16plus-eomt`
- `dinov3/vitb16-eomt`
- `dinov3/vitl16-eomt`
- `dinov3/vitl16plus-eomt`
- `dinov3/vith16plus-eomt`
- `dinov3/vit7b16-eomt`

All DINOv3 models are [pretrained by Meta](https://github.com/facebookresearch/dinov3/tree/main?tab=readme-ov-file#pretrained-models).

### DINOv2 Models

- `dinov2/vits14-eomt`
- `dinov2/vitb14-eomt`
- `dinov2/vitl14-eomt`
- `dinov2/vitg14-eomt`

All DINOv2 models are [pretrained by Meta](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models).

(semantic-segmentation-logging)=

## Logging

Logging is configured with the `logger_args` argument. The following loggers are
supported:

- [`mlflow`](#mlflow): Logs training metrics to MLflow (disabled by
  default, requires MLflow to be installed)
- [`tensorboard`](#tensorboard): Logs training metrics to TensorBoard (enabled by
  default, requires TensorBoard to be installed)
- [`wandb`](semantic-segmentation-wandb): Logs training metrics to Weights & Biases (disabled by
  default, requires wandb to be installed)

(semantic-segmentation-mlflow)=

### MLflow

```{important}
MLflow must be installed with `pip install "lightly-train[mlflow]"`.
```

The mlflow logger can be configured with the following arguments:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out/my_experiment",
        model="dinov2/vitl14-eomt",
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

(semantic-segmentation-tensorboard)=

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

(semantic-segmentation-wandb)=

### Weights & Biases

```{important}
Weights & Biases must be installed with `pip install "lightly-train[wandb]"`.
```

The Weights & Biases logger can be configured with the following arguments:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out/my_experiment",
        model="dinov3/vitl16-eomt-coco",
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

## Resume Training

Like in pretraining, there are two distinct ways to continue training, depending on your
intention. Therefore, the `resume_interrupted=True` parameter cannot be combined with
passing a checkpoint path to the `model` parameter.

### Resume Interrupted Training

Use `resume_interrupted=True` to **resume a previously interrupted or crashed training run**.
This will pick up exactly where the training left off.

- You **must use the same `out` directory** as the original run.
- You **must not change any training parameters** (e.g., learning rate, batch size, data, etc.).
- This is intended for continuing the *same* run without modification.

This will utilize the `.ckpt` checkpoint file `out/my_experiment/checkpoints/last.ckpt` to restore the entire training state, including model weights, optimizer state, and epoch count.

### Load Weights for a New Run

As stated above, you can specify `model="<checkpoint path">` to further fine-tune a model from a previous run.

- You are free to **change training parameters**.
- This is useful for continuing training with a different setup.

We recommend using the exported best model weights from `out/my_experiment/exported_models/exported_best.pt` for this purpose, though a `.ckpt` file can also be loaded.

(semantic-segmentation-pretrain-finetune)=

## Pretrain and Fine-tune a Semantic Segmentation Model

To further improve the performance of your semantic segmentation model, you can first
pretrain a DINOv2 model on unlabeled data using self-supervised learning and then
fine-tune it on your segmentation dataset. This is especially useful if your dataset
is only partially labeled or if you have access to a large amount of unlabeled data.

The following example shows how to pretrain and fine-tune the model. Check out the page
on [DINOv2](#methods-dinov2) to learn more about pretraining DINOv2 models on unlabeled
data.

```python
import lightly_train

if __name__ == "__main__":
    # Pretrain a DINOv2 model.
    lightly_train.pretrain(
        out="out/my_pretrain_experiment",
        data="my_pretrain_data_dir",
        model="dinov2/vitl14",
        method="dinov2",
    )

    # Fine-tune the DINOv2 model for semantic segmentation.
    lightly_train.train_semantic_segmentation(
        out="out/my_experiment",
        model="dinov2/vitl14-eomt",
        model_args={
            # Path to your pretrained DINOv2 model.
            "backbone_weights": "out/my_pretrain_experiment/exported_models/exported_best.pt",
        },
        data={
            "train": {
                "images": "my_data_dir/train/images",   # Path to training images
                "masks": "my_data_dir/train/masks",     # Path to training masks
            },
            "val": {
                "images": "my_data_dir/val/images",     # Path to validation images
                "masks": "my_data_dir/val/masks",       # Path to validation masks
            },
            "classes": {                                # Classes in the dataset                    
                0: "background",
                1: "car",
                2: "bicycle",
                # ...
            },
            # Optional, classes that are in the dataset but should be ignored during
            # training.
            "ignore_classes": [0], 
        },
    )
```

(semantic-segmentation-transform-arguments)=

## Default Image Transform Arguments

The following are the default train transform arguments for EoMT. The validation
arguments are automatically inferred from the train arguments. Specifically the
image size and normalization are shared between train and validation.

You can configure the image size and normalization like this:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out/my_experiment",
        model="dinov2/vitl14-eomt",
        data={
            "train": {
                "images": "my_data_dir/train/images",   # Path to training images
                "masks": "my_data_dir/train/masks",     # Path to training masks
            },
            "val": {
                "images": "my_data_dir/val/images",     # Path to validation images
                "masks": "my_data_dir/val/masks",       # Path to validation masks
            },
            "classes": {                                # Classes in the dataset                    
                0: "background",
                1: "car",
                2: "bicycle",
                # ...
            },
            # Optional, classes that are in the dataset but should be ignored during
            # training.
            "ignore_classes": [0], 
        },
        transform_args={
            "image_size": (518, 518), # (height, width)
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    )
```

`````{dropdown} EoMT DINOv2 Default Transform Arguments
````{dropdown} Train
```{include} _auto/dinov2eomtsemanticsegmentationtrain_train_transform_args.md
```
````
````{dropdown} Val
```{include} _auto/dinov2eomtsemanticsegmentationtrain_val_transform_args.md
```
````
`````

`````{dropdown} EoMT DINOv3 Default Transform Arguments
````{dropdown} Train
```{include} _auto/dinov3eomtsemanticsegmentationtrain_train_transform_args.md
```
````
````{dropdown} Val
```{include} _auto/dinov3eomtsemanticsegmentationtrain_val_transform_args.md
```
````
`````

In case you need different parameters for training and validation, you can pass an
optional `val` dictionary to `transform_args` to override the validation parameters:

```python
transform_args={
    "image_size": (518, 518), # (height, width)
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "val": {    # Override validation parameters
        "image_size": (512, 512), # (height, width)
    }
}
```

## Exporting a Checkpoint to ONNX

[Open Neural Network Exchange (ONNX)](https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange) is a standard format
for representing machine learning models in a framework independent manner. In particular, it is useful for deploying our
models on edge devices where PyTorch is not available.

Currently, we support exporting as ONNX for DINOv2 EoMT segmentation models. The support for DINOv3 EoMT will be released in the short term.

The following example shows how to export a previously trained checkpoint to ONNX using the `export_onnx` function.

```python
import lightly_train

lightly_train.export_onnx(
    out="model.onnx",
    checkpoint="out/my_experiment/exported_models/exported_best.pt",
    height=518,
    width=518
)
```

### Requirements

Exporting to ONNX requires some additional packages to be installed. Namely

- [onnx](https://pypi.org/project/onnx/)
- [onnxruntime](https://pypi.org/project/onnxruntime/) if `verify` is set to `True`.
- [onnxslim](https://pypi.org/project/onnxslim/) if `simplify` is set to `True`.
