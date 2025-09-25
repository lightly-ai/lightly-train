(semantic-segmentation)=

# Semantic Segmentation

```{note}
🔥 **New**: LightlyTrain now supports training **[DINOv3](#-use-eomt-with-dinov3-)** and DINOv2 models for semantic segmentation with the `train_semantic_segmentation` function! The method is based on the
state-of-the-art segmentation model [EoMT](https://arxiv.org/abs/2503.19108) by
Kerssies et al. and reaches 59.1% mIoU with DINOv3 weights and 58.4% mIoU with DINOv2 weights on the ADE20k dataset.
```

## Benchmark Results

Below we provide the model checkpoints and report the validation mIoUs and inference FPS of three different DINOv3 models fine-tuned on various datasets with LightlyTrain. We also made the comparison to the results obtained in the original EoMT paper, if available.

The experiments, unless stated otherwise, generally follow the protocol in the original EoMT paper, using a batch size of `16` and a learning rate of `1e-4`. The average FPS values were measured with model compilation using `torch.compile` on a single NVIDIA T4 GPU with FP16 precision.

You can also explore inferencing with these model weights using our Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_semantic_segmentation.ipynb)

### ADE20k

| Backbone Model | #Params (M) | Input Size | Val mIoU | Avg. FPS | Checkpoint |
|----------------|-------------|------------|----------|----------|------------|
| dinov3/vits16-eomt | 21.6 | 512×512 | 0.466 | 48.9 | [link](https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3_eomt/dinov3_eomt_vits16_ade20k.ckpt) |
| dinov3/vitb16-eomt | 85.7 | 512×512 | 0.544 | 48.1 | [link](https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3_eomt/dinov3_eomt_vitb16_ade20k.ckpt) |
| dinov3/vitl16-eomt | 303.2 | 512×512 | **0.591** | 22.6 | [link](https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3_eomt/dinov3_eomt_vitl16_ade20k.ckpt) |
| dinov2/vitl16-eomt (original) | 319 | 512×512 | 0.584 | - | - |

We trained the models with 40k steps and `num_queries=100` , as in the setting of the original EoMT paper.

### COCO-Stuff

| Backbone Model | #Params (M) | Input Size | Val mIoU | Avg. FPS | Checkpoint |
|----------------|-------------|------------|----------|----------|------------|
| dinov3/vits16-eomt | 21.6 | 512×512 | 0.465 | 88.7 | [link](https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3_eomt/lightlytrain_dinov3_eomt_vits16_cocostuff.pt) |
| dinov3/vitb16-eomt | 85.7 | 512×512 | 0.520 | 43.3 | [link](https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3_eomt/lightlytrain_dinov3_eomt_vitb16_cocostuff.pt) |
| dinov3/vitl16-eomt | 303.2 | 512×512 | **0.544** | 20.4 | [link](https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3_eomt/lightlytrain_dinov3_eomt_vitl16_cocostuff.pt) |

We trained with 12 epochs (~88k steps) on the COCO-Stuff dataset with `num_queries=200` for EoMT.

### Cityscapes

| Backbone Model | #Params (M) | Input Size | Val mIoU | Avg. FPS | Checkpoint |
|----------------|-------------|------------|----------|----------|------------|
| dinov3/vits16-eomt | 21.6 | 1024×1024 | 0.786 | 18.6 | [link](https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3_eomt/lightlytrain_dinov3_eomt_vits16_cityscapes.pt) |
| dinov3/vitb16-eomt | 85.7 | 1024×1024 | 0.810 | 8.7 | [link](https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3_eomt/lightlytrain_dinov3_eomt_vitb16_cityscapes.pt) |
| dinov3/vitl16-eomt | 303.2 | 1024×1024 | **0.844** | 3.9 | [link](https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3_eomt/lightlytrain_dinov3_eomt_vitl16_cityscapes.pt) |
| dinov2/vitl16-eomt (original) | 319 | 1024×1024 | 0.842 | - | - |

We trained with 107 epochs (~20k steps) on the Cityscapes dataset with `num_queries=200` for EoMT.

## Semantic Segmentation with EoMT in LightlyTrain

Training a semantic segmentation model with LightlyTrain is straightforward and
only requires a few lines of code. The dataset must follow the [ADE20K format](https://ade20k.csail.mit.edu/)
with RGB images and integer masks in PNG format. See [data](#semantic-segmentation-data)
for more details.

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

You can also train from an already fine-tuned model by loading the weights with the `checkpoint` parameter:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out/my_experiment",
        model="dinov2/vitl14-eomt", 
        checkpoint="/path/to/vitl14-eomt/exported_model.pt",
        data={...},
    )
```

### Load the Trained Model from Checkpoint and Predict

After the training completes you can load the model for inference like this:

```python
import lightly_train

model = lightly_train.load_model_from_checkpoint(
    "out/my_experiment/exported_models/exported_last.pt"
)
masks = model.predict("path/to/image.jpg")
```

### Visualize the Result

And visualize the predicted masks like this:

```python
# ruff: noqa: F821
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

To fine-tune EoMT from DINOv3, you have to [sign up and accept the terms of use](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) from Meta to get access to the DINOv3 checkpoints. After signing up, you will receive an email with the download links. You can then use these links in your training script.

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out/my_experiment",
        model="dinov3/vits16-eomt",
        model_args={
            # Replace with your own url
            "backbone_url": "https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth<SOME-KEY>",
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

See [here](#dinov3-models) for the list of available DINOv3 models.

(semantic-segmentation-output)=

## Out

The `out` argument specifies the output directory where all training logs, model exports,
and checkpoints are saved. It looks like this after training:

```text
out/my_experiment
├── checkpoints
│   └── last.ckpt                                       # Last checkpoint
├── events.out.tfevents.1721899772.host.1839736.0       # TensorBoard logs
└── train.log                                           # Training logs
```

The final model checkpoint is saved to `out/my_experiment/checkpoints/last.ckpt`.

```{tip}
Create a new output directory for each experiment to keep training logs, model exports,
and checkpoints organized.
```

(semantic-segmentation-data)=

## Data

Lightly**Train** supports training semantic segmentation models with images and masks.
Every image must have a corresponding mask with the same filename except for the file
extension. The masks must be PNG images in grayscale integer format, where each pixel
value corresponds to a class ID.

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

The following mask formats are supported:

- png

Example of a directory structure with training and validation images and masks:

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

To train with this folder structure, set the `data` argument like this:

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
    lightly_train.train(
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
            "backbone_weights": "out/my_pretrain_experiment/exported_models/exported_last.pt",
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
    checkpoint="out/my_experiment/exported_models/exported_last.pt",
    height=518,
    width=518
)
```

### Requirements

Exporting to ONNX requires some additional packages to be installed. Namely

- [onnx](https://pypi.org/project/onnx/)
- [onnxruntime](https://pypi.org/project/onnxruntime/) if `verify` is set to `True`.
- [onnxslim](https://pypi.org/project/onnxslim/) if `simplify` is set to `True`.
