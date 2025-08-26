(semantic-segmentation)=

# Semantic Segmentation

```{note}
🔥 **New**: LightlyTrain now supports training **[DINOv3](#-use-eomt-with-dinov3-)** and DINOv2 models for semantic segmentation with the `train_semantic_segmentation` function! The method is based on the
state-of-the-art segmentation model [EoMT](https://arxiv.org/abs/2503.19108) by
Kerssies et al. and reaches 58.4% mIoU on the ADE20k dataset with DINOv2 weights.
```

Training a semantic segmentation model with LightlyTrain is straightforward and
only requires a few lines of code. The dataset must follow the [ADE20K format](https://ade20k.csail.mit.edu/)
with RGB images and integer masks in PNG format. See [data](#semantic-segmentation-data)
for more details.

````{tab} Python
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
````

After the training completes you can load the model for inference like this:

```python
import lightly_train

model = lightly_train.load_model_from_checkpoint(
    "out/my_experiment/checkpoints/last.ckpt"
)
masks = model.predict("path/to/image.jpg")
```

And visualize the predicted masks like this:

```python
import torch
import matplotlib.pyplot as plt
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

````{tab} Python
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
````

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

````{tab} Python
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
````

(semantic-segmentation-tensorboard)=

### TensorBoard

TensorBoard logs are automatically saved to the output directory. Run TensorBoard in
a new terminal to visualize the training progress:

```bash
tensorboard --logdir out/my_experiment
```

Disable the TensorBoard logger with:

````{tab} Python
```python
logger_args={"tensorboard": None}
```
````

(semantic-segmentation-pretrain-finetune)=

## Pretrain and Fine-tune a Semantic Segmentation Model

To further improve the performance of your semantic segmentation model, you can first
pretrain a DINOv2 model on unlabeled data using self-supervised learning and then
fine-tune it on your segmentation dataset. This is especially useful if your dataset
is only partially labeled or if you have access to a large amount of unlabeled data.

The following example shows how to pretrain and fine-tune the model. Check out the page
on [DINOv2](#methods-dinov2) to learn more about pretraining DINOv2 models on unlabeled
data.

````{tab} Python
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
````

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
