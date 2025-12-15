(quick-start-distillation)=

# Quick Start - Distillation

```{image} https://colab.research.google.com/assets/colab-badge.svg
:target: https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/distillation.ipynb
```

This guide demonstrates how to pretrain a model on **unlabeled data** with
distillation. Distillation is a special form of pretraining where a large,
pretrained teacher model, like DINOv2 or DINOv3, is used to guide the training of
a smaller student model. This is the ideal starting point if you want to improve
performance of any model that is not already a large vision foundation model, like
YOLO, ConvNet, or special transformer architectures.

The quick start covers the following steps:

1. Install Lightly**Train**
1. Prepare your unlabeled dataset
1. Pretrain a model with distillation
1. Fine-tune the pretrained model on a downstream task

## Installation

```bash
pip install lightly-train
```

```{important}
Lightly**Train** is officially supported on:
- Linux: CPU or CUDA
- MacOS: CPU only
- Windows (experimental): CPU or CUDA

We are planning to support MPS for MacOS.

Check the [installation instructions](installation.md#installation) for more
details.
```

## Prepare Data

You can use any image dataset for training. No labels are required, and the
dataset can be structured in any way, including subdirectories. If you don't have
a dataset at hand, you can download an example dataset:

```bash
wget https://github.com/lightly-ai/coco128_unlabeled/releases/download/v0.0.1/coco128_unlabeled.zip && unzip -q coco128_unlabeled.zip
```

See the [data guide](pretrain-data) for more information on
supported data formats.

In this example, the dataset looks like this:

```text
coco128_unlabeled
└── images
    ├── 000000000009.jpg
    ├── 000000000025.jpg
    ├── ...
    └── 000000000650.jpg
```

## Pretrain with Distillation

Once the data is ready, you can pretrain the model like this:

```python
import lightly_train

# Pretrain the model
lightly_train.pretrain(
    out="out/my_experiment",  # Output directory
    data="coco128_unlabeled",  # Directory with images
    model="dinov3/vitt16",  # Model to train
    method="distillation",  # Pretraining method
    method_args={
        "teacher": "dinov3/vits16"  # Teacher model for distillation
    },
    epochs=5,  # Small number of epochs for demonstration
    batch_size=32,  # Small batch size for demonstration
)
```

```{note}
This is a minimal example for illustration purposes. In practice you would want to use a
larger dataset (>=10'000 images), more epochs (>=100, ideally ~1000), and a larger
batch size (>=128). For pretraining larger models than `dinov3/vitt16` we also recommend
using a larger teacher model and setting `method="distillationv1"`.
```

```{tip}
Lightly**Train** supports many [popular models](pretrain_distill/models/index.md)
out of the box.
```

This pretrains a tiny DINOv3 ViT model using images from `coco128_unlabeled`. All training
logs, model exports, and checkpoints are saved to the output directory at
`out/my_experiment`.

Once the training is complete, the `out/my_experiment` directory contains the
following files:

```text
out/my_experiment
├── checkpoints
│   ├── epoch=03-step=123.ckpt          # Intermediate checkpoint
│   └── last.ckpt                       # Last checkpoint
├── events.out.tfevents.123.0           # Tensorboard logs
├── exported_models
|   └── exported_last.pt                # Final model exported
├── metrics.jsonl                       # Training metrics
└── train.log                           # Training logs
```

The final model is exported to `out/my_experiment/exported_models/exported_last.pt`
in the default format of the used library. It can directly be used for fine-tuning. See
[export format](pretrain_distill/export.md#format) for more information on how to
export models to other formats or on how to export intermediate checkpoints.

While the trained model has already learned good representations of the images, it
cannot yet make any predictions for tasks such as classification, detection, or
segmentation. To solve these tasks, the model needs to be fine-tuned on a labeled
dataset.

## Fine-Tune

Now the model is ready for fine-tuning! You can use your favorite library for this
step. We'll use Lightly**Train**'s built-in fine-tuning for object detection as an
example.

### Prepare Labeled Data

A labeled dataset is required for fine-tuning. You can download an example dataset
from here:

```bash
wget https://github.com/lightly-ai/coco128_yolo/releases/download/v0.0.1/coco128_yolo.zip && unzip -q coco128_yolo.zip
```

The dataset looks like this after the download completes:

```text
coco128_yolo
├── images
│   ├── train2017
│   │   ├── 000000000009.jpg
│   │   ├── 000000000025.jpg
│   │   ├── ...
│   │   └── 000000000650.jpg
│   └── val2017
│       ├── 000000000139.jpg
│       ├── 000000000285.jpg
│       ├── ...
│       └── 000000013201.jpg
└── labels
    ├── train2017
    │   ├── 000000000009.txt
    │   ├── 000000000025.txt
    │   ├── ...
    │   └── 000000000659.txt
    └── val2017
        ├── 000000000139.txt
        ├── 000000000285.txt
        ├── ...
        └── 000000013201.txt
```

### Fine-Tune the Pretrained Model

Once the dataset is ready, you can fine-tune the pretrained model like this:

```python
import lightly_train

lightly_train.train_object_detection(
    out="out/my_finetune_experiment",
    model="dinov3/vitt16-ltdetr",
    model_args={
        # Load the pretrained weights.
        "backbone_weights": "out/my_experiment/exported_models/exported_last.pt",
    },
    steps=100,  # Small number of steps for demonstration, default is 90_000.
    batch_size=4,  # Small batch size for demonstration, default is 16.
    data={
        "path": "coco128_yolo",
        "train": "images/train2017",
        "val": "images/val2017",
        "names": {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
        },
    },
)
```

This will load the pretrained model from
`out/my_experiment/exported_models/exported_last.pt` and fine-tune it on a subset
of the labeled COCO dataset for 100 steps.

Congratulations! You've just trained and fine-tuned a model using
Lightly**Train**!

## Generate Embeddings

Instead of fine-tuning the model, you can also use it to generate image embeddings.
This is useful for clustering, retrieval, or visualization tasks. The `embed`
command generates embeddings for all images in a directory:

```python
import lightly_train

lightly_train.embed(
    out="my_embeddings.pth",  # Exported embeddings
    checkpoint="out/my_experiment/checkpoints/last.ckpt",  # LightlyTrain checkpoint
    data="coco128_unlabeled",  # Directory with images
)
```

The embeddings are saved to `my_embeddings.pth`. Let's load them and take a look:

```python
import torch

embeddings = torch.load("my_embeddings.pth")
print("First five filenames:")
print(embeddings["filenames"][:5])  # Print first five filenames
print("\nEmbedding tensor shape:")
print(
    embeddings["embeddings"].shape
)  # Tensor with embeddings with shape (num_images, embedding_dim)
```

## Next Steps

- [Object Detection Quick Start](quick-start-object-detection): If you want to
  learn more about fine-tuning and how to use the fine-tuned model for inference.
- [Distillation Guide](pretrain-distill): If you want to learn more about
  distillation and how to pretrain any model with it.
- [DINOv2 Pretraining](methods-dinov2): If you want to learn how to pretrain
  foundation models.
