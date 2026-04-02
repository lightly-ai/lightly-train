(object-detection)=

# Object Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb)

```{note}
🔥 LightlyTrain now supports training **LTDETR**: **DINOv3**- and **DINOv2**-based object detection models
with the super fast RT-DETR detection architecture! Our largest model achieves an mAP<sub>50:95</sub> of 60.0 on the COCO validation set!
```

(object-detection-benchmark-results)=

## Benchmark Results

Below we provide the model checkpoints and report the validation mAP<sub>50:95</sub> and
inference latency of different DINOv3 and DINOv2-based models, fine-tuned on the COCO
dataset. You can check [here](object-detection-use-model-weights) for how to use these
model checkpoints for further fine-tuning. The average latency values were measured
using TensorRT version `10.13.3.9` and on a Nvidia T4 GPU with batch size 1.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb)

### COCO

| Implementation |               Model               | Val mAP<sub>50:95</sub> | Latency (ms) | Params (M) | Input Size |
| :------------: | :-------------------------------: | :---------------------: | :----------: | :--------: | :--------: |
|  LightlyTrain  |          picodet-s-coco           |         26.7\*          |    2.2\*     |    1.17    |  416×416   |
|  LightlyTrain  |          picodet-l-coco           |         32.0\*          |    2.4\*     |    3.75    |  416×416   |
|  LightlyTrain  |     dinov3/vitt16-ltdetr-coco     |          49.8           |     5.4      |    10.1    |  640×640   |
|  LightlyTrain  |   dinov3/vitt16plus-ltdetr-coco   |          52.5           |     7.0      |    18.1    |  640×640   |
|  LightlyTrain  |     dinov3/vits16-ltdetr-coco     |          55.4           |     10.5     |    36.4    |  640×640   |
|  LightlyTrain  | dinov3/convnext-tiny-ltdetr-coco  |          54.4           |     13.3     |    61.1    |  640×640   |
|  LightlyTrain  | dinov3/convnext-small-ltdetr-coco |          56.9           |     17.7     |    82.7    |  640×640   |
|  LightlyTrain  | dinov3/convnext-base-ltdetr-coco  |          58.6           |     24.7     |   121.0    |  640×640   |
|  LightlyTrain  | dinov3/convnext-large-ltdetr-coco |          60.0           |     42.3     |   230.0    |  640×640   |

\*Picodet models are in preview and we report preliminary results.

## Object Detection with LTDETR

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb)

Training an object detection model with LightlyTrain is straightforward and only
requires a few lines of code. See [data](#object-detection-data) for details on how to
prepare your dataset.

### Train an Object Detection Model

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_object_detection(
        out="out/my_experiment",
        model="dinov3/vitt16-ltdetr-coco",
        data={
            "format": "yolo",
            "path": "my_data_dir",
            "train": "images/train2017",
            "val": "images/val2017",
            "names": {
                0: "person",
                1: "bicycle",
                # ...
            },
            # Optional, classes that are in the dataset but should be ignored during
            # training.
            # "ignore_classes": [0],
            #
            # Optional, skip images without label files. By default, these are included
            # as negative samples.
            # "skip_if_label_file_missing": True,
        }
    )
```

During training, both the

- best (with highest validation mAP<sub>50:95</sub>) and
- last (last validation round as determined by
  `save_checkpoint_args.save_every_num_steps`)

model weights are exported to `out/my_experiment/exported_models/`, unless disabled in
`save_checkpoint_args`. You can use these weights to continue fine-tuning on another
task by loading the weights via `model="<checkpoint path>"`:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_object_detection(
        out="out/my_experiment",
        model="out/my_experiment/exported_models/exported_best.pt", # Use the best model to continue training
        data={...},
    )
```

<!-- TODO (Lionel, 10/25) Add instructions for re-using classification head when it is supported. -->

(object-detection-pretrain-finetune)=

## Pretrain and Fine-tune an Object Detection Model

To further improve the performance of your object detection model, you can first
pretrain a DINOv2 model on unlabeled data using self-supervised learning and then
fine-tune it on your object detection dataset. This is especially useful if your dataset
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
        model="dinov2/vits14-noreg",
        method="dinov2",
    )

    # Fine-tune the DINOv2 model for object detection.
    lightly_train.train_object_detection(
        out="out/my_experiment",
        model="dinov2/vits14-noreg-ltdetr",
        model_args={
            # Path to your pretrained DINOv2 model.
            "backbone_weights": "out/my_pretrain_experiment/exported_models/exported_best.pt",
        },
        data={
            "format": "yolo",
            "path": "my_data_dir",
            "train": "images/train2012",
            "val": "images/val2012",
            "names": {
                0: "person",
                1: "bicycle",
                # ...
            },
        }
    )
```

(object-detection-use-model-weights)=

### Load the Trained Model from Checkpoint and Predict

After the training completes, you can load the best model checkpoints for inference like
this:

```python
import lightly_train

model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
results = model.predict("path/to/image.jpg")
```

Or use one of the models provided by LightlyTrain:

```python
import lightly_train

model = lightly_train.load_model("dinov3/vitt16-ltdetr-coco")
results = model.predict("image.jpg")
results["labels"]   # Class labels, tensor of shape (num_boxes,)
results["bboxes"]   # Bounding boxes in (xmin, ymin, xmax, ymax) absolute pixel
                    # coordinates of the original image. Tensor of shape (num_boxes, 4).
results["scores"]   # Confidence scores, tensor of shape (num_boxes,)
```

### Visualize the Result

After making the predictions with the model weights, you can visualize the predicted
bounding boxes like this:

```python
import matplotlib.pyplot as plt
from torchvision import io, utils

import lightly_train

model = lightly_train.load_model("dinov3/vitt16-ltdetr-coco")
results = model.predict_sahi(image="image.jpg")
results["labels"]   # Class labels, tensor of shape (num_boxes,)
results["bboxes"]   # Bounding boxes in (xmin, ymin, xmax, ymax) absolute pixel
                    # coordinates of the original image. Tensor of shape (num_boxes, 4).
results["scores"]   # Confidence scores, tensor of shape (num_boxes,)

# Visualize predictions.
image_with_boxes = utils.draw_bounding_boxes(
    image=io.read_image("image.jpg"),
    boxes=results["bboxes"],
    labels=[model.classes[i.item()] for i in results["labels"]],
)

fig, ax = plt.subplots(figsize=(30, 30))
ax.imshow(image_with_boxes.permute(1, 2, 0))
fig.savefig("predictions.png")
```

The predicted boxes are in the absolute (x_min, y_min, x_max, y_max) format, i.e.
represent the size of the dimension of the bounding boxes in pixels of the original
image.

### Improving Small Objects Detection

Detecting small objects in high-resolution images can be challenging because they may
occupy only a few pixels when the image is resized to the model’s input resolution. To
address this, we support Slicing Aided Hyper Inference (SAHI) allowing the model to make
predictions from overlapping tiles of the original image at full resolution and then
merge the predictions.

Using tiled inference requires no extra setup:

```python
import lightly_train

model = lightly_train.load_model("dinov3/vitt16-ltdetr-coco")
results = model.predict_sahi(image="image.jpg")
results["labels"]   # Class labels, tensor of shape (num_boxes,)
results["bboxes"]   # Bounding boxes in (xmin, ymin, xmax, ymax) absolute pixel
                    # coordinates of the original image. Tensor of shape (num_boxes, 4).
results["scores"]   # Confidence scores, tensor of shape (num_boxes,)
```

You can customize the behavior via the following parameters:

- `overlap`: Fraction of overlap between neighboring tiles. Higher values increase
  small-object recall but also increase computation.
- `threshold`: Minimum confidence score required to keep a predicted box.
- `nms_iou_threshold`: IoU threshold used for non-maximum suppression when merging
  predictions coming from different tiles.
- `global_local_iou_threshold`: Our SAHI-style inference combines predictions from both
  the *global* (full-image) view and the *local* tiles. To avoid duplicate detections,
  tile predictions are suppressed when they significantly overlap
  (`iou > global_local_iou_threshold`) with a prediction of the same class coming from
  the global view.

<!--
# Figure created with
import lightly_train
import matplotlib.pyplot as plt
from torchvision.io import decode_image
from torchvision.utils import draw_bounding_boxes
import urllib.request

model = lightly_train.load_model("dinov3/convnext-tiny-ltdetr-coco")
img = "http://images.cocodataset.org/val2017/000000577932.jpg"
results = model.predict(img)

urllib.request.urlretrieve(img, "/tmp/image.jpg")
image = decode_image("/tmp/image.jpg")
image_with_boxes = draw_bounding_boxes(
    image,
    boxes=results["bboxes"],
    labels=[model.classes[label.item()] for label in results["labels"]],
)
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(image_with_boxes.permute(1, 2, 0))
ax.axis("off")
fig.savefig("out/preds/det.jpg", bbox_inches="tight")
fig.show()
-->

```{figure} /_static/images/object_detection/street.jpg
```

## Out

The `out` argument specifies the output directory where all training logs, model
exports, and checkpoints are saved. It looks like this after training:

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

The final model checkpoint is saved to `out/my_experiment/checkpoints/last.ckpt`. The
last and best model weights are exported to `out/my_experiment/exported_models/` unless
disabled in `save_checkpoint_args`.

```{tip}
Create a new output directory for each experiment to keep training logs, model exports,
and checkpoints organized.
```

(object-detection-data)=

## Data

Lightly**Train** supports training object detection models with images and bounding
boxes. We support inputs in either the [YOLO](#object-detection-data-yolo) or
[COCO](#object-detection-data-coco) object detection formats.

For more details on LightlyTrain's support for data input, please check the
[Data Input](#data-input) page.

We specify the training data with a `data` dictionary:

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    data={
        "format": ...,           # either "yolo" or "coco"
        "ignore_classes": [...], # optional list of class IDs that should be skipped during training
         # format specific options
    },
)
```

If you would like to skip specific classes during training, add their IDs to the
optional `ignore_classes` list. The trainer omits these classes from loss computation
and the exported model does not predict them.

(object-detection-data-yolo)=

### YOLO format

For the [YOLO](https://labelformat.com/formats/object-detection/yolov5/) format we have
for every image a corresponding label file that has for every object in the image a line
with the class ID and 4 normalized bounding box coordinates (x_center, y_center, width,
height). The file should have the `.txt` extension and an example annotation file for an
image with two objects could look like this:

```text
0 0.716797 0.395833 0.216406 0.147222
1 0.687500 0.379167 0.255208 0.175000
```

Your dataset directory should be organized like this:

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

Alternatively, the splits can also be at the top level:

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

Each class in the dataset must be listed in the `names` dictionary. The keys are the
class IDs used inside the YOLO annotations and the values are the human-readable class
names. All class IDs that appear in the label files must be present in the dictionary;
otherwise Lightly**Train** raises an error when it encounters an unknown class ID.

#### Missing Labels

There are three cases in which an image may not have any corresponding labels:

1. The label file is missing.
1. The label file is empty.
1. The label file only contains annotations for classes that are in `ignore_classes`.

LightlyTrain treats all three cases as "negative" samples and includes the images in
training with an empty list of bounding boxes.

If you would like to exclude images without label files from training, you can set the
`skip_if_label_file_missing` argument in the `data` configuration. This only excludes
images without a label file (case 1) but still includes cases 2 and 3 as negative
samples.

#### Example

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    data={
        "format": "yolo",
        "path": "my_data_dir",
        "train": "images/train",
        "val": "images/val",
        "names": {...},
        "skip_if_label_file_missing": True, # Skip images without label files.
    }
)
```

(object-detection-data-coco)=

### COCO format

For the [COCO](https://labelformat.com/formats/object-detection/coco/) we have for every
split a separate annotations JSON file that specifies which images and classes belong to
the split and also contain the bounding boxes. The structure of such a file is as
follows:

```json
{
    "images": [
        {
            "id": 1,
            "file_name": "image1.jpg"
        },
        {
            "id": 2,
            "file_name": "image2.jpg"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "cat"
        },
        {
            "id": 1,
            "name": "dog"
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 0,
            "bbox": [10, 20, 100, 80]
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [150, 30, 200, 120]
        },
        {
            "id": 3,
            "image_id": 2,
            "category_id": 0,
            "bbox": [5, 10, 90, 70]
        }
    ]
}
```

The `file_name` field can also be an absolute or relative path to an image. One can
optionally specify the `images` directory so that the paths are resolved relatively to
that directory. If it is omitted the paths are resolved relatively to the annotations
file. Furthermore, the `images` path itself is resolved relatively to the annotations
file.

It is good practice to have the same categories for all splits but in order to guarantee
consistency, we always take them from the train split.

The bounding boxes `bbox` are specified in absolute coordinates (pixels) as follows:

```python
[x, y, width, height]
```

#### Missing Labels

There are two cases in which an image may not have any corresponding labels:

1. There are no bounding boxes specified for an image in the annotations file.
1. The annotations file only contains annotations for classes that are in
   `ignore_classes`.

LightlyTrain treats both cases as "negative" samples and includes the images in training
with an empty list of bounding boxes.

If you would like to exclude images without bounding boxes from training, you can set
the `skip_if_annotations_missing` argument in the `data` configuration. This only
excludes images without bounding boxes (case 1) but still includes case 2 as negative
samples.

#### Example

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    data={
        "format": "coco",
        "train": {"annotations": "train_labels.json", "images": "train_images/"},
        "val": {"annotations": "val_labels.json", "images": "val_images/"},
        "skip_if_annotations_missing": True, # Skip images without bounding boxes
    }
)
```

If in this particular example we specified `file_name` like this in the train annotation
file

```json
{
    "id": 1,
    "file_name": "train_images/image1.jpg"
}
```

we could also omit `images`.

### Image Formats

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

(object-detection-model)=

## Model

The `model` argument defines the model used for object detection training. The following
models are available:

### PicoDet Models

- `picodet-s-coco` (pretrained on COCO)
- `picodet-l-coco` (pretrained on COCO)

### LTDETR DINOv3 Models

- `dinov3/vitt16-ltdetr-coco` (pretrained on COCO)
- `dinov3/vitt16plus-ltdetr-coco` (pretrained on COCO)
- `dinov3/vits16-ltdetr-coco` (pretrained on COCO)
- `dinov3/convnext-tiny-ltdetr-coco` (pretrained on COCO)
- `dinov3/convnext-small-ltdetr-coco` (pretrained on COCO)
- `dinov3/convnext-base-ltdetr-coco` (pretrained on COCO)
- `dinov3/convnext-large-ltdetr-coco` (pretrained on COCO)
- `dinov3/vitt16-ltdetr`
- `dinov3/vitt16plus-ltdetr`
- `dinov3/vits16-ltdetr`
- `dinov3/vitb16-ltdetr`
- `dinov3/vitl16-ltdetr`
- `dinov3/convnext-tiny-ltdetr`
- `dinov3/convnext-small-ltdetr`
- `dinov3/convnext-base-ltdetr`
- `dinov3/convnext-large-ltdetr`

All models are
[pretrained by Meta](https://github.com/facebookresearch/dinov3/tree/main?tab=readme-ov-file#pretrained-models).

### LTDETR DINOv2 Models

- `dinov2/vits14-ltdetr`
- `dinov2/vitb14-ltdetr`
- `dinov2/vitl14-ltdetr`
- `dinov2/vitg14-ltdetr`

All models are
[pretrained by Meta](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models).

## Training Settings

See [](train-settings) on how to configure training settings.

(object-detection-logging)=

(object-detection-tensorboard)=

(object-detection-mlflow)=

(object-detection-wandb)=

## Logging

See [](train-settings-logging) on how to configure logging.

(object-detection-resume-training)=

## Resume Training

See [](train-settings-resume-training) on how to resume training.

(object-detection-transform-args)=

## Default Image Transform Arguments

The following are the default image transform arguments. See
[](train-settings-transforms) on how to customize transforms.

`````{dropdown} DINOv3 LTDETR Default Transform Arguments
````{dropdown} Train
```{include} _auto/dinov3ltdetrobjectdetectiontrain_train_transform_args.md
```
````
````{dropdown} Val
```{include} _auto/dinov3ltdetrobjectdetectiontrain_val_transform_args.md
```
````
`````

`````{dropdown} DINOv2 LTDETR Default Transform Arguments
````{dropdown} Train
```{include} _auto/dinov2ltdetrobjectdetectiontrain_train_transform_args.md
```
````
````{dropdown} Val
```{include} _auto/dinov2ltdetrobjectdetectiontrain_val_transform_args.md
```
````
`````

`````{dropdown} PicoDet Default Transform Arguments
````{dropdown} Train
```{include} _auto/picodetobjectdetectiontrain_train_transform_args.md
```
````
````{dropdown} Val
```{include} _auto/picodetobjectdetectiontrain_val_transform_args.md
```
````
`````

(object-detection-onnx)=

## Exporting a Checkpoint to ONNX

[Open Neural Network Exchange (ONNX)](https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange)
is a standard format for representing machine learning models in a framework independent
manner. In particular, it is useful for deploying our models on edge devices where
PyTorch is not available.

### Requirements

Exporting to ONNX requires some additional packages to be installed. Namely

- [onnx](https://pypi.org/project/onnx/)
- [onnxruntime](https://pypi.org/project/onnxruntime/) if `verify` is set to `True`.
- [onnxslim](https://pypi.org/project/onnxslim/) if `simplify` is set to `True`.

You can install them with:

```bash
pip install "lightly-train[onnx,onnxruntime,onnxslim]"
```

The following example shows how to export a previously trained model to ONNX.

```python
import lightly_train

# Instantiate the model from a checkpoint.
model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")

# Export to ONNX.
model.export_onnx(
    out="out/my_experiment/exported_models/model.onnx"
    # precision="fp16", # Export model with FP16 weights for smaller size and faster inference.
)
```

See {py:meth}`~.DINOv3LTDETRObjectDetection.export_onnx` for all available options when
exporting to ONNX.

The following notebook shows how to export a model to ONNX in Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection_export.ipynb)

(object-detection-tensorrt)=

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

# Instantiate the model from a checkpoint.
model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")

# Export to TensorRT from an ONNX file.
model.export_tensorrt(
    out="out/my_experiment/exported_models/model.trt", # TensorRT engine destination.
    # precision="fp16", # Export model with FP16 weights for smaller size and faster inference.
)
```

See {py:meth}`~.DINOv3LTDETRObjectDetection.export_tensorrt` for all available options
when exporting to TensorRT.

You can also learn more about exporting LTDETR to TensorRT using our Colab notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection_export.ipynb)
