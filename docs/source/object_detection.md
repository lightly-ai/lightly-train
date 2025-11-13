(object-detection)=

# Object Detection

```{note}
🔥 LightlyTrain now supports training **LT-DETR**: **DINOv3**- and **DINOv2**-based object detection models
with the super fast RT-DETR detection architecture! Our largest model achieves an mAP<sub>50:95</sub> of 60.0 on the COCO validation set!
```

(object-detection-benchmark-results)=

## Benchmark Results

Below we provide the model checkpoints and report the validation mAP<sub>50:95</sub> and inference FPS of different DINOv3 and DINOv2-based models, fine-tuned on the COCO dataset. You can check [here](object-detection-use-model-weights) for how to use these model checkpoints for further fine-tuning. The average FPS values were measured using TensorRT in the version `10.13.3.9` and on a Nvidia T4 GPU with batch size 1.

<!-- TODO (Lionel, 10/25) Add Notebook for OD. -->

### COCO

| Implementation | Backbone Model | AP<sub>50:95</sub> | Latency (ms) | # Params (M) | Input Size | Checkpoint Name |
|:--------------:|:----------------------------:|:------------------:|:------------:|:------------:|:----------:|:---------------------------------:|
| LightlyTrain | dinov2/vits14-ltdetr | 55.7 | 16.87 | 55.3 | 644×644 | dinov2/vits14-noreg-ltdetr-coco |
| LightlyTrain | dinov3/convnext-tiny-ltdetr | 54.4 | 13.29 | 61.1 | 640×640 | dinov3/convnext-tiny-ltdetr-coco |
| LightlyTrain | dinov3/convnext-small-ltdetr | 56.9 | 17.65 | 82.7 | 640×640 | dinov3/convnext-small-ltdetr-coco |
| LightlyTrain | dinov3/convnext-base-ltdetr | 58.6 | 24.68 | 121.0 | 640×640 | dinov3/convnext-base-ltdetr-coco |
| LightlyTrain | dinov3/convnext-large-ltdetr | 60.0 | 42.30 | 230.0 | 640×640 | dinov3/convnext-large-ltdetr-coco |

## Object Detection with LT-DETR

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb)

Training an object detection model with LightlyTrain is straightforward and only
requires a few lines of code. See [data](#object-detection-data) for details on how
to prepare your dataset.

### Train an Object Detection Model

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_object_detection(
        out="out/my_experiment",
        model="dinov3/convnext-small-ltdetr-coco",
        data={
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

During training, both the

- best (with highest validation mAP<sub>50:95</sub>) and
- last (last validation round as determined by `save_checkpoint_args.save_every_num_steps`)

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

(object-detection-use-model-weights)=

### Load the Trained Model from Checkpoint and Predict

After the training completes, you can load the best model checkpoints for inference like this:

```python
import lightly_train

model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
results = model.predict("path/to/image.jpg")
```

Or use one of the pre-trained model weights directly from LightlyTrain:

```python
import lightly_train

model = lightly_train.load_model("dinov3/convnext-tiny-ltdetr-coco")
results = model.predict("path/to/image.jpg")
```

### Visualize the Result

After making the predictions with the model weights, you can visualize the predicted bounding boxes like this:

```python
import matplotlib.pyplot as plt
from torchvision import io, utils

import lightly_train

model = lightly_train.load_model("dinov3/convnext-tiny-ltdetr-coco")
labels, boxes, scores = model.predict("image.jpg").values()

# Visualize predictions.
image_with_boxes = utils.draw_bounding_boxes(
    image=io.read_image("image.jpg"),
    boxes=boxes,
    labels=[model.classes[i.item()] for i in labels],
)

fig, ax = plt.subplots(figsize=(30, 30))
ax.imshow(image_with_boxes.permute(1, 2, 0))
fig.savefig("predictions.png")
```

The predicted boxes are in the absolute (x_min, y_min, x_max, y_max) format, i.e. represent the size of the dimension of the bounding boxes in pixels.

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

(object-detection-data)=

## Data

Lightly**Train** supports training object detection models with images and bounding boxes.
Every image must have a corresponding annotation file (in [YOLO format](https://labelformat.com/formats/object-detection/yolov5/)) that contains for every object in the image a line with the class ID and 4 normalized bounding box coordinates (x_center, y_center, width, height). The file should have the `.txt` extension and an example annotation file for an image with two objects could look like this:

```text
0 0.716797 0.395833 0.216406 0.147222
1 0.687500 0.379167 0.255208 0.175000
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
- dcm (DICOM)

For more details on LightlyTrain's support for data input, please check the [Data Input](#data-input) page.

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
