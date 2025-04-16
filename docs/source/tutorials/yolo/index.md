(tutorials-yolo)=

# Object Detection with Ultralytics' YOLO

This tutorial demonstrates how to pretrain a YOLO model using LightlyTrain and then fine-tune it for object detection using the `ultralytics` framework. We will perform both steps on the [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/).

```{warning}
Using Ultralytics models might require a commercial Ultralytics license. See the
[Ultralytics website](https://www.ultralytics.com/license) for more information.
```

## Install Dependencies

Install the required packages:

- `lightly-train` for pretraining, with support for `ultralytics`' YOLO models
- [`supervision`](https://github.com/roboflow/supervision) to visualize some of the annotated pictures

```bash
pip install "lightly-train[ultralytics]" "supervision==0.25.1"
```

## Download the Dataset

We can download the dataset directly using Ultralytics' API with the `check_det_dataset` function:

```python
from ultralytics.data.utils import check_det_dataset

dataset = check_det_dataset("VOC.yaml")
```

Ultralytics always downloads your datasets to a fixed location, which you can fetch via their `settings` module:

```python
from ultralytics import settings

print(settings["datasets_dir"])
```

Inside that directory (<DATASET-DIR>), you will now have the following structure of images and labels:

```bash
tree -d <DATASET-DIR>/VOC -I VOCdevkit

>    datasets/VOC
>    ├── images
>    │   ├── test2007
>    │   ├── train2007
>    │   ├── train2012
>    │   ├── val2007
>    │   └── val2012
>    └── labels
>        ├── test2007
>        ├── train2007
>        ├── train2012
>        ├── val2007
>        └── val2012
```

```{note}
Labels are not required for self-supervised pretraining. We will use the labels only for finetuning.
```

## Inspect a few Images

Let's use `supervision` and look at a few of the annotated samples to get a feeling of what the data looks like:

```python
import random

import matplotlib.pyplot as plt
import supervision as sv
import yaml
from ultralytics import settings
from ultralytics.data.utils import check_det_dataset

dataset = check_det_dataset("VOC.yaml")

detections = sv.DetectionDataset.from_yolo(
    data_yaml_path=dataset["yaml_file"],
    images_directory_path=f"{settings['datasets_dir']}/VOC/images/train2012",
    annotations_directory_path=f"{settings['datasets_dir']}/VOC/labels/train2012",
)

with open(dataset["yaml_file"], "r") as f:
    data = yaml.safe_load(f)

names = data["names"]

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.flatten()

detections = [detections[random.randint(0, len(detections))] for _ in range(4)]

for i, (path, image, annotation) in enumerate(detections):
    annotated_image = box_annotator.annotate(scene=image, detections=annotation)
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=annotation,
        labels=[names[elem] for elem in annotation.class_id],
    )
    ax[i].imshow(annotated_image[..., ::-1])
    ax[i].axis("off")

fig.tight_layout()
fig.show()
```

![VOC2012 Training Samples](samples_VOC_train2012.png)

## Pretrain and Fine-tune

We will use LightlyTrain to pretrain a YOLO11 model.

The following scripts or CLI commands will:

- Initialize a YOLO11s model with random weights.
- Pretrain the YOLO11s model on the training images of PASCAL VOC using distillation pretraining.
- Export the pretrained YOLO11s model.
- Fine-tune the pretrained model on PASCAL VOC dataset using labels.

````{tab} Python
```python
# pretrain_yolo.py
import lightly_train
from ultralytics import settings

data_path = f"{settings["datasets_dir"]}/VOC/images/train2012"

if __name__ == "__main__":
    # Pretrain with LightlyTrain.
    lightly_train.train(
        out="out/my_experiment",            # Output directory.
        model="ultralytics/yolo11s.yaml",   # Pass the YOLO model.
        data=data_path,                     # Path to a directory with training images.
        epochs=100,                         # Adjust epochs for faster training.
        batch_size=64,                      # Adjust batch size based on hardware.
    )
```

```python
# finetune_yolo.py

from ultralytics import YOLO

if __name__ == "__main__":
    # Load the exported model.
    model = YOLO("out/my_experiment/exported_models/exported_last.pt")

    # Fine-tune with ultralytics.
    model.train(data="VOC.yaml", epochs=100)
```
````

````{tab} Command Line
```bash
lightly-train train out="out/my_experiment" data="<DATASET-DIR>/VOC/images/train2012" model="ultralytics/yolo11s.yaml" epochs=100 batch_size=64
```

```bash
yolo detect train model="out/my_experiment/exported_models/exported_last.pt" data="VOC.yaml" epochs=100
```
````

Congratulations! You have successfully pretrained a model using LightlyTrain and fine-tuned it for object detection using Ultralytics.

For more advanced options, explore the [LightlyTrain Python API](#lightly-train) and [Ultralytics documentation](https://docs.ultralytics.com).

## Next Steps

- Go beyond the default distillation pretraining and experiment other pretraining learning methods in LightlyTrain. Check [Methods](#methods) for more information.
- Try various YOLO models (`YOLOv5`, `YOLOv6`, `YOLOv8`).
- Use the pretrained model for other tasks, like {ref}`image embeddings <embed>`.
