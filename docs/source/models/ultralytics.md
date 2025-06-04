(models-ultralytics)=

# Ultralytics

This page describes how to use Ultralytics models with LightlyTrain.

```{important}
[Ultralytics](https://github.com/ultralytics/ultralytics) must be installed with
`pip install "lightly-train[ultralytics]"`.
```

```{warning}
Using Ultralytics models might require a commercial Ultralytics license. See the
[Ultralytics website](https://www.ultralytics.com/license) for more information.
```

```{note}
For YOLOv12, we recommend using the [original YOLOv12 package](#models-yolov12) developed by the authors of YOLOv12, since the official Ultralytics implementations are not stable yet.
```

Models ending with `.pt` load pretrained weights by Ultralytics. Models ending with
`.yaml` are not pretrained.

## Pretrain and Fine-tune an Ultralytics Model

### Pretrain

Pretraining Ultralytics models with LightlyTrain is straightforward. Below we provide
the minimum scripts for pretraining using `ultralytics/yolov8s` as an example:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",                # Output directory.
        data="my_data_dir",                     # Directory with images.
        model="ultralytics/yolov8s.yaml",       # Pass the YOLO model.
    )

```

Or alternatively, pass directly a YOLO model instance:
```python
from ultralytics import YOLO

import lightly_train

if __name__ == "__main__":
    model = YOLO("yolov8s.yaml")                # Load the YOLO model.
    lightly_train.train(
        out="out/my_experiment",                # Output directory.
        data="my_data_dir",                     # Directory with images.
        model=model,                            # Pass the YOLO model.
    )
```
````

````{tab} Command Line
```bash
lightly-train train out="out/my_experiment" data="my_data_dir" model="ultralytics/yolov8s.yaml"
````

### Fine-tune

After pretraining, you can load the exported model for fine-tuning with Ultralytics:

````{tab} Python
```python
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("out/my_experiment/exported_models/exported_last.pt")
    model.train(data="coco8.yaml")
```
````

````{tab} Command Line
```bash
yolo detect train model=out/my_experiment/exported_models/exported_last.pt data="coco8.yaml"
````

## Supported Models

- YOLOv5
  - `ultralytics/yolov5l.yaml`
  - `ultralytics/yolov5l6u.pt`
  - `ultralytics/yolov5lu.pt`
  - `ultralytics/yolov5lu.yaml`
  - `ultralytics/yolov5m.yaml`
  - `ultralytics/yolov5m6u.pt`
  - `ultralytics/yolov5mu.pt`
  - `ultralytics/yolov5mu.yaml`
  - `ultralytics/yolov5n.yaml`
  - `ultralytics/yolov5n6u.pt`
  - `ultralytics/yolov5nu.pt`
  - `ultralytics/yolov5nu.yaml`
  - `ultralytics/yolov5s.yaml`
  - `ultralytics/yolov5s6u.pt`
  - `ultralytics/yolov5su.pt`
  - `ultralytics/yolov5su.yaml`
  - `ultralytics/yolov5x.yaml`
  - `ultralytics/yolov5x6u.pt`
  - `ultralytics/yolov5xu.pt`
  - `ultralytics/yolov5xu.yaml`
- YOLOv6
  - `ultralytics/yolov6l.yaml`
  - `ultralytics/yolov6m.yaml`
  - `ultralytics/yolov6n.yaml`
  - `ultralytics/yolov6s.yaml`
  - `ultralytics/yolov6x.yaml`
- YOLOv8
  - `ultralytics/yolov8l-cls.pt`
  - `ultralytics/yolov8l-cls.yaml`
  - `ultralytics/yolov8l-obb.pt`
  - `ultralytics/yolov8l-obb.yaml`
  - `ultralytics/yolov8l-oiv7.pt`
  - `ultralytics/yolov8l-pose.pt`
  - `ultralytics/yolov8l-pose.yaml`
  - `ultralytics/yolov8l-seg.pt`
  - `ultralytics/yolov8l-seg.yaml`
  - `ultralytics/yolov8l-world.pt`
  - `ultralytics/yolov8l-world.yaml`
  - `ultralytics/yolov8l-worldv2.pt`
  - `ultralytics/yolov8l-worldv2.yaml`
  - `ultralytics/yolov8l.pt`
  - `ultralytics/yolov8l.yaml`
  - `ultralytics/yolov8m-cls.pt`
  - `ultralytics/yolov8m-cls.yaml`
  - `ultralytics/yolov8m-obb.pt`
  - `ultralytics/yolov8m-obb.yaml`
  - `ultralytics/yolov8m-oiv7.pt`
  - `ultralytics/yolov8m-pose.pt`
  - `ultralytics/yolov8m-pose.yaml`
  - `ultralytics/yolov8m-seg.pt`
  - `ultralytics/yolov8m-seg.yaml`
  - `ultralytics/yolov8m-world.pt`
  - `ultralytics/yolov8m-world.yaml`
  - `ultralytics/yolov8m-worldv2.pt`
  - `ultralytics/yolov8m-worldv2.yaml`
  - `ultralytics/yolov8m.pt`
  - `ultralytics/yolov8m.yaml`
  - `ultralytics/yolov8n-cls.pt`
  - `ultralytics/yolov8n-cls.yaml`
  - `ultralytics/yolov8n-obb.pt`
  - `ultralytics/yolov8n-obb.yaml`
  - `ultralytics/yolov8n-oiv7.pt`
  - `ultralytics/yolov8n-pose.pt`
  - `ultralytics/yolov8n-pose.yaml`
  - `ultralytics/yolov8n-seg.pt`
  - `ultralytics/yolov8n-seg.yaml`
  - `ultralytics/yolov8n.pt`
  - `ultralytics/yolov8n.yaml`
  - `ultralytics/yolov8s-cls.pt`
  - `ultralytics/yolov8s-cls.yaml`
  - `ultralytics/yolov8s-obb.pt`
  - `ultralytics/yolov8s-obb.yaml`
  - `ultralytics/yolov8s-oiv7.pt`
  - `ultralytics/yolov8s-pose.pt`
  - `ultralytics/yolov8s-pose.yaml`
  - `ultralytics/yolov8s-seg.pt`
  - `ultralytics/yolov8s-seg.yaml`
  - `ultralytics/yolov8s-world.pt`
  - `ultralytics/yolov8s-world.yaml`
  - `ultralytics/yolov8s-worldv2.pt`
  - `ultralytics/yolov8s-worldv2.yaml`
  - `ultralytics/yolov8s.pt`
  - `ultralytics/yolov8s.yaml`
  - `ultralytics/yolov8x-cls.pt`
  - `ultralytics/yolov8x-cls.yaml`
  - `ultralytics/yolov8x-obb.pt`
  - `ultralytics/yolov8x-obb.yaml`
  - `ultralytics/yolov8x-oiv7.pt`
  - `ultralytics/yolov8x-pose.pt`
  - `ultralytics/yolov8x-pose.yaml`
  - `ultralytics/yolov8x-seg.pt`
  - `ultralytics/yolov8x-seg.yaml`
  - `ultralytics/yolov8x-world.pt`
  - `ultralytics/yolov8x-world.yaml`
  - `ultralytics/yolov8x-worldv2.pt`
  - `ultralytics/yolov8x-worldv2.yaml`
  - `ultralytics/yolov8x.pt`
  - `ultralytics/yolov8x.yaml`
- YOLO11
  - `ultralytics/yolo11n-cls.yaml`
  - `ultralytics/yolo11n-cls.pt`
  - `ultralytics/yolo11n-obb.yaml`
  - `ultralytics/yolo11n-obb.pt`
  - `ultralytics/yolo11n-pose.yaml`
  - `ultralytics/yolo11n-pose.pt`
  - `ultralytics/yolo11n-seg.yaml`
  - `ultralytics/yolo11n-seg.pt`
  - `ultralytics/yolo11n.yaml`
  - `ultralytics/yolo11n.pt`
  - `ultralytics/yolo11s-cls.yaml`
  - `ultralytics/yolo11s-cls.pt`
  - `ultralytics/yolo11s-obb.yaml`
  - `ultralytics/yolo11s-obb.pt`
  - `ultralytics/yolo11s-pose.yaml`
  - `ultralytics/yolo11s-pose.pt`
  - `ultralytics/yolo11s-seg.yaml`
  - `ultralytics/yolo11s-seg.pt`
  - `ultralytics/yolo11s.yaml`
  - `ultralytics/yolo11s.pt`
  - `ultralytics/yolo11m-cls.yaml`
  - `ultralytics/yolo11m-cls.pt`
  - `ultralytics/yolo11m-obb.yaml`
  - `ultralytics/yolo11m-obb.pt`
  - `ultralytics/yolo11m-pose.yaml`
  - `ultralytics/yolo11m-pose.pt`
  - `ultralytics/yolo11m-seg.yaml`
  - `ultralytics/yolo11m-seg.pt`
  - `ultralytics/yolo11m.yaml`
  - `ultralytics/yolo11m.pt`
  - `ultralytics/yolo11l-cls.yaml`
  - `ultralytics/yolo11l-cls.pt`
  - `ultralytics/yolo11l-obb.yaml`
  - `ultralytics/yolo11l-obb.pt`
  - `ultralytics/yolo11l-pose.yaml`
  - `ultralytics/yolo11l-pose.pt`
  - `ultralytics/yolo11l-seg.yaml`
  - `ultralytics/yolo11l-seg.pt`
  - `ultralytics/yolo11l.yaml`
  - `ultralytics/yolo11l.pt`
  - `ultralytics/yolo11x-cls.yaml`
  - `ultralytics/yolo11x-cls.pt`
  - `ultralytics/yolo11x-obb.yaml`
  - `ultralytics/yolo11x-obb.pt`
  - `ultralytics/yolo11x-pose.yaml`
  - `ultralytics/yolo11x-pose.pt`
  - `ultralytics/yolo11x-seg.yaml`
  - `ultralytics/yolo11x-seg.pt`
  - `ultralytics/yolo11x.yaml`
  - `ultralytics/yolo11x.pt`
- YOLO12
  - `ultralytics/yolo12n-cls.yaml`
  - `ultralytics/yolo12n-cls.pt`
  - `ultralytics/yolo12n-obb.yaml`
  - `ultralytics/yolo12n-obb.pt`
  - `ultralytics/yolo12n-pose.yaml`
  - `ultralytics/yolo12n-pose.pt`
  - `ultralytics/yolo12n-seg.yaml`
  - `ultralytics/yolo12n-seg.pt`
  - `ultralytics/yolo12n.yaml`
  - `ultralytics/yolo12n.pt`
  - `ultralytics/yolo12s-cls.yaml`
  - `ultralytics/yolo12s-cls.pt`
  - `ultralytics/yolo12s-obb.yaml`
  - `ultralytics/yolo12s-obb.pt`
  - `ultralytics/yolo12s-pose.yaml`
  - `ultralytics/yolo12s-pose.pt`
  - `ultralytics/yolo12s-seg.yaml`
  - `ultralytics/yolo12s-seg.pt`
  - `ultralytics/yolo12s.yaml`
  - `ultralytics/yolo12s.pt`
  - `ultralytics/yolo12m-cls.yaml`
  - `ultralytics/yolo12m-cls.pt`
  - `ultralytics/yolo12m-obb.yaml`
  - `ultralytics/yolo12m-obb.pt`
  - `ultralytics/yolo12m-pose.yaml`
  - `ultralytics/yolo12m-pose.pt`
  - `ultralytics/yolo12m-seg.yaml`
  - `ultralytics/yolo12m-seg.pt`
  - `ultralytics/yolo12m.yaml`
  - `ultralytics/yolo12m.pt`
  - `ultralytics/yolo12l-cls.yaml`
  - `ultralytics/yolo12l-cls.pt`
  - `ultralytics/yolo12l-obb.yaml`
  - `ultralytics/yolo12l-obb.pt`
  - `ultralytics/yolo12l-pose.yaml`
  - `ultralytics/yolo12l-pose.pt`
  - `ultralytics/yolo12l-seg.yaml`
  - `ultralytics/yolo12l-seg.pt`
  - `ultralytics/yolo12l.yaml`
  - `ultralytics/yolo12l.pt`
  - `ultralytics/yolo12x-cls.yaml`
  - `ultralytics/yolo12x-cls.pt`
  - `ultralytics/yolo12x-obb.yaml`
  - `ultralytics/yolo12x-obb.pt`
  - `ultralytics/yolo12x-pose.yaml`
  - `ultralytics/yolo12x-pose.pt`
  - `ultralytics/yolo12x-seg.yaml`
  - `ultralytics/yolo12x-seg.pt`
  - `ultralytics/yolo12x.yaml`
  - `ultralytics/yolo12x.pt`
