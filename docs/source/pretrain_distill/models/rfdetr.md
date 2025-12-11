(models-rfdetr)=

# RF-DETR

This page describes how to use the [RF-DETR models](https://github.com/roboflow/rf-detr) with LightlyTrain.

```{important}
We have observed difficulties with the installation of RF-DETR in Python>=3.12, since it requires manual builds of some dependencies with `cmake`. We therefore strongly recommend using Python 3.9, 3.10 or 3.11.
```

You can install the required packages by running:

```bash
pip install "lightly-train[rfdetr]"
```

## Pretrain and Fine-tune an RF-DETR Model

### Pretrain

Pretraining RF-DETR models with LightlyTrain is straightforward. Below we provide
the minimum scripts for pretraining using `rfdetr/rf-detr-base` as an example:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",                # Output directory.
        data="my_data_dir",                     # Directory with images.
        model="rfdetr/rf-detr-base",            # Pass the RF-DETR model.
    )
```

Or alternatively, pass directly an RF-DETR model instance:

```python
from rfdetr import RFDETRBase

import lightly_train

if __name__ == "__main__":
    model = RFDETRBase()                        # Load the RF-DETR model.
    lightly_train.train(
        out="out/my_experiment",                # Output directory.
        data="my_data_dir",                     # Directory with images.
        model=model,                            # Pass the RF-DETR model.
    )
````

````{tab} Command Line
```bash
lightly-train train out="out/my_experiment" data="my_data_dir" model="rfdetr/rf-detr-base"
````

### Fine-tune

After pretraining, you can load the exported model for fine-tuning with RF-DETR.
For now, RF-DETR only supports datasets in COCO JSON format. Below we provide
the minimum scripts for fine-tuning using the [Coconuts dataset from Roboflow](https://universe.roboflow.com/traindataset/coconuts-plj8h/dataset/1/download/coco)
in COCO JSON format:

```python
# fine_tune.py

from rfdetr import RFDETRBase
from roboflow import Roboflow

if __name__ == "__main__":
    model = RFDETRBase(pretrain_weights="out/my_experiment/exported_models/exported_last.pt")

    rf = Roboflow(api_key="your_roboflow_api_key")
    project = rf.workspace("traindataset").project("coconuts-plj8h")
    version = project.version(1)
    dataset = version.download("coco")
      
    model.train(dataset_dir=dataset.location)
```

which can be run with RF-DETR's DDP training:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env fine_tune.py
```

## Supported Models

The following RF-DETR models are supported:

- `rfdetr/rf-detr-base`
- `rfdetr/rf-detr-base-2` (a less converged model that may be better for finetuning but worse for inference)
- `rfdetr/rf-detr-large`
