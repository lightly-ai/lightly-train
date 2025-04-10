(models-torchvision)=

# Torchvision

This page describes how to use Torchvision models with LightlyTrain.

## Pretrain a Torchvision Model

Pretraining Torchvision models with LightlyTrain is straightforward. Below we will provide the minimum scripts for pretraining using `torchvision/resnet18` as an example:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",                # Output directory.
        data="my_data_dir",                     # Directory with images.
        model="torchvision/resnet18",           # Pass the Torchvision model.
    )

```
````

````{tab} Command Line
```bash
lightly-train train out="out/my_experiment" data="my_data_dir" model="torchvision/resnet18"
````

You can reload the exported model by:

```python
import torch
from torchvision.models import resnet18

model = resnet18()
state_dict = torch.load("out/my_experiment/exported_models/exported_last.pt")
model.load_state_dict(state_dict)
```

## Supported Models

The following Torchvision models are supported:

- ResNet
  - `torchvision/resnet18`
  - `torchvision/resnet34`
  - `torchvision/resnet50`
  - `torchvision/resnet101`
  - `torchvision/resnet152`
- ConvNext
  - `torchvision/convnext_base`
  - `torchvision/convnext_large`
  - `torchvision/convnext_small`
  - `torchvision/convnext_tiny`
