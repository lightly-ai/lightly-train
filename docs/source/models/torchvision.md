(models-torchvision)=

# Torchvision

This page describes how to use Torchvision models with LightlyTrain.

## Pretrain and Fine-tune a Torchvision Model

### Pretrain

Pretraining Torchvision models with LightlyTrain is straightforward. Below we provide
the minimum scripts for pretraining using `torchvision/resnet18` as an example:

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

Or alternatively, pass directly a Torchvision model instance:

```python
from torchvision.models import resnet18

import lightly_train

if __name__ == "__main__":
    model = resnet18()                        # Load the Torchvision model.
    lightly_train.train(
        out="out/my_experiment",              # Output directory.
        data="my_data_dir",                   # Directory with images.
        model=model,                          # Pass the Torchvision model.
    )
````

````{tab} Command Line
```bash
lightly-train train out="out/my_experiment" data="my_data_dir" model="torchvision/resnet18"
````

### Fine-tune

After pretraining, you can load the exported model for fine-tuning with Torchvision:

```python
import torch
from torchvision.models import resnet18

model = resnet18()
state_dict = torch.load("out/my_experiment/exported_models/exported_last.pt", weights_only=True)
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
- ShuffleNetV2
  - `torchvision/shufflenet_v2_x0_5`
  - `torchvision/shufflenet_v2_x1_0`
  - `torchvision/shufflenet_v2_x1_5`
  - `torchvision/shufflenet_v2_x2_0`
