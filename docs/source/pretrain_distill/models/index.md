(models)=

# Models

Lightly**Train** supports training models from various libraries. See [Supported Libraries](#supported-libraries) for a list of supported libraries and models.

The model is specified in the `train` command with the `model` argument:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",
        data="my_data_dir",
        model="torchvision/resnet50",
    )
````

````{tab} Command Line
```bash
lightly-train train out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50"
````

Model names always follow the pattern `<library name>/<model name>`.

Instead of passing a model name, it is also possible to pass a model instance directly to the `train` function:

````{tab} Python
```python
import lightly_train
from torchvision.models import resnet50

if __name__ == "__main__":
    model = resnet50()                  # Load the model.
    lightly_train.train(
        out="out/my_experiment",
        data="my_data_dir",
        model=model,                    # Pass the model.
    )
````

## List Models

The `list_models` command lists all available models. Only models from installed packages are listed.

````{tab} Python
```python
import lightly_train

print(lightly_train.list_models())
````

````{tab} Command Line
```bash
lightly-train list_models
````

(models-supported-libraries)=

## Supported Libraries

The following libraries are supported (follow the links to get to the respective docs pages):

- [Torchvision](#models-torchvision)
- [TIMM](#models-timm)
- [Ultralytics](#models-ultralytics)
- [RT-DETR](#models-rtdetr)
- [RF-DETR](#models-rfdetr)
- [YOLOv12](#models-yolov12)
- [SuperGradients](#models-supergradients)

```{toctree}
---
hidden:
maxdepth: 1
---
torchvision
timm
ultralytics
rtdetr
rfdetr
yolov12
supergradients
custom_models
```

% Alternative reference to avoid overwriting the reference to the custom models page.
(models-custom-models)=

## Custom Models

See {ref}`Custom Models <custom-models>` for information on how to train custom models.
