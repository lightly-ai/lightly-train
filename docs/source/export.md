(export)=

# Export

The `export` command is used to prepare a model for fine-tuning or inference. It allows
exporting the model from [training checkpoints](#train-output) which contain additional
information such as optimizer states that are not needed for fine-tuning or inference.

```{tip}
After training the model is automatically exported in the default format of the used 
library to `out/my_experiment/exported_models/exported_last.pt`.
```

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",
        data="my_data_dir",
        model="torchvision/resnet50",
    )

    lightly_train.export(
        out="my_exported_model.pt",
        checkpoint="out/my_experiment/checkpoints/last.ckpt",
        part="model",
        format="torch_state_dict",
    )
````

````{tab} Command Line
```bash
lightly-train train out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50"
lightly-train export out="my_exported_model.pt" checkpoint="out/my_experiment/checkpoints/last.ckpt" part="model" format="torch_state_dict"
````

The above code example trains a model and exports the last training checkpoint as a
torch state dictionary.

```{tip}
See {meth}`lightly_train.export` for a complete list of arguments.
```

## Out

The `out` argument specifies the output file where the exported model is saved.

## Checkpoint

The `checkpoint` argument specifies the Lightly**Train** checkpoint to use for exporting the
model. This is the checkpoint saved to `out/my_experiment/checkpoints/<some>.ckpt` after
training.

(export-format)=

## Format

The optional `format` argument specifies the format in which the model is exported. The following
formats are supported.

- `package_default` (default)

  This format option can be used to automatically determine the required export format.
  It will fall back to `torch_state_dict` format if the model cannot be associated with
  a [supported library](models/index.md#supported-libraries).

  The model is saved in the native format of the used package:

  | Model | Format |
  | --- | --- |
  | `custom` | `state dict` |
  | `super_gradients` | `super_gradients` |
  | `timm` | `timm` |
  | `torchvision` | `state dict` |
  | `ultralytics` | `ultralytics` |

  Usage examples:

  ````{dropdown} torchvision
  ```python
  import torch
  from torchvision.models import resnet50

  model = resnet50()
  model.load_state_dict(torch.load("my_exported_model.pt", weights_only=True))
  ```
  ````

  ````{dropdown} ultralytics
  ```python
  from ultralytics import YOLO

  model = YOLO("my_exported_model.pt")
  ```
  ````

  ````{dropdown} super_gradients
  ```python
  from super_gradients.training import models

  model = models.get(
    model_name="yolo_nas_s",
    num_classes=3,
    checkpoint_path="my_exported_model.pt",
  )
  ```
  ````

  ````{dropdown} timm
  ```python
  import timm

  model = timm.create_model(
    "resnet18", pretrained=False, checkpoint_path="my_exported_model.pt",
  )
  ```
  ````

- `torch_state_dict`

  Only the model's state dict is saved which can be loaded with:

  ```python
  import torch
  from torchvision.models import resnet50

  model = resnet50()
  model.load_state_dict(torch.load("my_exported_model.pt", weights_only=True))
  ```

- `torch_model`

  The model is saved as a torch module which can be loaded with:

  ```python
  import torch

  model = torch.load("my_exported_model.pt")
  ```

  This requires that the same Lightly**Train** version is installed when the model is
  exported and when it is loaded again.

(export-part)=

## Part

The optional `part` argument specifies which part of the model to export. The following parts are
supported.

- `model` (default)

  Exports the model as passed with the `model` argument in the `train` function.

- `embedding_model`

  Exports the embedding model. This includes the model passed with the `model` argument
  in the `train` function and an extra embedding layer if the `embed_dim` argument was
  set during training. This is useful if you want to use the model for embedding images.
