(models-timm)=

# TIMM

This page describes how to use TIMM models with LightlyTrain.

```{important}
[TIMM](https://github.com/huggingface/pytorch-image-models) must be installed with
`pip install "lightly-train[timm]"`.
```

## Pretrain and Fine-tune a TIMM Model

### Pretrain

Pretraining TIMM models with LightlyTrain is straightforward. Below we provide the
minimum scripts for pretraining using `timm/resnet18` as an example:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",                # Output directory.
        data="my_data_dir",                     # Directory with images.
        model="timm/resnet18",                  # Pass the timm model.
    )

```

Or alternatively, pass directly a TIMM model instance:

```python
import timm

import lightly_train

if __name__ == "__main__":
    model = timm.create_model("resnet18")       # Load the model.
    lightly_train.train(
        out="out/my_experiment",                # Output directory.
        data="my_data_dir",                     # Directory with images.
        model=model,                            # Pass the TIMM model.
    )
````

````{tab} Command Line
```bash
lightly-train train out="out/my_experiment" data="my_data_dir" model="timm/resnet18"
````

### Fine-tune

After pretraining, you can load the exported model for fine-tuning with TIMM:

```python
import timm

model = timm.create_model(
  model_name="resnet18",
  checkpoint_path="out/my_experiment/exported_models/exported_last.pt",
)
```

## Supported Models

All timm models are supported, see [timm docs](https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file#models) for a full list.

Examples:

- `timm/resnet50`
- `timm/convnext_base`
- `timm/vit_base_patch16_224`
