(models-timm)=

# TIMM

This page describes how to use TIMM models with LightlyTrain.

```{important}
[TIMM](https://github.com/huggingface/pytorch-image-models) must be installed with
`pip install "lightly-train[timm]"`.
```

## Pretrain a TIMM Model

Pretraining TIMM models with LightlyTrain is straightforward. Below we will provide the minimum scripts for pretraining using `timm/resnet18` as an example:

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
````

````{tab} Command Line
```bash
lightly-train train out="out/my_experiment" data="my_data_dir" model="timm/resnet18"
````

You can reload the exported model by timm directly:

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
