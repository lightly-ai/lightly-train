(non-rgb)=

# Non-RGB Images

In addition to standard RGB images, LightlyTrain supports single- and multi-channel non-RGB inputs for pretraining, distillation, and fine-tuning.

Specify the number of image channels via `transform_args={"num_channels": <num channels>}` in the respective LightlyTrain training function. For example, to fine-tune a semantic segmentation model on 4-channel images:

```python
import lightly_train

lightly_train.train_semantic_segmentation(
    out="out/my_experiment",
    model="dinov2/vitl14-eomt",
    data={
        ... # multi-channel image data (e.g. RGB-NIR)
    },
    logger_args={
        "mlflow": {
            "experiment_name": "my_experiment",
            "run_name": "my_run",
            "tracking_uri": "tracking_uri",
        },
    },
    transform_args={
        "num_channels": 4 # specify number of channels here
    },
)
```

```{note}
You may also want to customize normalization with the `normalize` parameter in `transform_args`. If you don't set it, LightlyTrain repeats the ImageNet default RGB `mean` and `std` values for any additional channels.
```

## Models

The following models support multi-channel image input:

| Library | Supported Models | Docs |
|---------|------------------|------|
| LightlyTrain | DINOv3 | |
| LightlyTrain | DINOv2 | |
| TIMM | All models | [🔗](#models-timm) |

## Transforms

The following image transforms are disabled for non-RGB images:

- `ColorJitter`
- `RandomGrayscale`
- `Solarize`

If any other transform defaults are incompatible with your data, you can disable them by setting the corresponding transform argument to `None`. For example, to disable `GaussianBlur`:

```python
transform_args={
    "num_channels": 4,
    "gaussian_blur": None
},
```

See [Configure Transform Arguments](#method-transform-args) for details on customizing transforms.
