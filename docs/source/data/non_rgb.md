(non-rgb)=

# Non-RGB Images

Besides normal RGB images, LightlyTrain also supports single- and multi-channel non-RGB image input for pretraining, distillation, and fine-tuning.

You can specify the number of channels of your images with `transform_args={"num_channels": <num channels>}` in the respecitive training function in LightlyTrain. For example, to use 4-channel images for fine-tuning a semantic segmentation model, you can do:

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
In the above example, you might also want to customize the image normalization with the `normalize` parameter in `transform_args` to fit your use case. If you don't explicitly set this parameter, LightlyTrain will simply repeat the ImageNet default `mean` and `std` values of the RGB channels for the extra channels.
```

## Models

Currently, multi-channel image input is supported by the following models:

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

If, for any reason, you want to disable other transforms whose defaults are not compatible with your input, you can do so by setting the respective transform arguments to `None`. For example, to disable `GaussianBlur`, you can do:

```python
transform_args={
    "num_channels": 4,
    "gaussian_blur": None
},
```

Please check [Configure Transform Arguments](#method-transform-args) for more details on how to customize transforms.
