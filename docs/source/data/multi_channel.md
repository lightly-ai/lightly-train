(multi-channel)=

# Single- and Multi-Channel Images

In addition to standard RGB images, LightlyTrain supports single- and multi-channel input for pretraining, and fine-tuning.

```{note}
Multi-channel input is not supported for direct distillation because the DINOv2/v3 teacher models expect 3-channel input. However, you could load n-channel images and then reduce them to 3-channels with the [`ChannelDrop`](#method-transform-args-channel-drop) augmentation.
```

Specify the number of image channels and normalization parameteres in the respective LightlyTrain training function. For example, to fine-tune a semantic segmentation model on 4-channel images:

```python
import lightly_train

lightly_train.train_semantic_segmentation(
    out="out/my_experiment",
    model="dinov2/vitl14-eomt",
    data={
        ... # multi-channel image data (e.g. RGB-NIR)
    },
    transform_args={
        "num_channels": 4, # specify number of channels here
        "normalize": {
            "mean": [0, 0, 0, 0],
            "std": [1, 1, 1, 1],
        },
    },
)
```

## Models

The following models support multi-channel image input:

| Library | Supported Models | Docs |
|---------|------------------|------|
| LightlyTrain | DINOv3 | |
| LightlyTrain | DINOv2 | |
| TIMM | All models | [🔗](#models-timm) |

## Transforms

The following image transforms are disabled for images that do not have 3 channels:

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
