(non-rgb)=

# Non-RGB Image Support

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
        "num_channels": 4
    },
)
```

## Models

Add DINOv2, DINOv3, and TIMM model multi-channel support

## Transforms

The following transforms are disabled for non-RGB images:

- ColorJitter
- RandomGrayscale
- Solarize

### ChannelDrop

Add channel drop augmentation for fine-tuning
