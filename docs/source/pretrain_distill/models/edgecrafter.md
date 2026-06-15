(models-edgecrafter)=

# EdgeCrafter

This page describes how to use [EdgeCrafter](https://arxiv.org/abs/2603.18739) ECViT
models with LightlyTrain.

EdgeCrafter ECViT models are compact Vision Transformers designed for efficient dense
prediction. LightlyTrain supports ECViT backbones for LTDETR object detection through
the `edgecrafter` model package.

```{note}
EdgeCrafter is released under the
[Apache 2.0 license](https://github.com/lightly-ai/lightly-train/blob/main/licences/EDGECRAFTER_LICENSE).
```

## Fine-tune an EdgeCrafter Model

EdgeCrafter ECViT backbones can be fine-tuned for object detection with LTDETR. The
backbone weights are downloaded automatically when using one of the supported model
names:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_object_detection(
        out="out/my_experiment",
        data={
            "format": "yolo",
            "path": "my_data_dir",
            "train": "images/train2017",
            "val": "images/val2017",
            "names": {
                0: "person",
                1: "bicycle",
                # ...
            },
        },
        model="edgecrafter/ecvitt-ltdetr",
    )
```

See [Object Detection](#object-detection) for more details on dataset formats, training,
inference, and export.

```{note}
EdgeCrafter ECViT backbones currently support RGB images only. Multi-channel input is
not supported.
```

## Supported Models

### Backbone Models

The following ECViT backbone models are supported:

- `edgecrafter/ecvitt`
- `edgecrafter/ecvittplus`
- `edgecrafter/ecvits`
- `edgecrafter/ecvitsplus`

### Object Detection Models

The following LTDETR object detection models are supported:

- `edgecrafter/ecvitt-ltdetr`
- `edgecrafter/ecvittplus-ltdetr`
- `edgecrafter/ecvits-ltdetr`
- `edgecrafter/ecvitsplus-ltdetr`
