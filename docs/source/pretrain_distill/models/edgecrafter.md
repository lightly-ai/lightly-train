(models-edgecrafter)=

# EdgeCrafter

This page describes how to use [EdgeCrafter](https://arxiv.org/abs/2603.18739) ECViT
models with LightlyTrain for pretraining and distillation.

EdgeCrafter ECViT models are compact Vision Transformers designed for efficient dense
prediction. LightlyTrain supports ECViT backbones through the `edgecrafter` model
package.

```{note}
EdgeCrafter is released under the
[Apache 2.0 license](https://github.com/lightly-ai/lightly-train/blob/main/licences/EDGECRAFTER_LICENSE).
```

## Pretrain an EdgeCrafter Model

EdgeCrafter ECViT backbones can be pretrained with LightlyTrain. For example, you can
distill a compact ECViT backbone like this:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        model="edgecrafter/ecvitt",
        method="distillation",
    )
```

See [Distillation method](#methods-distillation) for more details on pretraining and its
configuration options.

```{note}
EdgeCrafter ECViT backbones currently support RGB images only. Multi-channel input is
not supported.
```

## Supported Models

The following ECViT backbone models are supported:

- `edgecrafter/ecvitt`
- `edgecrafter/ecvittplus`
- `edgecrafter/ecvits`
- `edgecrafter/ecvitsplus`

For examples on how to fine-tune ECViT backbones inside LT-DETR for object detection,
see [Object Detection](#object-detection). For instance segmentation, see
[Instance Segmentation](#instance-segmentation-ltdetrv2).
