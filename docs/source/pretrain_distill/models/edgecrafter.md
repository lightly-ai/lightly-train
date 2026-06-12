(models-edgecrafter)=

# EdgeCrafter (ECViT)

This page describes how to use EdgeCrafter ECViT models with LightlyTrain.

[EdgeCrafter](https://github.com/capsule2077/edgecrafter) is a family of compact Vision
Transformers designed for edge dense prediction via task-specialized distillation. The
four ECViT-NN backbones are exposed under the `edgecrafter/<model>-ltdetr` prefix and
are used as backbones for the [LTDETR object detection task](#object-detection).

```{note}
ECViT models are released under the
[EdgeCrafter / DINOv3 license](https://github.com/capsule2077/edgecrafter).
```

## Supported Models

The following ECViT models are supported. All backbones are initialized from the
[upstream ECViT pretrained weights](https://github.com/capsule2077/edgecrafter/releases)
shipped with the package and require no additional pretraining step.

- `edgecrafter/ecvitt` — ViT-T, projected channels 192
- `edgecrafter/ecvittplus` — ViT-T+, projected channels 256
- `edgecrafter/ecvits` — ViT-S, projected channels 256
- `edgecrafter/ecvitsplus` — ViT-S+, projected channels 256

For object detection, append the `-ltdetr` task suffix:

- `edgecrafter/ecvitt-ltdetr`
- `edgecrafter/ecvittplus-ltdetr`
- `edgecrafter/ecvits-ltdetr`
- `edgecrafter/ecvitsplus-ltdetr`

## Fine-tune an ECViT Backbone for Object Detection

ECViT backbones are used as drop-in replacements for DINOv3 ViT backbones in the LTDETR
object detection task model. See [Object Detection](#object-detection) for the training
and inference code; the `model="edgecrafter/ecvitt-ltdetr"` argument is the only change
required.

## Notes

- The ECViT-NN uses a `ConvPyramidPatchEmbed` with a fixed patch size of 16; the
  train/val transforms use this value directly. Multi-channel input is not supported
  (always 3 input channels).
- The backbone is exposed via a thin pass-through adapter (no `SpatialPriorModule` and
  no `mask_token` freeze, unlike the DINOv3 ViT path).
- The per-level channel counts that the ECViT wrapper emits match the DINOv3 ViT-shaped
  LTDETR encoder/decoder configs (`proj_dim` 192 → ViTT, 256 → ViTTPlus), so the
  existing `HybridEncoder`, `RTDETRTransformerv2`, and `DFINETransformer` are reused
  without modification.
