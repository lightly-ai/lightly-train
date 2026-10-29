(models-lightlyvit)=

# LightlyViT

This page describes how to use LightlyViT models with LightlyTrain.

LightlyViT is Lightly's efficient Vision Transformer family designed for deployment on
resource-constrained hardware. It interleaves cheap depthwise-convolution local mixing
with global attention in a small subset of blocks, which makes it fast at high input
resolutions while keeping strong feature quality for detection and segmentation.

| Variant  | Width | Depth | Heads | Global blocks    | Default recipe                | Approx. backbone parameters |
| -------- | ----: | ----: | ----: | ---------------- | ----------------------------- | --------------------------: |
| ViT-T/16 |   224 |    12 |     4 | `[3, 7, 11]`     | linear mixer + RoPE positions |                        5.7M |
| ViT-S/16 |   432 |    12 |     9 | `[3, 7, 11]`     | linear mixer + RoPE positions |                       20.7M |
| ViT-M/16 |   640 |    16 |    10 | `[3, 7, 11, 15]` | linear mixer + RoPE positions |                       60.0M |

All variants use patch size 16, four register tokens, LayerNorm, GELU, MLP ratio 4,
LayerScale, and a `7 x 7` depthwise local mixer. Each variant's default recipe is the
one its pretrained checkpoints are trained with, so exported weights load into the model
without extra arguments.

## Architecture Options

Options beyond the default recipe can be set with `model_args`:

- `global_mixer_kind`: `"linear"` (default) or `"softmax"` attention in the global
  blocks. Linear attention is faster at high resolutions.
- `global_block_indices`: which blocks get the global mixer.
- `pos_encoding`: `"rope"` (default) or `"learned_absolute"`. Rotary position embeddings
  remove the resolution-bound learned position table, which is useful when pretraining
  and fine-tuning at different resolutions or aspect ratios.

## Pretrain and Fine-tune a LightlyViT Model

### Pretrain

LightlyViT models are pretrained from scratch with the [DINOv2 method](#methods-dinov2):

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",               # Output directory.
        data="my_data_dir",                    # Directory with images.
        model="lightlyvit/lightlyvit_s16",     # Pass the LightlyViT model.
        method="dinov2",                       # Use the DINOv2 pretraining method.
    )
```
````

````{tab} Command Line
```bash
lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="lightlyvit/lightlyvit_s16" method="dinov2"
```
````

### Fine-tune

After pretraining, export the backbone and load it into a downstream task model via the
`backbone_weights` argument. Refer to the following pages for fine-tuning instructions
and example code:

- [Object Detection](#object-detection) — fine-tune a LightlyViT-based LTDETR model;
  supports loading custom pretrained backbone weights via `backbone_weights`.
- [Semantic Segmentation](#semantic-segmentation) — fine-tune a LightlyViT-based linear
  segmentation model; supports `backbone_weights`.

If the model was pretrained with non-default `model_args` (for example a different
`global_mixer_kind` or `pos_encoding`), pass the same values as `model_args` when
fine-tuning so the exported weights load strictly.

## Supported Models

All models start from random initialization and are useful when pretraining from scratch
on a custom dataset:

- `lightlyvit/lightlyvit_t16`
- `lightlyvit/lightlyvit_s16`
- `lightlyvit/lightlyvit_m16`

```{note}
LightlyTrain is licensed under AGPL-3.0, which also covers model weights trained with
it. See [Licensing](#licensing) for details.
```
