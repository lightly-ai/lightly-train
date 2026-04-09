(models-dinov3)=

# DINOv3

This page describes how to use DINOv3 models with LightlyTrain.

[DINOv3](https://github.com/facebookresearch/dinov3) models are Vision Transformers
(ViTs) and ConvNeXt models pretrained by Meta using the DINOv3 self-supervised learning
method on the large-scale LVD-1689M dataset. They are state-of-the-art vision foundation
models and serve as strong backbones for downstream tasks such as object detection,
segmentation, and image classification.

```{note}
DINOv3 models are released under the
[DINOv3 license](https://github.com/lightly-ai/lightly-train/blob/main/licences/DINOv3_LICENSE.md).
Use [DINOv2](#models-dinov2) models instead for a more permissive Apache 2.0 license.
```

## Pretrain and Fine-tune a DINOv3 Model

### Pretrain

DINOv3 ViT-T/16 models (`dinov3/vitt16` and `dinov3/vitt16plus`) are efficient tiny
models trained by Lightly using the [distillation method](#methods-distillation) with
DINOv3 ViT-L/16 as the teacher. They are not part of Meta's official DINOv3 release but
follow the same architecture. The ViT-T architecture is based on the design proposed in
[Touvron et al., 2022](https://arxiv.org/abs/2207.10666).

You can distill your own DINOv3 ViT-T/16 model from DINOv3 ViT-L/16 on your custom
dataset as follows:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",                # Output directory.
        data="my_data_dir",                     # Directory with images.
        model="dinov3/vitt16",                  # Student: DINOv3 ViT-T/16.
        method="distillation",
        method_args={
            "teacher": "dinov3/vitl16",         # Teacher: DINOv3 ViT-L/16.
        },
    )
```
````

````{tab} Command Line
```bash
lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="dinov3/vitt16" method="distillation" method_args.teacher="dinov3/vitl16"
````

See [Distillation method](#methods-distillation) for more details on the pretraining
method and its configuration options.

### Fine-tune

DINOv3 models come with high-quality pretrained weights from Meta and can be used
directly as fine-tuning backbones without additional pretraining. After pretraining on a
custom dataset, the exported backbone can also be loaded via the `backbone_weights`
argument. Refer to the following pages for fine-tuning instructions and example code:

- [Object Detection](#object-detection) — fine-tune a DINOv3-based LTDETR model;
  supports loading custom pretrained backbone weights via `backbone_weights` (see
  [Pretrain and Fine-tune](#object-detection-pretrain-finetune)).
- [Semantic Segmentation](#semantic-segmentation) — fine-tune a DINOv3-based EoMT model;
  supports loading custom pretrained backbone weights via `backbone_weights` (see
  [Pretrain and Fine-tune](#semantic-segmentation-pretrain-finetune)).
- [Instance Segmentation](#instance-segmentation) — fine-tune a DINOv3-based EoMT model.
- [Panoptic Segmentation](#panoptic-segmentation) — fine-tune a DINOv3-based EoMT model.
- [Image Classification](#image-classification) — fine-tune a DINOv3 backbone for
  classification.

## Supported Models

### ViT Models

The following ViT models are supported. The LVD-1689M and SAT-493M models are
[pretrained by Meta](https://github.com/facebookresearch/dinov3/tree/main?tab=readme-ov-file#pretrained-models).
The ViT-T/16 models are trained by Lightly using knowledge distillation from DINOv3
ViT-L/16.

- ViT-T (Lightly, LVD-1689M distillation)
  - `dinov3/vitt16`
  - `dinov3/vitt16plus`
  - `dinov3/vitt16-distillationv1`
  - `dinov3/vitt16plus-distillationv1`
  - `dinov3/vitt16-notpretrained`
  - `dinov3/vitt16plus-notpretrained`
- ViT-S (Meta, LVD-1689M)
  - `dinov3/vits16`
  - `dinov3/vits16plus`
- ViT-B (Meta, LVD-1689M)
  - `dinov3/vitb16`
- ViT-L (Meta)
  - `dinov3/vitl16` (LVD-1689M)
  - `dinov3/vitl16-sat493m` (SAT-493M)
- ViT-H (Meta, LVD-1689M)
  - `dinov3/vith16plus`
- ViT-7B (Meta)
  - `dinov3/vit7b16` (LVD-1689M)
  - `dinov3/vit7b16-sat493m` (SAT-493M)

### ConvNeXt Models

The following ConvNeXt models are supported. All are
[pretrained by Meta](https://github.com/facebookresearch/dinov3/tree/main?tab=readme-ov-file#pretrained-models)
on the LVD-1689M dataset.

- `dinov3/convnext-tiny`
- `dinov3/convnext-small`
- `dinov3/convnext-base`
- `dinov3/convnext-large`
