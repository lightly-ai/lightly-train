(methods-dinov2)=

# DINOv2 (beta 🔬)

DINOv2 is a state-of-the-art self-supervised learning method for training vision
foundation models. It is optimized for large-scale models and datasets.
DINOv2 pretrained models are effective across a wide range of tasks, including
image classification, object detection, and segmentation. They are also known to
generate high-quality features that can be used without fine-tuning the model.

## Use DINOv2 in LightlyTrain

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment", 
        data="my_data_dir",
        model="dinov2_vit/vitb14_pretrain",
        method="dinov2",
        method_args={
            # Only set these arguments when starting from a pretrained model
            "student_freeze_backbone_epochs": 1,  # Freeze the student backbone for 1 epoch
            "student_freeze_last_layer_epochs": 0,  # Unfreeze the student last layer
        },
    )
````

````{tab} Command Line
```bash
lightly-train train out=out/my_experiment data=my_data_dir model="dinov2_vit/vitb14_pretrain" method="dinov2"
```
````

The following models are available for DINOv2 pretraining:

- `dinov2_vit/vits14`
- `dinov2_vit/vits14_pretrain`
- `dinov2_vit/vitb14`
- `dinov2_vit/vitb14_pretrain`
- `dinov2_vit/vitl14`
- `dinov2_vit/vitl14_pretrain`
- `dinov2_vit/vitg14`
- `dinov2_vit/vitg14_pretrain`

Models with a `_pretrain` suffix are [pretrained by Meta](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models).

````{note}
When starting from a pretrained model we highly recommend to set the
`student_freeze_backbone_epochs` and `student_freeze_last_layer_epochs` arguments:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",
        data="my_data_dir",
        model="dinov2_vit/vitb14_pretrain",
        method="dinov2",
        method_args={
            "student_freeze_backbone_epochs": 1,  # Freeze the student backbone for 1 epoch
            "student_freeze_last_layer_epochs": 0,  # Unfreeze the student last layer
        },
    )
```

The reason for this is that the pretrained models only contain weights for the backbone
but not the head. Freezing the backbone for the first epoch allows the model to
initialize the head weights based on the pretrained backbone.

If you start from scratch, then do **not** set these arguments.
````

## What's under the Hood

DINOv2 combines the strengths of DINO and iBOT, two previous self-supervised learning
methods. Following DINO, it trains a student network to match the output of a
momentum-averaged teacher network without labels. It also incorporates the masked
image modeling loss from iBOT, which helps the model learn strong local semantic
features.

## Lightly Recommendations

- **Models**: DINOv2 can only be used with ViTs. If you want to use a different model,
  we recommend first pretraining a ViT with DINOv2 and then distilling the knowledge
  of the ViT into your model of choice with the [distillation method](methods-distillation).
- **Batch Size**: We recommend somewhere around 3072 for DINOv2 as the original paper
  suggested.
- **Number of Epochs**: We recommend somewhere between 100 to 300 epochs. However,
  DINOv2 benefits from longer schedules and may still improve after training for more
  than 300 epochs.
- **Large Datasets**: DINOv2 is optimized for large datasets. We recommend at least
  1 million images for training from scratch.

## Default Method Arguments

The following are the default method arguments for DINOv2. To learn how you can
override these settings, see {ref}`method-args`.

````{dropdown} Default Method Arguments
```{include} _auto/dinov2_method_args.md
```
````

## Default Image Transform Arguments

The following are the default transform arguments for DINOv2. To learn how you can
override these settings, see {ref}`method-transform-args`.

````{dropdown} Default Image Transforms
```{include} _auto/dinov2_transform_args.md
```
````
