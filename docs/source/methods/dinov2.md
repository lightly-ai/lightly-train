(methods-dinov2)=

# DINOv2

[DINOv2](https://arxiv.org/abs/2304.07193) is the SOTA self-supervised learning framework for visual representation learning that builds on a student-teacher architecture with an exponential moving average (EMA) of the student as the teacher. DINOv2 learns strong, general-purpose features from unlabeled data and achieves impressive out-of-the-box performance on classification, segmentation, and depth estimation tasks. DINOv2 is well suited for image-level and for pixel-level tasks. The models of the DINOv2 family are considered as vision foundation models.

## Use DINO in LightlyTrain

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment", 
        data="my_data_dir",
        model="dinov2_vit/vitb14",
        method="dinov2",
    )
````

````{tab} Command Line
```bash
lightly-train train out=out/my_experiment data=my_data_dir model="dinov2_vit/vitb14" method="dinov2"
````

## What's under the Hood

Like DINO, DINOv2 trains a student network to match the output of a momentum-averaged teacher network without labels. Building on DINO, DINOv2 combines the discriminative learning objectives of DINO and iBOT, incorporates SwAV-style feature centering, and adds a regularizer to encourage feature diversity. DINOv2 mostly uses DINO augmentations with only little adjustments.

## Lightly Recommendations

- **Models**: DINOv2 is to be used with ViTs.
- **Batch Size**: We recommend somewhere around 3072 for DINOv2 as the original paper suggested.
- **Number of Epochs**: We recommend somewhere between 100 to 300 epochs. However, DINOv2 benefits from longer schedules and may still improve after training for more than 300 epochs.

## Default Augmentation Settings

The following are the default augmentation settings for DINOv2. To learn how you can override these settings, see {ref}`method-transform-args`.

```{include} _auto/dinov2_transform_args.md
```
