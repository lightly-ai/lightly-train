(methods-dinov2)=

# DINOv2

DINOv2 is a state-of-the-art self-supervised learning method for training vision
foundation models. It is optimized for large-scale models and datasets.
DINOv2 pretrained models are effective across a wide range of tasks, including
image classification, object detection, and segmentation. They are also known to
generate high-quality features that can be used without fine-tuning the model.

```{table}

| Implementation | Model | ImageNet k-NN |
|--------------|----------|---------------|
| LightlyTrain | ViT-L/16 | 81.9% |
| [Official](https://github.com/facebookresearch/dinov2) | ViT-L/16 | 81.6% |

```

*The LightlyTrain DINOv2 implementation matches or outperforms the official
implementation. All models are trained from scratch on ImageNet-1K.*

## Use DINOv2 in LightlyTrain

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment", 
        data="my_data_dir",
        model="dinov2/vitb14",
        method="dinov2",
    )
````

````{tab} Command Line
```bash
lightly-train train out=out/my_experiment data=my_data_dir model="dinov2/vitb14" method="dinov2"
```
````

The following models are available for DINOv2 pretraining:

- `dinov2/vits14`
- `dinov2/vitb14`
- `dinov2/vitl14`
- `dinov2/vitg14`

All models are [pretrained by Meta](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models).

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
