(methods-distillation)=

# Distillation (recommended 🚀)

Knowledge distillation involves transferring knowledge from a large, compute-intensive teacher model to a smaller, efficient student model by encouraging similarity between the student and teacher representations. It addresses the challenge of bridging the gap between state-of-the-art large-scale vision models and smaller, more computationally efficient models suitable for practical applications.

```{note}
Starting from **LightlyTrain 0.7.0**, `method="distillation"` uses a new, improved `v2` implementation
that achieves higher accuracy and trains up to 3x faster. The previous version is still available via
`method="distillationv1"` for backward compatibility.
```

## Use Distillation in LightlyTrain

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment", 
        data="my_data_dir",
        model="torchvision/resnet18",
        method="distillation",
    )
````

(methods-distillation-dinov3)=

### 🔥 Experimental: Try Distillation from [DINOv3](https://ai.meta.com/dinov3/) 🔥

Downloading DINOv3 checkpoints currently requires to [sign up and accept the terms of use](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/). Shortly thereafter you will receive an email with the download links, which you can pass to the `method_args` in the training script:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment", 
        data="my_data_dir",
        model="torchvision/resnet18",
        method="distillation",
        method_args={
            "teacher": "dinov3/vits16",
            "teacher_url": "https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth<SOME-KEY>",
        }
    )
```

We currently tested it with `"dinov3/vits16"`, `"dinov3/vits16plus"` and `"dinov3/vitb16"`.

````{tab} Command Line
```bash
lightly-train train out=out/my_experiment data=my_data_dir model="torchvision/resnet18" method="distillation"
````

## What's under the Hood

Our distillation method directly applies a mean squared error (MSE) loss between the features of the student and teacher networks when processing the same image. We use a ViT-B/14 backbone from [DINOv2](https://arxiv.org/pdf/2304.07193) as the teacher model. Inspired by [*Knowledge Distillation: A Good Teacher is Patient and Consistent*](https://arxiv.org/abs/2106.05237), we apply strong, identical augmentations to both teacher and student inputs to ensure consistency of the objective.

## Lightly Recommendations

- **Models**: Knowledge distillation is agnostic to the choice of student backbone networks.
- **Batch Size**: We recommend somewhere between 128 and 2048 for knowledge distillation.
- **Number of Epochs**: We recommend somewhere between 100 and 3000. However, distillation benefits from longer schedules and models still improve after training for more than 3000 epochs. For small datasets (\<100k images) it can also be beneficial to train up to 10000 epochs.

## Default Method Arguments

The following are the default method arguments for distillation. To learn how you can
override these settings, see {ref}`method-args`.

````{dropdown} Default Method Arguments
```{include} _auto/distillation_method_args.md
```
````

## Default Image Transform Arguments

The following are the default transform arguments for distillation. To learn how you can
override these settings, see {ref}`method-transform-args`.

````{dropdown} Default Image Transforms
```{include} _auto/distillation_transform_args.md
```
````
