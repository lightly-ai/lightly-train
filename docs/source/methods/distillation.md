(methods-distillation)=

# Distillation (recommended 🚀)

Knowledge distillation involves transferring knowledge from a large, compute-intensive teacher model to a smaller, efficient student model by encouraging similarity between the student and teacher representations. It addresses the challenge of bridging the gap between state-of-the-art large-scale vision models and smaller, more computationally efficient models suitable for practical applications.

```{note}
Starting from **LightlyTrain 0.7.0**, `method="distillation"` uses a new, improved `v2` implementation
that achieves higher accuracy and trains up to 3x faster. The previous version is still available via
`method="distillationv1"` for backward compatibility.
```

## Use Distillation in LightlyTrain

Follow the code below to distill the knowledge of the default DINOv2 ViT-B/14 teacher model into your model architecture. The example uses a `torchvision/resnet18` model as the student:

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
```
````

````{tab} Command Line
```bash
lightly-train train out=out/my_experiment data=my_data_dir model="torchvision/resnet18" method="distillation"
```
````

(methods-distillation-dinov3)=

### 🔥 Distill from DINOv3 🔥

To distill from DINOv3, you have to [sign up and accept the terms of use](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) from Meta to get access to the DINOv3 checkpoints. After signing up, you will receive an email with the download links. You can then use these links in your training script as shown below.

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment", 
        data="my_data_dir",
        model="torchvision/resnet18",
        method="distillation",
        method_args={
            "teacher": "dinov3/vitb16",
            # Replace with your own url
            "teacher_url": "https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth<SOME-KEY>",
        }
    )
```
````

(methods-distillation-dinov2-pretrain)=

### Pretrain and Distill Your Own DINOv2 Weights

LightlyTrain also supports [DINOv2 pretraining](#methods-dinov2), which can help you adjust the DINOv2 weights to your own domain data. Starting from **LightlyTrain 0.9.0**, after pretraining a ViT with DINOv2, you can distill your own pretrained model to your target model architecture with the distillation method. This is done by setting an optional `teacher_weights` argument in `method_args`.

The following example shows how to pretrain a ViT-B/14 model with DINOv2 and then distill the pretrained model to a ResNet-18 student model. Check out the [DINOv2 pretraining documentation](#methods-dinov2) for more details on how to pretrain a DINOv2 model.

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    # Pretrain a DINOv2 ViT-B/14 model.
    lightly_train.train(
        out="out/my_dinov2_pretrain_experiment",
        data="my_dinov2_pretrain_data_dir",
        model="dinov2/vitb14",
        method="dinov2",
    )

    # Distill the pretrained DINOv2 model to a ResNet-18 student model.
    lightly_train.train(
        out="out/my_distillation_pretrain_experiment",
        data="my_distillation_pretrain_data_dir",
        model="torchvision/resnet18",
        method="distillation",
        method_args={
            "teacher": "dinov2/vitb14",
            "teacher_weights": "out/my_dinov2_pretrain_experiment/exported_models/exported_last.pt", # pretrained `dinov2/vitb14` weights 
        }
    )
```
````

(methods-distillation-supported-models)=

### Supported Teacher Models

The following models for `teacher` are supported:

- DINOv3
  - `dinov3/vits16`
  - `dinov3/vits16plus`
  - `dinov3/vitb16`
  - `dinov3/vitl16`
  - `dinov3/vitl16plus`
  - `dinov3/vith16plus`
  - `dinov3/vit7b16`
- DINOv2
  - `dinov2/vits14`
  - `dinov2/vitb14`
  - `dinov2/vitl14`
  - `dinov2/vitg14`

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
