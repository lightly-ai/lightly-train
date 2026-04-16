(methods-distillation)=

# Distillation

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/distillation.ipynb)

Knowledge distillation involves transferring knowledge from a large, compute-intensive
teacher model to a smaller, efficient student model by encouraging similarity between
the student and teacher representations. It addresses the challenge of bridging the gap
between state-of-the-art large-scale vision models and smaller, more computationally
efficient models suitable for practical applications.

```{note}
Three distillation versions are available. Choose based on your downstream task:

- **`distillation`** (alias for `distillationv3`, default from **LightlyTrain 0.15.0**):
    Best compromise — strong on both global tasks (e.g., classification) and dense 
    tasks (e.g., detection, segmentation). Recommended for most use cases.
- **`distillationv1`**: Best for purely global tasks (e.g., image classification).
- **`distillationv2`**: Best for purely dense tasks (e.g., object detection, 
    segmentation) and we advise to only use it with DINOv2 teacher models (use 
    `distillationv3` for DINOv3 teachers instead).
```

## Use Distillation in LightlyTrain

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/distillation.ipynb)

Follow the code below to distill the knowledge of the default DINOv3 ViT-B/16 teacher
model into your model architecture. The example uses a `torchvision/resnet18` model as
the student:

```{note}
DINOv3 models are released under the [DINOv3 license](https://github.com/lightly-ai/lightly-train/blob/main/licences/DINOv3_LICENSE.md). 
Use DINOv2 models instead for a more permissive Apache 2.0 license.
```

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        model="torchvision/resnet18",
        method="distillation",
    )
```
````

````{tab} Command Line
```bash
lightly-train pretrain out=out/my_experiment data=my_data_dir model="torchvision/resnet18" method="distillation"
```
````

(methods-distillation-dinov3)=

(methods-distillation-custom-models)=

### Custom Teacher and Student Models

Besides the built-in support for many popular models, all distillation versions also
support using custom student models, by implementing the interface specified on the
[Custom Models](#custom-models) page.

With distillationv3, LightlyTrain now also supports custom teacher models, by
implementing the same interface for the teacher.

**Option A: String-based** — use any supported model string with optional constructor
args:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        model="timm/resnet18",           # student as string
        model_args={"num_classes": 120}, # optional student constructor args
        method="distillationv3",
        method_args={
            "teacher": "timm/vit_base_patch16_224",  # any supported model string
            "teacher_args": {"pretrained": False},    # optional teacher constructor args
            "teacher_weights": "path/to/teacher_weights.pt", # optional teacher weights path
        }
    )
```

**Option B: Pre-instantiated wrapper** — implement the interface on the
[Custom Models](#custom-models) page and pass an instance of your wrapper directly to
the `model` (for student) and `teacher` (for teacher) arguments:

```python
import lightly_train
from my_module import MyStudentWrapper, MyTeacherWrapper

if __name__ == "__main__":
    student = MyStudentWrapper(my_student_model)
    teacher = MyTeacherWrapper(my_teacher_model)

    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        model=student,
        method="distillationv3",
        method_args={"teacher": teacher},
    )
```

When passing pre-instantiated wrappers, `model_args` and `teacher_args` are ignored
since the models are already constructed.

(methods-distillation-dinov2-pretrain)=

### Pretrain and Distill Your Own DINOv2 Weights

LightlyTrain also supports [DINOv2 pretraining](#methods-dinov2), which can help you
adjust the DINOv2 weights to your own domain data. Starting from **LightlyTrain 0.9.0**,
after pretraining a ViT with DINOv2, you can distill your own pretrained model to your
target model architecture with the distillation method. This is done by setting an
optional `teacher_weights` argument in `method_args`.

The following example shows how to pretrain a ViT-B/14 model with DINOv2 and then
distill the pretrained model to a ResNet-18 student model. Check out the
[DINOv2 pretraining documentation](#methods-dinov2) for more details on how to pretrain
a DINOv2 model.

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    # Pretrain a DINOv2 ViT-B/14 model.
    lightly_train.pretrain(
        out="out/my_dinov2_pretrain_experiment",
        data="my_dinov2_pretrain_data_dir",
        model="dinov2/vitb14",
        method="dinov2",
    )

    # Distill the pretrained DINOv2 model to a ResNet-18 student model.
    lightly_train.pretrain(
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

For distillation v1/v2, the following models for `teacher` are supported:

- DINOv3
  - `dinov3/vits16`
  - `dinov3/vits16plus`
  - `dinov3/vitb16`
  - `dinov3/vitl16`
  - `dinov3/vitl16-sat493m`
  - `dinov3/vitl16plus`
  - `dinov3/vith16plus`
  - `dinov3/vit7b16`
  - `dinov3/vit7b16-sat493m`
  - `dinov3/convnext-tiny`
  - `dinov3/convnext-small`
  - `dinov3/convnext-base`
  - `dinov3/convnext-large`
- DINOv2
  - `dinov2/vits14`
  - `dinov2/vitb14`
  - `dinov2/vitl14`
  - `dinov2/vitg14`

For distillationv3, any model supported by LightlyTrain can be used (including custom
models). You can find the full list of supported models on the [Models](#models) page.

## What's under the Hood

All versions apply a loss between the features of the student and teacher networks when
processing the same image, using strong identical augmentations on both inputs for
consistency. The different versions draw heavy inspiration from:

- [*Knowledge Distillation: A Good Teacher is Patient and Consistent*](https://arxiv.org/abs/2106.05237).
- The [*AM-RADIO*](https://arxiv.org/pdf/2304.07193) series of papers.

The versions differ in how the loss is computed:

- **v1** uses a queue of teacher embeddings to compute pseudo labels (global loss). Best
  for global tasks such as classification or distilling your own embedding model.
- **v2** directly applies MSE loss on the spatial features (dense loss). Best for dense
  tasks such as detection and segmentation.
- **v3** combines both the queue-based pseudo label loss and loss on the spatial
  features, making it a strong general-purpose choice.

## Lightly Recommendations

- **Models**: Knowledge distillation is agnostic to the choice of student backbone
  networks.
- **Batch Size**: We recommend somewhere between 128 and 2048 for knowledge
  distillation.
- **Number of Epochs**: We recommend somewhere between 100 and 3000. However,
  distillation benefits from longer schedules and models still improve after pretraining
  for more than 3000 epochs. For small datasets (\<100k images) it can also be
  beneficial to pretrain up to 10000 epochs.

## Default Method Arguments

The following are the default method arguments for distillation. To learn how you can
override these settings, see {ref}`method-args`.

````{dropdown} distillation (v3)
```{include} _auto/distillation_method_args.md
```
````

````{dropdown} distillationv1
```{include} _auto/distillationv1_method_args.md
```
````

````{dropdown} distillationv2
```{include} _auto/distillationv2_method_args.md
```
````

## Default Image Transform Arguments

The following are the default transform arguments for distillation. To learn how you can
override these settings, see {ref}`method-transform-args`.

````{dropdown} Default Image Transforms
```{include} _auto/distillation_transform_args.md
```
````
