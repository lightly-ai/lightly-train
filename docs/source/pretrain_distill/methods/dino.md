(methods-dino)=

# DINO

[DINO (Distillation with No Labels)](https://arxiv.org/abs/2104.14294) is a self-supervised learning framework for visual representation learning using knowledge distillation but without the need for labels. Similar to knowledge distillation, DINO uses a teacher-student setup where the student learns to mimic the teacher's outputs. The major difference is that DINO uses an exponential moving average of the student as teacher. DINO achieves strong performance on image clustering, segmentation, and zero-shot transfer tasks.

## Use DINO in LightlyTrain

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment", 
        data="my_data_dir",
        model="torchvision/resnet18",
        method="dino",
    )
````

````{tab} Command Line
```bash
lightly-train train out=out/my_experiment data=my_data_dir model="torchvision/resnet18" method="dino"
````

## What's under the Hood

DINO trains a student network to match the output of a momentum-averaged teacher network without labels. It employs a self-distillation objective with a cross-entropy loss between the student and teacher outputs. DINO uses random cropping, resizing, color jittering, and Gaussian blur to create diverse views of the same image. In particular, DINO employs a multi-crop augmentation strategy to generate two global views and multiple local views that are smaller crops of the original image. Additionally, centering and sharpening of the teacher pseudo labels is used to stabilize the training.

## Lightly Recommendations

- **Models**: DINO works well with both ViT and CNN.
- **Batch Size**: We recommend somewhere between 256 and 1024 for DINO as the original paper suggested.
- **Number of Epochs**: We recommend somewhere between 100 to 300 epochs. However, DINO benefits from longer schedules and may still improve after training for more than 300 epochs.

## Default Method Arguments

The following are the default method arguments for DINO. To learn how you can
override these settings, see {ref}`method-args`.

````{dropdown} Default Method Arguments
```{include} _auto/dino_method_args.md
```
````

## Default Image Transform Arguments

The following are the default transform arguments for DINO. To learn how you can override these settings, see {ref}`method-transform-args`.

````{dropdown} Default Image Transforms
```{include} _auto/dino_transform_args.md
```
````
