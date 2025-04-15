(methods-distillation)=

# Distillation (recommended 🚀)

Knowledge distillation involves transferring knowledge from a large, compute-intensive teacher model to a smaller, efficient student model by encouraging similarity between the student and teacher representations. It addresses the challenge of bridging the gap between state-of-the-art large-scale vision models and smaller, more computationally efficient models suitable for practical applications.

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

````{tab} Command Line
```bash
lightly-train train out=out/my_experiment data=my_data_dir model="torchvision/resnet18" method="distillation"
````

## What's under the Hood

Our distillation method draws inspiration from the [Knowledge Distillation: A Good Teacher is Patient and Consistent](https://arxiv.org/abs/2106.05237) paper. We made some modification so that labels are not required by obtaining the weights of a pseudo classifier using the different image-level representations from the batch. More specifically, we use a ViT-B/14 from [DINOv2](https://arxiv.org/pdf/2304.07193) as the teacher backbone, which we use to compute a queue of representations to serve the role of a pseudo classifier. The teacher batch representations are projected on the queue to obtain soft pseudo labels which can then be used to supervise the student representations when projected on the queue. The KL-divergence is used to enforce similarity between the teacher pseudo-labels and the student predictions.

## Lightly Recommendations

- **Models**: Knowledge distillation is agnostic to the choice of student backbone networks.
- **Batch Size**: We recommend somewhere between 128 and 1536 for knowledge distillation.
- **Number of Epochs**: We recommend somewhere between 100 and 3000. However, distillation benefits from longer schedules and models still improve after training for more than 3000 epochs. For small datasets (\<100k images) it can also be beneficial to train up to 10000 epochs.

## Default Augmentation Settings

The following are the default augmentation settings for Distillation. To learn how you can override these settings, see {ref}`method-transform-args`.

```{include} _auto/distillation_transform_args.md
```
