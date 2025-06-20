(methods-simclr)=

# SimCLR

[SimCLR](https://arxiv.org/abs/2002.05709) is a self-supervised learning method that employs contrastive learning. More specifically, it enforces similarity between the representations of two augmented views of the same image and dissimilarity w.r.t. to the other instances in the batch. Using strong data augmentations and large batch sizes, it achieves classification performance on ImageNet-1k that is comparable to that of supervised learning approaches.

## Use SimCLR in LightlyTrain

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment", 
        data="my_data_dir",
        model="torchvision/resnet18",
        method="simclr",
    )
````

````{tab} Command Line
```bash
lightly-train train out=out/my_experiment data=my_data_dir model="torchvision/resnet18" method="simclr"
````

## What's under the Hood

SimCLR learns representations by creating two augmented views of the same image—using techniques like random cropping, resizing, color jittering, and Gaussian blur—and then training the model to maximize agreement between these augmented views while distinguishing them from other images. It employs the normalized temperature-scaled cross-entropy loss (NT-Xent) to encourage similar pairs to align and dissimilar pairs to diverge. The method benefits from large batch sizes, enabling it to achieve performance comparable to supervised learning on benchmarks like ImageNet-1k.

## Lightly Recommendations

- **Models**: SimCLR is specifically optimized for convolutional neural networks, with a focus on ResNet architectures. Using transformer-based models is doable but less common.
- **Batch Size**: We recommend a minimum of 256, though somewhere between 1024 and 4096 is ideal since SimCLR usually benefits from large batch sizes.
- **Number of Epochs**: We recommend a minimum of 800 epochs based on the top-5 linear evaluation results using ResNet-50 on ImageNet-1k reported by the original paper. The top-1 results continues to increase even after 3200 epochs. Also, using a large number of epochs compensates for using a relatively smaller batch size.

## Default Method Arguments

The following are the default method arguments for SimCLR. To learn how you can
override these settings, see {ref}`method-args`.

````{dropdown} Default Method Arguments
```{include} _auto/simclr_method_args.md
```
````

## Default Image Transform Arguments

The following are the default transform arguments for SimCLR. To learn how you can override these settings, see {ref}`method-transform-args`.

````{dropdown} Default Image Transforms
```{include} _auto/simclr_transform_args.md
```
````
