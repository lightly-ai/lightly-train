(methods)=

# Methods

Lightly**Train** supports the following pretraining methods:

- [Distillation (recommended 🚀)](#methods-distillation)
- [DINO](#methods-dino)
- [SimCLR](#methods-simclr)

For detailed information on each method, please refer to the respective sections below.

(methods-comparison)=

## Which Method to Choose?

We strongly recommend Lightly’s custom distillation method (the default in LightlyTrain) for pretraining your models.

### Why use Distillation?

Distillation achieves the best performance on various tasks compared to DINO and SimCLR. It has the following advantages:

#### Pros

- **Domain Adaptability**: Distillation works across different data domains such as **Video Analytics, Robotics, Advanced Driver-Assistance Systems, and Agriculture**.
- **Memory Efficiency**: It is faster and requires less GPU memory compared to SSL methods like SimCLR, which usually demand large batch sizes.
- **Compute Efficiency**: It trains a smaller, inference-friendly student model that maintains the performance level of a much larger teacher model, making deployment efficient.
- **No Hyperparameter Tuning**: It has strong default parameters that require no or minimal tuning, simplifying the training process.

#### Cons

- **Performance Limitation**: However, the performance of knowledge distillation can be limited by the capabilities of the teacher model. In this case, you may want to consider using DINO or SimCLR.

### When to use DINO?

#### Pros

- **Domain Adaptability**: Like distillation, DINO works quite well across different data domains.
- **No Fine-tuning**: DINO performs excellently in the frozen regime, so it could be used out-of-the-box after pretraining if no fine-tuning is planned.

#### Cons

- **Compute Intensive**: DINO requires a lot more compute than distillation, partly due to the number of crops required in its multi-crop strategy. However, it is still less compute-intensive than SimCLR.
- **Instable Training**: DINO uses a “momentum teacher” whose weights update more slowly than the student’s. If some of the parameters (e.g. the teacher temperature) is not set properly, the teacher’s embeddings can shift in a way that the student cannot catch up. This destabilizes training and can lead to a oscillating and even rising loss.

### When to use SimCLR?

#### Pros

- **Fine-grained Features**: SimCLR’s contrastive learning approach is particularly effective for distinguishing subtle differences between samples, especially when you have abundant data and can accommodate large batch sizes. Thus SimCLR is well-suited for tasks like **visual quality inspection** which requires fine-grained differentiation.

#### Cons

- **Memory Intensive**: SimCLR requires larger batch sizes to work well.
- **Hyperparameter Sensitivity**: Also, SimCLR is sensitive to the augmentation recipe, so you may need to experiment and come up with your own augmentation strategy for your specific domain.

(methods-distillation)=

## Distillation (recommended 🚀)

Knowledge distillation involves transferring knowledge from a large, compute-intensive teacher model to a smaller, efficient student model by encouraging similarity between the student and teacher representations. It addresses the challenge of bridging the gap between state-of-the-art large-scale vision models and smaller, more computationally efficient models suitable for practical applications.

### Use Distillation in LightlyTrain

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

### What's under the Hood

Our distillation method draws inspiration from the [Knowledge Distillation: A Good Teacher is Patient and Consistent](https://arxiv.org/abs/2106.05237) paper. We made some modification so that labels are not required by obtaining the weights of a pseudo classifier using the different image-level representations from the batch. More specifically, we use a ViT-B/14 from [DINOv2](https://arxiv.org/pdf/2304.07193) as the teacher backbone, which we use to compute a queue of representations to serve the role of a pseudo classifier. The teacher batch representations are projected on the queue to obtain soft pseudo labels which can then be used to supervise the student representations when projected on the queue. The KL-divergence is used to enforce similarity between the teacher pseudo-labels and the student predictions.

### Lightly Recommendations

- **Models**: Knowledge distillation is agnostic to the choice of student backbone networks.
- **Batch Size**: We recommend somewhere between 128 and 1536 for knowledge distillation.
- **Number of Epochs**: We recommend somewhere between 100 and 3000. However, distillation benefits from longer schedules and models still improve after training for more than 3000 epochs. For small datasets (\<100k images) it can also be beneficial to train up to 10000 epochs.

(methods-dino)=

## DINO

[DINO (Distillation with No Labels)](https://arxiv.org/abs/2104.14294) is a self-supervised learning framework for visual representation learning using knowledge distillation but without the need for labels. Similar to knowledge distillation, DINO uses a teacher-student setup where the student learns to mimic the teacher's outputs. The major difference is that DINO uses an exponential moving average of the student as teacher. DINO achieves strong performance on image clustering, segmentation, and zero-shot transfer tasks.

### Use DINO in LightlyTrain

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

### What's under the Hood

DINO trains a student network to match the output of a momentum-averaged teacher network without labels. It employs a self-distillation objective with a cross-entropy loss between the student and teacher outputs. DINO uses random cropping, resizing, color jittering, and Gaussian blur to create diverse views of the same image. In particular, DINO employs a multi-crop augmentation strategy to generate two global views and multiple local views that are smaller crops of the original image. Additionally, centering and sharpening of the teacher pseudo labels is used to stabilize the training.

### Lightly Recommendations

- **Models**: DINO works well with both ViT and CNN.
- **Batch Size**: We recommend somewhere between 256 and 1024 for DINO as the original paper suggested.
- **Number of Epochs**: We recommend somewhere between 100 to 300 epochs. However, DINO benefits from longer schedules and may still improve after training for more than 300 epochs.

(methods-simclr)=

## SimCLR

[SimCLR](https://arxiv.org/abs/2002.05709) is a self-supervised learning method that employs contrastive learning. More specifically, it enforces similarity between the representations of two augmented views of the same image and dissimilarity w.r.t. to the other instances in the batch. Using strong data augmentations and large batch sizes, it achieves classification performance on ImageNet-1k that is comparable to that of supervised learning approaches.

### Use SimCLR in LightlyTrain

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

### What's under the Hood

SimCLR learns representations by creating two augmented views of the same image—using techniques like random cropping, resizing, color jittering, and Gaussian blur—and then training the model to maximize agreement between these augmented views while distinguishing them from other images. It employs the normalized temperature-scaled cross-entropy loss (NT-Xent) to encourage similar pairs to align and dissimilar pairs to diverge. The method benefits from large batch sizes, enabling it to achieve performance comparable to supervised learning on benchmarks like ImageNet-1k.

### Lightly Recommendations

- **Models**: SimCLR is specifically optimized for convolutional neural networks, with a focus on ResNet architectures. Using transformer-based models is doable but less common.
- **Batch Size**: We recommend a minimum of 256, though somewhere between 1024 and 4096 is ideal since SimCLR usually benefits from large batch sizes.
- **Number of Epochs**: We recommend a minimum of 800 epochs based on the top-5 linear evaludation results using ResNet-50 on ImageNet-1k reported by the original paper. The top-1 results continues to increase even after 3200 epochs. Also, using a large number of epochs compensates for using a relatively smaller batch size.
