(methods)=

# Methods

Lightly**Train** supports the following pretraining methods:

- {ref}`methods-distillation`
- {ref}`methods-dinov2`
- {ref}`methods-dino`
- {ref}`methods-simclr`

```{seealso}
This page is meant to help you choose the best method for your use case. If you are
instead interested in the technical details of each method, please refer to the
individual method's pages linked above.
```

```{seealso}
Want to customize the augmentations for a specific method? Check out {ref}`method-transform-args`.
```

(methods-comparison)=

## Which Method to Choose?

We strongly recommend Lightly's custom distillation method (the default in LightlyTrain)
for pretraining your models.

### Why use Distillation?

Distillation achieves the best performance on various tasks compared to DINO and SimCLR.
It has the following advantages:

#### Pros

- **Domain Adaptability**: Distillation works across different data domains such as
  **Video Analytics, Robotics, Advanced Driver-Assistance Systems, and Agriculture**.
- **Memory Efficiency**: It is faster and requires less GPU memory compared to SSL methods like
  SimCLR, which usually demand large batch sizes.
- **Compute Efficiency**: It trains a smaller, inference-friendly student model that maintains
  the performance level of a much larger teacher model, making deployment efficient.
- **No Hyperparameter Tuning**: It has strong default parameters that require no or minimal
  tuning, simplifying the training process.

#### Cons

- **Performance Limitation**: However, the performance of knowledge distillation can be limited by
  the capabilities of the teacher model. In this case, you may want to consider using
  DINOv2 or one of the other methods.

### When to use DINOv2?

DINOv2 should be selected for the following use cases:

- **Vision Transformer (ViT) Models**: DINOv2 is specifically designed for ViT
  architectures. It does not support convolutional models like ResNet or YOLO.
- **Foundation Model Training**: DINOv2 is the state-of-the-art method for training
  vision foundation models.
- **Improve Distillation**: Models pretrained with DINOv2 on your dataset can be used
  as teacher models in the distillation method. This lets you transfer DINOv2
  knowledge to smaller models with architectures that are not limited to ViTs.

#### Pros

- **State-of-the-Art Performance**: DINOv2 is the state-of-the-art method for pretraining
  vision foundation models.
- **High Quality Features**: DINOv2 is known to produce high-quality features that can
  be used for various downstream tasks without the need for fine-tuning.
- **Combine with Distillation**: Models pretrained with DINOv2 can be used as teacher
  models in the distillation method. This lets you transfer DINOv2 knowledge to smaller
  models with architectures that are not limited to ViTs.
- **Pretrained Models Available**: Speed up DINOv2 pretraining on your dataset by
  starting from one of the pretrained models available in LightlyTrain.

#### Cons

- **Vision Transformer (ViT) Only**: DINOv2 is specifically designed for ViT
  architectures and does not support convolutional models like ResNet or YOLO.
- **Large Datasets**: DINOv2 requires large datasets for effective pretraining,
  which may not be feasible for all applications.
- **Compute Intensive**: DINOv2 is compute-intensive, especially when training large
  ViT models on large datasets. It requires substantial GPU resources and memory.

### When to use DINO?

#### Pros

- **Domain Adaptability**: Like distillation, DINO works quite well across different data domains.
- **No Fine-tuning**: DINO performs excellently in the frozen regime, so it could be used
  out-of-the-box after pretraining if no fine-tuning is planned.

#### Cons

- **Compute Intensive**: DINO requires a lot more compute than distillation, partly due to the
  number of crops required in its multi-crop strategy. However, it is still less compute-intensive
  than SimCLR.
- **Unstable Training**: DINO uses a “momentum teacher” whose weights update more slowly than the
  student's. If some of the parameters (e.g. the teacher temperature) are not set properly, the
  teacher's embeddings can shift in a way that the student cannot catch up. This destabilizes
  training and can lead to an oscillating and even rising loss.

### When to use SimCLR?

#### Pros

- **Fine-grained Features**: SimCLR's contrastive learning approach is particularly effective for
  distinguishing subtle differences between samples, especially when you have abundant data and
  can accommodate large batch sizes. Thus SimCLR is well-suited for tasks like **visual quality
  inspection** which requires fine-grained differentiation.

#### Cons

- **Memory Intensive**: SimCLR requires larger batch sizes to work well.
- **Hyperparameter Sensitivity**: Also, SimCLR is sensitive to the augmentation recipe, so you
  may need to experiment and come up with your own augmentation strategy for your specific
  domain.

```{toctree}
---
hidden:
maxdepth: 1
---
distillation
dinov2
dino
simclr
```
