(methods)=

# Methods

Lightly**Train** supports the following pretraining methods:

- {ref}`methods-distillation`
- {ref}`methods-dino`
- {ref}`methods-simclr`

```{seealso}
This page is meant to help you choosing the best method for your use case. If you are instead interested in the technical details of each method, please refer to the individual method’s pages linked above.
```

```{seealso}
Want to customize the augmentations for a specific method? Check out {ref}`method-transform-args`.
```

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

```{toctree}
---
hidden:
maxdepth: 1
---
distillation
dino
simclr
```
