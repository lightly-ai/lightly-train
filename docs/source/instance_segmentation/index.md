(instance-segmentation)=

# Instance Segmentation

Lightly**Train** supports fine-tuning instance segmentation models with the
{py:func}`train_instance_segmentation <lightly_train.train_instance_segmentation>`
function. Two model families are currently supported:

(instance-segmentation-ltdetrv2-overview)=

## LTDETRv2 (NEW)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/ltdetr_instance_segmentation.ipynb)

LightlyTrain's **LTDETRv2** is a DETR-based instance segmentation family built on ECViT
backbones from the [EdgeCrafter](https://arxiv.org/abs/2603.18739) paper by Liu et al.
It matches the accuracy of the original EdgeCrafter ECSeg implementation on COCO, while
being 10-20% faster on GPU. See [LTDETRv2](ltdetrv2.md) for benchmark results, available
models, and how to train, predict, and export a LTDETRv2 instance segmentation model.

## EoMT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_instance_segmentation.ipynb)

LightlyTrain also supports training **DINOv2** and **DINOv3**-based instance
segmentation models with the [EoMT architecture](https://arxiv.org/abs/2503.19108) by
Kerssies et al. See [EoMT](eomt.md) for benchmark results, available models, and how to
train, predict, and export an EoMT instance segmentation model.

```{toctree}
---
hidden:
maxdepth: 1
---
ltdetrv2
eomt
```
