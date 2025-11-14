# LightlyTrain - SOTA Pretraining, Fine-tuning and Distillation

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/quick_start.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%7C3.9%7C3.10%7C3.11%7C3.12-blue.svg)](https://docs.lightly.ai/train/stable/installation.html)
[![Docker](https://img.shields.io/badge/Docker-blue?logo=docker&logoColor=fff)](https://docs.lightly.ai/train/stable/docker.html#)
[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://docs.lightly.ai/train/stable/)
[![Discord](https://img.shields.io/discord/752876370337726585?logo=discord&logoColor=white&label=discord&color=7289da)](https://discord.gg/xvNJW94)

*LightlyTrain is the leading framework for transforming your unlabeled data into powerful vision foundation models – from pretraining, over knowledge distillation to fine-tuning.*

Unlock the full potential of your data with state-of-the-art (SOTA) computer vision methods like **DINOv2** and **DINOv3**. Train any model architecture (YOLO, transformers, and beyond) and fine-tune with **segmentation** and **object detection** for your specific use case.

## Benchmarks

### Custom Foundation Model: Train your own DINOv2!

With LightlyTrain you can train your very own foundation model like DINOv2 on your data.

| Implementation | Model | ImageNet k-NN | Docs |
|:--------------:|:-----:|:-------------:|:----:|
| LightlyTrain | dinov2/vitl16 | **81.9%** | [🔗](https://docs.lightly.ai/train/stable/semantic_segmentation.html#semantic-segmentation-eomt-dinov3) |
| DINOv2 | dinov2/vitl16 | 81.6% | [🔗](https://github.com/facebookresearch/dinov2) |

### Object Detection: Fine-Tune DINOv2 or DINOv3 for detection!

#### COCO Dataset

| Implementation | Backbone Model | AP<sub>50:95</sub> | Latency (ms) | # Params (M) | Input Size | Checkpoint Name |
|:--------------:|:----------------------------:|:------------------:|:------------:|:------------:|:----------:|:---------------------------------:|
| LightlyTrain | dinov2/vits14-ltdetr | 55.7 | 16.87 | 55.3 | 644×644 | dinov2/vits14-ltdetr-coco |
| LightlyTrain | dinov3/convnext-tiny-ltdetr | 54.4 | 13.29 | 61.1 | 640×640 | dinov3/convnext-tiny-ltdetr-coco |
| LightlyTrain | dinov3/convnext-small-ltdetr | 56.9 | 17.65 | 82.7 | 640×640 | dinov3/convnext-small-ltdetr-coco |
| LightlyTrain | dinov3/convnext-base-ltdetr | 58.6 | 24.68 | 121.0 | 640×640 | dinov3/convnext-base-ltdetr-coco |
| LightlyTrain | dinov3/convnext-large-ltdetr | 60.0 | 42.30 | 230.0 | 640×640 | dinov3/convnext-large-ltdetr-coco |

Latency is measured on a single NVIDIA T4 GPU with batch size 1. All models are compiled and optimized using `tensorrt==10.13.3.9`.

### Semantic Segmentation: Use SOTA method from CVPR 2025!

#### COCO-Stuff Dataset

| Implementation | Backbone Model | Val mIoU | Avg. FPS | # Params (M) | Input Size | Checkpoint Name |
|:--------------:|:------------------:|:---------:|:--------:|:------------:|:----------:|:-----------------------:|
| LightlyTrain | dinov3/vits16-eomt | 0.465 | 88.7 | 21.6 | 518×518 | dinov3/vits16-eomt-coco |
| LightlyTrain | dinov3/vitb16-eomt | 0.520 | 43.3 | 85.7 | 518×518 | dinov3/vitb16-eomt-coco |
| LightlyTrain | dinov3/vitl16-eomt | **0.544** | 20.4 | 303.2 | 518×518 | dinov3/vitl16-eomt-coco |

Avg. FPS is measured on a single NVIDIA T4 GPU with batch size 1. All models are compiled and optimized using `torch.compile`.

#### Cityscapes Dataset

| Implementation | Backbone Model | Val mIoU | Avg. FPS | # Params (M) | Input Size | Checkpoint Name |
|:------------------------------------:|:------------------:|:---------:|:--------:|:------------:|:----------:|:-----------------------------:|
| LightlyTrain | dinov3/vits16-eomt | 0.786 | 18.6 | 21.6 | 1024×1024 | dinov3/vits16-eomt-cityscapes |
| LightlyTrain | dinov3/vitb16-eomt | 0.810 | 8.7 | 85.7 | 1024×1024 | dinov3/vitb16-eomt-cityscapes |
| LightlyTrain | dinov3/vitl16-eomt | **0.844** | 3.9 | 303.2 | 1024×1024 | dinov3/vitl16-eomt-cityscapes |
| EoMT (CVPR 2025 paper, current SOTA) | dinov2/vitl16-eomt | 0.842 | - | 319 | 1024×1024 | - |

Avg. FPS is measured on a single NVIDIA T4 GPU with batch size 1. All models are compiled and optimized using `torch.compile`.

#### ADE20k Dataset

| Implementation | Model Name | Autolabel | Val mIoU | # Params (M) | Input Size | Checkpoint Name |
|:--------------:|:------------------:|:------:|:--------------------:|:------------:|:----------:| :----------------:|
| LightlyTrain | dinov3/vits16-eomt | ❌ | 0.466 | 21.6 | 518×518 | |
| LightlyTrain | dinov3/vits16-eomt | ✅ | **0.533** | 21.6 | 518×518 | dinov3/vits16-eomt-ade20k |
| LightlyTrain | dinov3/vitb16-eomt | ❌ | 0.544 | 85.7 | 518×518 | |
| LightlyTrain | dinov3/vitb16-eomt-ade20k | ✅ | **0.573** | 85.7 | 518×518 | dinov3/vitb16-eomt-ade20k |

The better results with auto-labeling were achieved by fine-tuning a ViT-H+ on the ADE20k dataset, which reaches 0.595 validation mIoU. We then used the checkpoint to create pseudo masks for the SUN397 dataset (~100k images). Using these masks, we subsequently fine-tuned the smaller models, and then used the ADE20k dataset for validation.

## News

- \[[0.12.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-12-0)\] - 2025-11-06: 💡 **New DINOv3 Object Detection:** Run inference or fine-tune DINOv3 models for [object detection](https://docs.lightly.ai/train/stable/object_detection.html)! 💡
- \[[0.11.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-11-0)\] - 2025-08-15: 🚀 **New DINOv3 Support:** Pretrain your own model with [distillation](https://docs.lightly.ai/train/stable/methods/distillation.html#methods-distillation-dinov3) from DINOv3 weights. Or fine-tune our SOTA [EoMT semantic segmentation model](https://docs.lightly.ai/train/stable/semantic_segmentation.html#semantic-segmentation-eomt-dinov3) with a DINOv3 backbone! 🚀
- \[[0.10.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-10-0)\] - 2025-08-04:
  🔥 **Train state-of-the-art semantic segmentation models** with our new
  [**DINOv2 semantic segmentation**](https://docs.lightly.ai/train/stable/semantic_segmentation.html)
  fine-tuning method! 🔥
- \[[0.9.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-9-0)\] - 2025-07-21:
  [**DINOv2 pretraining**](https://docs.lightly.ai/train/stable/methods/dinov2.html) is
  now out of beta and officially available!
- \[[0.8.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-8-0)\] - 2025-06-10:
  [**DINOv2 pretraining**](https://docs.lightly.ai/train/stable/methods/dinov2.html) is
  now available (beta 🔬)!
- \[[0.7.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-7-0)\] - 2025-05-26:
  Up to **3x faster distillation** and higher accuracy with [**Distillation v2**](https://docs.lightly.ai/train/stable/methods/distillation.html)
  (new default method)!

## Installation

Lightly**Train** requires Python 3.8+ and runs on Windows, Linux and MacOS.

```bash
pip install lightly-train
```

## 🔥 Pretrain Your Own DINOv2 Foundation Model 🔥

Pretrain a DINOv2 model on your own unlabeled images. LightlyTrain's DINOv2
implementation matches or outperforms the official implementation on ImageNet-1K.
See our [documentation](https://docs.lightly.ai/train/stable/methods/dinov2.html) on
how to get started!

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment", 
        data="my_data_dir",
        model="dinov2/vitb14",
        method="dinov2",
    )
```

See our [documentation](https://docs.lightly.ai/train/stable/methods/dinov2.html)
for more details.

## 🔥 Distill DINOv2/v3 Into Any Model Architecture 🔥

Pretrain any model architecture with unlabeled data by distilling the knowledge from
DINOv2 or DINOv3 foundation models into your model. On the COCO dataset, YOLOv8-s
models pretrained with LightlyTrain achieve high performance across all tested label
fractions. These improvements hold for other architectures like YOLOv11, RT-DETR,
and Faster R-CNN. See our [announcement post](https://www.lightly.ai/blog/introducing-lightly-train)
for more benchmarks and details. See our [documentation](https://docs.lightly.ai/train/stable/methods/distillation.html)
on how to get started!

![Benchmark Results](https://cdn.prod.website-files.com/62cd5ce03261cb3e98188470/67fe4efa0209fb4eb0c3da5c_Introducing%20LightlyTrain_imag_1.png)

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment", 
        data="my_data_dir",
        model="ultralytics/yolov8s.pt",
        method="distillation",
    )
```

See our [documentation](https://docs.lightly.ai/train/stable/methods/distillation.html)
for more details.

## 🔥 High-Performance Object Detection Models 🔥

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb)

LightlyTrain’s LT-DETR models, powered by DINOv2 and DINOv3 backbones, demonstrate strong performance across different scales.

🚀 We are actively working on new models with improved speed and accuracy. Updates coming soon, so stay tuned!

```python
import lightly_train
from torchvision import utils, io
import matplotlib.pyplot as plt

model = lightly_train.load_model("dinov3/convnext-tiny-ltdetr-coco")

labels, boxes, scores = model.predict("<image>.jpg").values()

# Visualize predictions.
image_with_boxes = utils.draw_bounding_boxes(
    image=io.read_image("<image>.jpg"),
    boxes=boxes,
    labels=[model.classes[i.item()] for i in labels],
)

fig, ax = plt.subplots(figsize=(30, 30))
ax.imshow(image_with_boxes.permute(1, 2, 0))
fig.savefig(f"predictions.png")
```

Or fine-tune on your own dataset:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_object_detection(
        out="out/my_experiment",
        model="dinov3/convnext-tiny-ltdetr-coco",
        data={
          # data config ...
        }
    )
```

See our [documentation](https://docs.lightly.ai/train/stable/object_detection.html) for more details.

## 🔥 Fine-tune SOTA Instance Segmentation Models 🔥

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_instance_segmentation.ipynb)

LightlyTrain's EoMT instance segmentation model based on DINOv3 achieves a new
state-of-the-art on the COCO benchmark! See our [documentation](https://docs.lightly.ai/train/stable/instance_segmentation.html)
for more details.

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_instance_segmentation(
        out="out/my_experiment",
        model="dinov3/vits16-eomt-inst-coco",
        data={
          # data config ...
        }
    )
```

## 🔥 Fine-tune SOTA Semantic Segmentation Models 🔥

LightlyTrain's EoMT semantic segmentation model based on DINOv3 achieves a new
state-of-the-art on the ADE20K benchmark! See our [documentation](https://docs.lightly.ai/train/stable/semantic_segmentation.html)
for more details.

You can explore training semantic segmentation models with the example code below:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out/my_experiment",
        model="dinov3/vits16-eomt",
        data={
          # data config ...
        }
    )
```

## Tutorials

- **Fine-tuning Your Pretrained Models**: Looking for code example for fine-tuning after pretraining your model? Head over to the [Quick Start](https://docs.lightly.ai/train/stable/quick_start.html#fine-tune)!

- **Embedding Example**: Want to use your pretrained model to generate image embeddings instead? Check out the [embed](https://docs.lightly.ai/train/stable/embed.html) guide!

- **Instance Segmentation Fine-tuning**: Want to train a state-of-the-art instance segmentation model? Head over to the [instance segmentation guide](https://docs.lightly.ai/train/stable/instance_segmentation.html)!

- **Semantic Segmentation Fine-tuning**: Want to train a state-of-the-art semantic segmentation model? Head over to the [semantic segmentation guide](https://docs.lightly.ai/train/stable/semantic_segmentation.html)!

- **More Tutorials**: Want to get more hands-on with LightlyTrain? Check out our [Tutorials](https://docs.lightly.ai/train/stable/tutorials/index.html) for more examples!

## Features

Model Pretraining (**no self-supervised learning expertise required!**)

- Pretrain DINOv2 foundation models on your own data
- Distill knowledge from DINOv2 or DINOv3 into any model architecture
- Pretrain models from popular libraries such as [Torchvision](https://docs.lightly.ai/train/stable/models/torchvision.html),
  [TIMM](https://docs.lightly.ai/train/stable/models/timm.html),
  [Ultralytics](https://docs.lightly.ai/train/stable/models/ultralytics.html),
  [SuperGradients](https://docs.lightly.ai/train/stable/models/supergradients.html),
  [RT-DETR](https://docs.lightly.ai/train/stable/models/rtdetr.html),
  [RF-DETR](https://docs.lightly.ai/train/stable/models/rfdetr.html),
  and [YOLOv12](https://docs.lightly.ai/train/stable/models/yolov12.html)
- Pretrain [custom models](https://docs.lightly.ai/train/stable/models/custom_models.html) with ease
- [Export models in their native format](https://docs.lightly.ai/train/stable/export.html) for fine-tuning or inference
- Generate and export [image embeddings](https://docs.lightly.ai/train/stable/embed.html)

Model Fine-tuning

- Fine-tune DINOv2 and DINOv3 for [object detection](https://docs.lightly.ai/train/stable/object_detection.html)
- Fine-tune DINOv3 for [instance segmentation](https://docs.lightly.ai/train/stable/instance_segmentation.html)
- Fine-tune DINOv2 and DINOv3 for [semantic segmentation](https://docs.lightly.ai/train/stable/semantic_segmentation.html)

MLOps

- Python, Command Line, and [Docker](https://docs.lightly.ai/train/stable/docker.html) support
- Built for [high performance](https://docs.lightly.ai/train/stable/performance/index.html) including [multi-GPU](https://docs.lightly.ai/train/stable/performance/multi_gpu.html) and [multi-node](https://docs.lightly.ai/train/stable/performance/multi_node.html) support
- [Monitor training progress](https://docs.lightly.ai/train/stable/train.html#loggers) with MLflow, TensorBoard, Weights & Biases, and more
- Runs fully on-premises with no API authentication and no telemetry

### Supported Models

LightlyTrain supports a wide range of frameworks and models out of the box.

| Framework | Model | Pretrain<br><sub>(*Unlabeled Images*)</sub> | Distill From<br> DINOv2/v3<br><sub>(*Unlabeled Images*)</sub> | Fine-tune<br><sub>(*Labeled Images*)</sub> | | |
|:--------------:|:------------:|:------------------------------------------------------------------------:|:-------:|:----------------------:|:----------------------:|:----------------------:|
| | | | | Object<br>Detection | Instance<br>Segmentation | Semantic<br>Segmentation |
| LightlyTrain | DINOv3 | | ✅ [🔗](https://docs.lightly.ai/train/stable/methods/distillation.html#distill-from-dinov3) | ✅ [🔗](https://docs.lightly.ai/train/stable/object_detection.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/instance_segmentation.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/semantic_segmentation.html#use-eomt-with-dinov3) |
| | DINOv2 | ✅ [🔗](https://docs.lightly.ai/train/stable/methods/dinov2.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/methods/distillation.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/object_detection.html) | | ✅ [🔗](https://docs.lightly.ai/train/stable/semantic_segmentation.html) |
| Torchvision | ResNet | ✅ [🔗](https://docs.lightly.ai/train/stable/models/torchvision.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/torchvision.html) | | | |
| | ConvNext | ✅ [🔗](https://docs.lightly.ai/train/stable/models/torchvision.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/torchvision.html) | | | |
| | ShuffleNetV2 | ✅ [🔗](https://docs.lightly.ai/train/stable/models/torchvision.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/torchvision.html) | | | |
| TIMM | All models | ✅ [🔗](https://docs.lightly.ai/train/stable/models/timm.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/timm.html) | | | |
| Ultralytics | YOLOv5 | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | | | |
| | YOLOv6 | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | | | |
| | YOLOv8 | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | | | |
| | YOLO11 | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | | | |
| | YOLO12 | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | | | |
| RT-DETR | RT-DETR | ✅ [🔗](https://docs.lightly.ai/train/stable/models/rtdetr.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/rtdetr.html) | | | |
| | RT-DETRv2 | ✅ [🔗](https://docs.lightly.ai/train/stable/models/rtdetr.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/rtdetr.html) | | | |
| RF-DETR | RF-DETR | ✅ [🔗](https://docs.lightly.ai/train/stable/models/rfdetr.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/rfdetr.html) | | | |
| YOLOv12 | YOLOv12 | ✅ [🔗](https://docs.lightly.ai/train/stable/models/yolov12.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/yolov12.html) | | | |
| SuperGradients | PP-LiteSeg | ✅ [🔗](https://docs.lightly.ai/train/stable/models/supergradients.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/supergradients.html) | | | |
| | SSD | ✅ [🔗](https://docs.lightly.ai/train/stable/models/supergradients.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/supergradients.html) | | | |
| | YOLO-NAS | ✅ [🔗](https://docs.lightly.ai/train/stable/models/supergradients.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/supergradients.html) | | | |
| Custom Models | Any PyTorch model | ✅ [🔗](https://docs.lightly.ai/train/stable/models/custom_models.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/custom_models.html) | | | |

For an overview of all supported models and usage instructions, see the full [model docs](https://docs.lightly.ai/train/stable/models/index.html).

[Contact](#contact) us if you need support for additional models or libraries.

### Supported Pretraining & Distillation Methods

- [DINOv3 Distillation](https://docs.lightly.ai/train/stable/methods/index.html#methods-distillation-dinov3)
- [DINOv2 Distillation](https://docs.lightly.ai/train/stable/methods/index.html#methods-distillation) (recommended 🚀)
- [DINOv2](https://docs.lightly.ai/train/stable/methods/index.html#methods-dinov2)
- [DINO](https://docs.lightly.ai/train/stable/methods/index.html#methods-dino)
- [SimCLR](https://docs.lightly.ai/train/stable/methods/index.html#methods-simclr)

See the full [methods docs](https://docs.lightly.ai/train/stable/methods/index.html) for details.

## FAQ

<details>
<summary><strong>Who is LightlyTrain for?</strong></summary>

LightlyTrain is designed for engineers and teams who want to use their unlabeled data to its
full potential. It is ideal if any of the following applies to you:

- You want to speedup model development cycles
- You have limited labeled data but abundant unlabeled data
- You have slow and expensive labeling processes
- You want to build your own foundation model
- You work with domain-specific datasets (video analytics, robotics, medical, agriculture, etc.)
- You cannot use public pretrained models
- No pretrained models are available for your specific architecture
- You want to leverage the latest research in self-supervised learning and distillation

</details>

<details>
<summary><strong>How much data do I need?</strong></summary>

We recommend a minimum of several thousand unlabeled images for training with LightlyTrain and 100+ labeled images for fine-tuning afterwards.

For best results:

- Use at least 5x more unlabeled than labeled data
- Even a 2x ratio of unlabeled to labeled data yields strong improvements
- Larger datasets (>100,000 images) benefit from pretraining up to 3,000 epochs
- Smaller datasets (\<100,000 images) benefit from longer pretraining of up to 10,000 epochs

The unlabeled dataset must always be treated like a training split—never include validation images in pretraining to avoid data leakage.

</details>

<details>
<summary><strong>What's the difference between LightlyTrain and other self-supervised learning implementations?</strong></summary>

LightlyTrain offers several advantages:

- **User-friendly**: You don't need to be an SSL expert - focus on training your model instead of implementation details.
- **Works with various model architectures**: Integrates directly with different libraries such as Torchvision, Ultralytics, etc.
- **Handles complexity**: Manages scaling from single GPU to multi-GPU training and optimizes hyperparameters.
- **Seamless workflow**: Automatically pretrains the correct layers and exports models in the right format for fine-tuning.

</details>

<details>
<summary><strong>Why should I use LightlyTrain instead of other already pretrained models?</strong></summary>

LightlyTrain is most beneficial when:

- **Working with domain-specific data**: When your data has a very different distribution from standard datasets (medical images, industrial data, etc.)
- **Facing policy or license restrictions**: When you can't use models pretrained on datasets with unclear licensing
- **Having limited labeled data**: When you have access to a lot of unlabeled data but few labeled examples
- **Using custom architectures**: When no pretrained checkpoints are available for your model

LightlyTrain is complementary to existing pretrained models and can start from either random weights or existing pretrained weights.

</details>

Check our [complete FAQ](https://docs.lightly.ai/train/stable/faq.html) for more information.

## Usage Events

LightlyTrain collects anonymous usage events to help us improve the product. We only track training method, model architecture, and system information (OS, GPU).
To opt-out, set the environment variable: `export LIGHTLY_TRAIN_EVENTS_DISABLED=1`

## License

Lightly**Train** offers flexible licensing options to suit your specific needs:

- **AGPL-3.0 License**: Perfect for open-source projects, academic research, and community contributions.
  Share your innovations with the world while benefiting from community improvements.

- **Commercial License**: Ideal for businesses and organizations that need proprietary development freedom.
  Enjoy all the benefits of LightlyTrain while keeping your code and models private.

- **Free Community License**: Available for students, researchers, startups in early stages, or anyone exploring or experimenting with LightlyTrain.
  Empower the next generation of innovators with full access to the world of pretraining.

We're committed to supporting both open-source and commercial users.
[Contact us](https://www.lightly.ai/contact) to discuss the best licensing option for your project!

## Contact

[![Website](https://img.shields.io/badge/Website-lightly.ai-blue?style=for-the-badge&logo=safari&logoColor=white)](https://www.lightly.ai/lightly-train) <br>
[![Discord](https://img.shields.io/discord/752876370337726585?style=for-the-badge&logo=discord&logoColor=white&label=discord&color=7289da)](https://discord.gg/xvNJW94) <br>
[![GitHub](https://img.shields.io/badge/GitHub-lightly--ai-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/lightly-ai/lightly-train) <br>
[![X](https://img.shields.io/badge/X-lightlyai-black?style=for-the-badge&logo=x&logoColor=white)](https://x.com/lightlyai) <br>
[![YouTube](https://img.shields.io/badge/YouTube-lightly--tech-blue?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/channel/UCAz60UdQ9Q3jPqqZi-6bmXw) <br>
[![LinkedIn](https://img.shields.io/badge/LinkedIn-lightly--tech-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/lightly-tech)
