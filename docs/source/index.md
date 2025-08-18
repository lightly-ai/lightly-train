# <span style="display:none;">LightlyTrain Documentation</span>

```{eval-rst}
.. image:: _static/lightly_train_light.svg
   :align: center
   :class: only-light

.. image:: _static/lightly_train_dark.svg
   :align: center
   :class: only-dark
```

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/quick_start.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%7C3.9%7C3.10%7C3.11%7C3.12-blue.svg)](https://docs.lightly.ai/train/stable/installation.html)
[![OS](https://img.shields.io/badge/OS-Linux%7CMacOS%7CWindows-blue.svg)](https://docs.lightly.ai/train/stable/installation.html)
[![Docker](https://img.shields.io/badge/Docker-blue?logo=docker&logoColor=fff)](https://docs.lightly.ai/train/stable/docker.html#)
[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://docs.lightly.ai/train/stable/)
[![Discord](https://img.shields.io/discord/752876370337726585?logo=discord&logoColor=white&label=discord&color=7289da)](https://discord.gg/xvNJW94)

*Train Better Models, Faster - No Labels Needed*

LightlyTrain brings self-supervised pretraining to real-world computer vision pipelines, using
your unlabeled data to reduce labeling costs and speed up model deployment. Leveraging the
state-of-the-art from research, it pretrains your model on your unlabeled, domain-specific
data, significantly reducing the amount of labeling needed to reach a high model performance.

This allows you to focus on new features and domains instead of managing your labeling cycles.
LightlyTrain is designed for simple integration into existing training pipelines and supports
a wide range of model architectures and use cases out of the box.

## News

- \[[0.11.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-11-0)\] - 2025-08-15: 🚀 **New DINOv3 Support:** Pretrain your own model with [distillation](https://docs.lightly.ai/train/stable/methods/distillation.html#methods-distillation-dinov3) from DINOv3 weights. Or fine-tune our SOTA [EoMT semantic segmentation model](https://docs.lightly.ai/train/stable/semantic_segmentation.html#semantic-segmentation-eomt-dinov3) with a DINOv3 backbone! 🚀
- \[[0.10.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-10-0)\] - 2025-08-04:
  🔥 **Train state-of-the-art semantic segmentation models** with our new
  [**DINOv2 semantic segmentation**](https://docs.lightly.ai/train/stable/semantic_segmentation.html)
  fine-tuning method! 🔥
- \[[0.9.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-9-0)\] - 2025-07-21:
  [**DINOv2 pretraining**](https://docs.lightly.ai/train/stable/methods/dinov2.html) is
  now officially available!
- \[[0.8.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-8-0)\] - 2025-06-10:
  [**DINOv2 pretraining**](https://docs.lightly.ai/train/stable/methods/dinov2.html) is
  now available (beta 🔬)!
- \[[0.7.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-7-0)\] - 2025-05-26:
  Up to **3x faster distillation** and higher accuracy with [**Distillation v2**](https://docs.lightly.ai/train/stable/methods/distillation.html)
  (new default method)!

## Why Lightly**Train**?

- 💸 **No Labels Required**: Speed up development by pretraining models on your unlabeled image and video data.
- 🔄 **Domain Adaptation**: Improve models by pretraining on your domain-specific data (e.g. video analytics, agriculture, automotive, healthcare, manufacturing, retail, and more).
- 🏗️ **Model & Task Agnostic**: Compatible with any architecture and task, including detection, classification, and segmentation.
- 🚀 **Industrial-Scale Support**: LightlyTrain scales from thousands to millions of images. Supports on-prem, cloud, single, and multi-GPU setups.

```{figure} https://cdn.prod.website-files.com/62cd5ce03261cb3e98188470/67fe4efa0209fb4eb0c3da5c_Introducing%20LightlyTrain_imag_1.png
:alt: benchmark results

On COCO, YOLOv8-s models pretrained with LightlyTrain achieve high performance across all tested label fractions.
These improvements hold for other architectures like YOLOv11, RT-DETR, and Faster R-CNN.
See our [announcement post](https://www.lightly.ai/blog/introducing-lightly-train) for more details.
```

## How It Works [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/quick_start.ipynb)

Install Lightly**Train**:

```bash
pip install lightly-train
```

Then start pretraining with:

```python
import lightly_train

if __name__ == "__main__":
  lightly_train.train(
      out="out/my_experiment",            # Output directory
      data="my_data_dir",                 # Directory with images
      model="torchvision/resnet50",       # Model to train
  )
```

This will pretrain a Torchvision ResNet-50 model using unlabeled images from `my_data_dir`.
All training logs, model exports, and checkpoints are saved to the output directory
at `out/my_experiment`. The final model is exported to `out/my_experiment/exported_models/exported_last.pt`.

Finally, load the pretrained model and fine-tune it using your existing training pipeline:

```python
import torch
from torchvision import models

# Load the pretrained model
model = models.resnet50()
model.load_state_dict(torch.load("out/my_experiment/exported_models/exported_last.pt", weights_only=True))

# Fine-tune the model with your existing training pipeline
...
```

```{seealso}
Looking for a full fine-tuning example? Head over to the [Quick Start](quick_start.md#fine-tune)!
```

```{seealso}
🔥 **New:** Want to train a state-of-the-art semantic segmentation model? Head over to
the [semantic segmentation guide](#semantic-segmentation)!
```

```{seealso}
Want to use your model to generate image embeddings instead? Check out the {ref}`embed` guide!
```

## Features

- Train models on any image data without labels
- Train models from popular libraries such as [Torchvision](#models-torchvision),
  [TIMM](#models-timm), [Ultralytics](#models-ultralytics), [SuperGradients](#models-supergradients),
  [RT-DETR](#models-rtdetr), [RF-DETR](#models-rfdetr), and [YOLOv12](#models-yolov12)
- Train [custom models](#custom-models) with ease
- No self-supervised learning expertise required
- Automatic SSL method selection (coming soon!)
- Python, Command Line, and {ref}`docker` support
- Built for [high performance](#performance) including [multi-GPU](#multi-gpu) and [multi-node](#multi-node) support
- {ref}`Export models <export>` for fine-tuning or inference
- Generate and export {ref}`image embeddings <embed>`
- [Monitor training progress](#logging) with TensorBoard, Weights & Biases, and more
- Runs fully on-premises with no API authentication and no telemetry

### Supported Models

| Library | Supported Models | Docs |
|------------------|----------------------------------------|------|
| Torchvision | ResNet, ConvNext, ShuffleNetV2 | [🔗](#models-torchvision) |
| TIMM | All models | [🔗](#models-timm) |
| Ultralytics | YOLOv5, YOLOv6, YOLOv8, YOLO11, YOLO12 | [🔗](#models-ultralytics) |
| RT-DETR | RT-DETR & RT-DETRv2 | [🔗](#models-rtdetr) |
| RF-DETR | RF-DETR | [🔗](#models-rfdetr) |
| YOLOv12 | YOLOv12 | [🔗](#models-yolov12) |
| SuperGradients | PP-LiteSeg, SSD, YOLO-NAS | [🔗](#models-supergradients) |
| Custom Models | Any PyTorch model | [🔗](#custom-models) |

For an overview of all supported models and usage instructions, see the full [model docs](#models-supported-libraries).

[Contact](#contact) us if you need support for additional models or libraries.

### Supported Training Methods

- [DINOv2 Distillation](#methods-distillation) (recommended 🚀)
- [DINOv2](#methods-dinov2)
- [DINO](#methods-dino)
- [SimCLR](#methods-simclr)

See the full [methods docs](#methods) for details.

## FAQ

```{dropdown} Who is LightlyTrain for?
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
```

```{dropdown} How much data do I need?
We recommend a minimum of several thousand unlabeled images for training with LightlyTrain and 100+ labeled images for fine-tuning afterwards.

For best results:

- Use at least 5x more unlabeled than labeled data
- Even a 2x ratio of unlabeled to labeled data yields strong improvements
- Larger datasets (>100,000 images) benefit from pretraining up to 3,000 epochs
- Smaller datasets (\<100,000 images) benefit from longer pretraining of up to 10,000 epochs

The unlabeled dataset must always be treated like a training split—never include validation images in pretraining to avoid data leakage.
```

```{dropdown} What's the difference between LightlyTrain and other self-supervised learning implementations?

LightlyTrain offers several advantages:

- **User-friendly**: You don't need to be an SSL expert - focus on training your model instead of implementation details.
- **Works with various model architectures**: Integrates directly with different libraries such as Torchvision, Ultralytics, etc.
- **Handles complexity**: Manages scaling from single GPU to multi-GPU training and optimizes hyperparameters.
- **Seamless workflow**: Automatically pretrains the correct layers and exports models in the right format for fine-tuning.
```

```{dropdown} Why should I use LightlyTrain instead of other already pretrained models?

LightlyTrain is most beneficial when:

- **Working with domain-specific data**: When your data has a very different distribution from standard datasets (medical images, industrial data, etc.)
- **Facing policy or license restrictions**: When you can't use models pretrained on datasets with unclear licensing
- **Having limited labeled data**: When you have access to a lot of unlabeled data but few labeled examples
- **Using custom architectures**: When no pretrained checkpoints are available for your model

LightlyTrain is complementary to existing pretrained models and can start from either random weights or existing pretrained weights.
```

Check our [complete FAQ](#faq) for more information.

## License

Lightly**Train** offers flexible licensing options to suit your specific needs:

- **AGPL-3.0 License**: Perfect for open-source projects, academic research, and community contributions.
  Share your innovations with the world while benefiting from community improvements.

- **Commercial License**: Ideal for businesses and organizations that need proprietary development freedom.
  Enjoy all the benefits of LightlyTrain while keeping your code and models private.

We're committed to supporting both open-source and commercial users.
Please [contact us](https://www.lightly.ai/contact) to discuss the best licensing option for your project!

## Contact

[![Website](https://img.shields.io/badge/Website-lightly.ai-blue?style=for-the-badge&logo=safari&logoColor=white)](https://www.lightly.ai/lightly-train) <br>
[![Discord](https://img.shields.io/discord/752876370337726585?style=for-the-badge&logo=discord&logoColor=white&label=discord&color=7289da)](https://discord.gg/xvNJW94) <br>
[![GitHub](https://img.shields.io/badge/GitHub-lightly--ai-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/lightly-ai/lightly-train) <br>
[![X](https://img.shields.io/badge/X-lightlyai-black?style=for-the-badge&logo=x&logoColor=white)](https://x.com/lightlyai) <br>
[![LinkedIn](https://img.shields.io/badge/LinkedIn-lightly--tech-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/lightly-tech)

```{toctree}
---
hidden:
maxdepth: 2
---
quick_start
installation
train/index
semantic_segmentation
export
embed
models/index
methods/index
performance/index
docker
tutorials/index
python_api/index
faq
changelog
```
