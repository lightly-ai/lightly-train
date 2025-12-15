# <span style="display:none;">LightlyTrain Documentation</span>

<div style="text-align: center; margin-bottom: 2rem;">
  <a href="https://www.lightly.ai/lightly-train" target="_blank" class="mobile-only">
    <img src="_static/lightlyBannerMobile.svg" alt="LightlyTrain Banner" style="max-width: 100%; height: auto;" />
  </a>
  <a href="https://www.lightly.ai/lightly-train" target="_blank" class="desktop-only">
    <img src="_static/lightlyBanner.svg" alt="LightlyTrain Banner" style="max-width: 100%; height: auto;" />
  </a>
</div>

```{eval-rst}
.. image:: _static/lightly_train_light.svg
   :align: center
   :class: only-light

.. image:: _static/lightly_train_dark.svg
   :align: center
   :class: only-dark
```

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%7C3.9%7C3.10%7C3.11%7C3.12-blue.svg)](https://docs.lightly.ai/train/stable/installation.html)
[![OS](https://img.shields.io/badge/OS-Linux%7CMacOS%7CWindows-blue.svg)](https://docs.lightly.ai/train/stable/installation.html)
[![Docker](https://img.shields.io/badge/Docker-blue?logo=docker&logoColor=fff)](https://docs.lightly.ai/train/stable/docker.html#)
[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://docs.lightly.ai/train/stable/)
[![Discord](https://img.shields.io/discord/752876370337726585?logo=discord&logoColor=white&label=discord&color=7289da)](https://discord.gg/xvNJW94)

*Train Better Models, Faster*

LightlyTrain is the leading framework for transforming your data into state-of-the-art
computer vision models. It covers the entire model development lifecycle from pretraining
DINOv2/v3 vision foundation models on your unlabeled data to fine-tuning transformer and
YOLO models on detection and segmentation tasks for edge deployment.

[Contact us](https://www.lightly.ai/contact) to request a license for commercial use.

## News

- \[[0.13.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-13-0)\] - 2025-12-15: 🐥 **New Tiny Object Detection Models:** We release tiny DINOv3 models pretrained on COCO for [object detection](https://docs.lightly.ai/train/stable/object_detection.html#coco)! 🐥
- \[[0.12.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-12-0)\] - 2025-11-06: 💡 **New DINOv3 Object Detection:** Run inference or fine-tune DINOv3 models for [object detection](https://docs.lightly.ai/train/stable/object_detection.html)! 💡
- \[[0.11.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-11-0)\] - 2025-08-15: 🚀 **New DINOv3 Support:** Pretrain your own model with [distillation](https://docs.lightly.ai/train/stable/pretrain_distill/methods/distillation.html#methods-distillation-dinov3) from DINOv3 weights. Or fine-tune our SOTA [EoMT semantic segmentation model](https://docs.lightly.ai/train/stable/semantic_segmentation.html#semantic-segmentation-eomt-dinov3) with a DINOv3 backbone! 🚀
- \[[0.10.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-10-0)\] - 2025-08-04:
  🔥 **Train state-of-the-art semantic segmentation models** with our new
  [**DINOv2 semantic segmentation**](https://docs.lightly.ai/train/stable/semantic_segmentation.html)
  fine-tuning method! 🔥
- \[[0.9.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-9-0)\] - 2025-07-21:
  [**DINOv2 pretraining**](https://docs.lightly.ai/train/stable/pretrain_distill/methods/dinov2.html) is
  now officially available!

## Workflows

<!-- TODO(12/25, Guarin): Add image for each workflow. -->

````{grid} 1 1 2 3

```{grid-item-card} Object Detection
:link: object_detection.html
<!-- <img src="_static/images/object_detection/street.jpg" height="64"><br> -->
Train LTDETR detection models with DINOv2 or DINOv3 backbones.<br>
```

```{grid-item-card} Instance Segmentation
:link: instance_segmentation.html
<!-- <img src="_static/images/object_detection/street.jpg" height="64"><br> -->
Train EoMT segmentation models with DINOv3 backbones.<br>
```

```{grid-item-card} Panoptic Segmentation
:link: panoptic_segmentation.html
<!-- <img src="_static/images/object_detection/street.jpg" height="64"><br> -->
Train EoMT segmentation models with DINOv3 backbones.<br>
```

```{grid-item-card} Semantic Segmentation
:link: semantic_segmentation.html
<!-- <img src="_static/images/object_detection/street.jpg" height="64"><br> -->
Train EoMT segmentation models with DINOv2 or DINOv3 backbones.<br>
```

```{grid-item-card} Distillation
:link: pretrain_distill/methods/distillation.html
<!-- <img src="_static/images/object_detection/street.jpg" height="64"><br> -->
Distill knowledge from DINOv2 or DINOv3 into any model architecture.<br>
```

```{grid-item-card} Pretraining
:link: pretrain_distill/methods/dinov2.html
<!-- <img src="_static/images/object_detection/street.jpg" height="64"><br> -->
Pretrain DINOv2 foundation models on your domain data.<br>
```

```{grid-item-card} Autolabeling
:link: predict_autolabel.html
<!-- <img src="_static/images/object_detection/street.jpg" height="64"><br> -->
Generate high-quality pseudo labels for detection and segmentation tasks.<br>
```
````

## How It Works [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb)

Install Lightly**Train** on Python 3.8+ for Windows, Linux or MacOS.

```bash
pip install lightly-train
```

Then train an object detection model with:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_object_detection(
        out="out/my_experiment",
        model="dinov3/vitt16-ltdetr-coco",
        data={
            # ... Data configuration
        }
      )
```

And run inference like this:

```python
import lightly_train

# Load the model from the best checkpoint
model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
# Or load one of the models hosted by LightlyTrain
model = lightly_train.load_model("dinov3/vitt16-ltdetr-coco")
results = model.predict("image.jpg")
```

See the full [quick start guide](quick-start-object-detection) for more details.

## Features

- Python, Command Line, and [Docker](https://docs.lightly.ai/train/stable/docker.html) support
- Built for [high performance](https://docs.lightly.ai/train/stable/performance/index.html) including [multi-GPU](https://docs.lightly.ai/train/stable/performance/multi_gpu.html) and [multi-node](https://docs.lightly.ai/train/stable/performance/multi_node.html) support
- [Monitor training progress](https://docs.lightly.ai/train/stable/pretrain_distill.html#logging) with MLflow, TensorBoard, Weights & Biases, and more
- Runs fully on-premises with no API authentication
- Export models in their native format for fine-tuning or inference
- Export models in ONNX or TensorRT format for edge deployment

## Models

LightlyTrain supports the following model and workflow combinations.

### Fine-tuning

| Model | Object<br>Detection | Instance<br>Segmentation | Panoptic<br>Segmentation | Semantic<br>Segmentation |
| ------ | :----------------------------------------------------------------: | :---------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| DINOv3 | ✅ [🔗](https://docs.lightly.ai/train/stable/object_detection.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/instance_segmentation.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/panoptic_segmentation.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/semantic_segmentation.html#use-eomt-with-dinov3) |
| DINOv2 | ✅ [🔗](https://docs.lightly.ai/train/stable/object_detection.html) | | | ✅ [🔗](https://docs.lightly.ai/train/stable/semantic_segmentation.html) |

### Distillation & Pretraining

| Model | Distillation | Pretraining |
| ------------------------------ | :----------------------------------------------------------------------------------------: | :--------------------------------------------------------------------: |
| DINOv3 | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/methods/distillation.html#distill-from-dinov3) | |
| DINOv2 | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/methods/distillation.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/methods/dinov2.html) |
| Torchvision ResNet, ConvNext, ShuffleNetV2 | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/models/torchvision.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/models/torchvision.html) |
| TIMM models | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/models/timm.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/models/timm.html) |
| Ultralytics YOLOv5–YOLO12 | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/models/ultralytics.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/models/ultralytics.html) |
| RT-DETR, RT-DETRv2 | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/models/rtdetr.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/models/rtdetr.html) |
| RF-DETR | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/models/rfdetr.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/models/rfdetr.html) |
| YOLOv12 | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/models/yolov12.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/models/yolov12.html) |
| Custom PyTorch Model | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/models/custom_models.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/pretrain_distill/models/custom_models.html) |

[Contact us](https://www.lightly.ai/contact) if you need support for additional models.

## Usage Events

LightlyTrain collects anonymous usage events to help us improve the product. We only
track training method, model architecture, and system information (OS, GPU). To opt-out,
set the environment variable: `export LIGHTLY_TRAIN_EVENTS_DISABLED=1`

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
quick_start_object_detection
quick_start_distillation
installation
object_detection
instance_segmentation
panoptic_segmentation
semantic_segmentation
pretrain_distill/index
predict_autolabel
embed
data/index
performance/index
docker
tutorials/index
python_api/index
faq
changelog
```
