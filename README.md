# LightlyTrain - SOTA Pretraining, Fine-tuning and Distillation

[![Python](https://img.shields.io/badge/Python-3.8%7C3.9%7C3.10%7C3.11%7C3.12-blue.svg)](https://docs.lightly.ai/train/stable/installation.html)
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

- \[[0.12.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-12-0)\] - 2025-11-06: 💡 **New DINOv3 Object Detection:** Run inference or fine-tune DINOv3 models for [object detection](https://docs.lightly.ai/train/stable/object_detection.html)! 💡
- \[[0.11.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-11-0)\] - 2025-08-15: 🚀 **New DINOv3 Support:** Pretrain your own model with [distillation](https://docs.lightly.ai/train/stable/methods/distillation.html#methods-distillation-dinov3) from DINOv3 weights. Or fine-tune our SOTA [EoMT semantic segmentation model](https://docs.lightly.ai/train/stable/semantic_segmentation.html#semantic-segmentation-eomt-dinov3) with a DINOv3 backbone! 🚀
- \[[0.10.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-10-0)\] - 2025-08-04:
  🔥 **Train state-of-the-art semantic segmentation models** with our new
  [**DINOv2 semantic segmentation**](https://docs.lightly.ai/train/stable/semantic_segmentation.html)
  fine-tuning method! 🔥
- \[[0.9.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-9-0)\] - 2025-07-21:
  [**DINOv2 pretraining**](https://docs.lightly.ai/train/stable/methods/dinov2.html) is
  now officially available!

## Installation

Install Lightly**Train** on Python 3.8+ for Windows, Linux or MacOS with:

```bash
pip install lightly-train
```

## Workflows

<details open>
<summary><strong>Object Detection</strong></summary>

Train LTDETR detection models with DINOv2 or DINOv3 backbones.

#### COCO Results

| Implementation | Model | Val mAP<sub>50:95</sub> | Latency (ms) | Params (M) | Input Size |
|:--------------:|:----------------------------:|:------------------:|:------------:|:-----------:|:----------:|
| LightlyTrain | dinov2/vits14-ltdetr-coco | 55.7 | 16.87 | 55.3 | 644×644 |
| LightlyTrain | dinov3/convnext-tiny-ltdetr-coco | 54.4 | 13.29 | 61.1 | 640×640 |
| LightlyTrain | dinov3/convnext-small-ltdetr-coco | 56.9 | 17.65 | 82.7 | 640×640 |
| LightlyTrain | dinov3/convnext-base-ltdetr-coco | 58.6 | 24.68 | 121.0 | 640×640 |
| LightlyTrain | dinov3/convnext-large-ltdetr-coco | 60.0 | 42.30 | 230.0 | 640×640 |

Models are trained for 60 epochs on the COCO 2017 dataset and evaluated on the validation
set with single-scale testing. Latency is measured with TensorRT on a NVIDIA T4 GPU with
batch size 1. All models are compiled and optimized using `tensorrt==10.13.3.9`.

#### Usage

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_object_detection(
        out="out/my_experiment",
        model="dinov3/convnext-small-ltdetr-coco",
        data={
            "path": "my_data_dir",
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "person",
                1: "bicycle",
                2: "car",
            },
        },
    )
```

See the [documentation](https://docs.lightly.ai/train/stable/object_detection.html) and
[Google Colab notebook](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb)
for more details.

</details>

<details>
<summary><strong>Instance Segmentation</strong></summary>

Train state-of-the-art instance segmentation models with DINOv3 backbones using the
EoMT method from CVPR 2025.

#### COCO Results

| Implementation | Model | Val mAP mask | Avg. FPS | Params (M) | Input Size |
|----------------|----------------|-------------|----------|-----------|------------|
| LightlyTrain | dinov3/vits16-eomt-inst-coco | 32.6 | 51.5 | 21.6 | 640×640 |
| LightlyTrain | dinov3/vitb16-eomt-inst-coco | 40.3 | 25.2 | 85.7 | 640×640 |
| LightlyTrain | dinov3/vitl16-eomt-inst-coco | **46.2** | 12.5 | 303.2 | 640×640 |
| EoMT (CVPR 2025 paper, current SOTA) | dinov3/vitl16-eomt-inst-coco | 45.9 | - | 303.2 | 640×640 |

Models are trained for 12 epochs on the COCO 2017 dataset and evaluated on the validation
set with single-scale testing. Avg. FPS is measured on a single NVIDIA T4 GPU with batch
size 1. All models are compiled and optimized using `torch.compile`.

#### Usage

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_instance_segmentation(
        out="out/my_experiment",
        model="dinov3/vitb16-eomt-inst-coco",
        data={
            "path": "my_data_dir",
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "background",
                1: "vehicle",
                2: "pedestrian",
                # ...
            },
        },
    )
```

See the [documentation](https://docs.lightly.ai/train/stable/instance_segmentation.html)
and
[Google Colab notebook](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_instance_segmentation.ipynb)
for more details.

</details>

<details>
<summary><strong>Semantic Segmentation</strong></summary>

Train state-of-the-art semantic segmentation models with DINOv2 or DINOv3 backbones using
the EoMT method from CVPR 2025.

#### COCO-Stuff Results

| Implementation | Model | Val mIoU | Avg. FPS | Params (M) | Input Size |
|----------------|----------------------|----------|----------|-----------|------------|
| LightlyTrain | dinov3/vits16-eomt-coco | 0.465 | 88.7 | 21.6 | 512×512 |
| LightlyTrain | dinov3/vitb16-eomt-coco | 0.520 | 43.3 | 85.7 | 512×512 |
| LightlyTrain | dinov3/vitl16-eomt-coco | **0.544** | 20.4 | 303.2 | 512×512 |

Models are trained for 12 epochs with `num_queries=200` on the COCO-Stuff dataset and
evaluated on the validation set with single-scale testing. Avg. FPS is measured on a
single NVIDIA T4 GPU with batch size 1. All models are compiled and optimized using
`torch.compile`.

#### Cityscapes Results

| Implementation | Model | Val mIoU | Avg. FPS | Params (M) | Input Size |
|:------------------------------------:|:------------------------------:|:---------:|:--------:|:-----------:|:----------:|
| LightlyTrain | dinov3/vits16-eomt-cityscapes | 0.786 | 18.6 | 21.6 | 1024×1024 |
| LightlyTrain | dinov3/vitb16-eomt-cityscapes | 0.810 | 8.7 | 85.7 | 1024×1024 |
| LightlyTrain | dinov3/vitl16-eomt-cityscapes | **0.844** | 3.9 | 303.2 | 1024×1024 |
| EoMT (CVPR 2025 paper, current SOTA) | dinov2/vitl16-eomt | 0.842 | - | 319 | 1024×1024 |

Avg. FPS is measured on a single NVIDIA T4 GPU with batch size 1. All models are compiled
and optimized using `torch.compile`.

#### Usage

```python
import lightly_train

if __name__ == "__main__":
  lightly_train.train_semantic_segmentation(
    out="out/my_experiment",
    model="dinov3/vits16-eomt",
    data={
      "train": {
        "images": "my_data_dir/train/images",
        "masks": "my_data_dir/train/masks",
      },
      "val": {
        "images": "my_data_dir/val/images",
        "masks": "my_data_dir/val/masks",
      },
      "classes": {
        0: "background",
        1: "road",
        2: "building",
        # ...
      },
    },
  )
```

See the [documentation](https://docs.lightly.ai/train/stable/semantic_segmentation.html)
and
[Google Colab notebook](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_semantic_segmentation.ipynb)
for more details.

</details>

<details>
<summary><strong>Distillation (DINOv2/v3)</strong></summary>

Pretrain any model architecture with unlabeled data by distilling the knowledge from
DINOv2 or DINOv3 foundation models into your model. On the COCO dataset, YOLOv8-s models
pretrained with LightlyTrain achieve high performance across all tested label fractions.
These improvements hold for other architectures like YOLOv11, RT-DETR, and Faster R-CNN.
See our [announcement post](https://www.lightly.ai/blog/introducing-lightly-train)
for more benchmarks and details.

![Benchmark Results](https://cdn.prod.website-files.com/62cd5ce03261cb3e98188470/67fe4efa0209fb4eb0c3da5c_Introducing%20LightlyTrain_imag_1.png)

#### Usage

```python
import lightly_train

if __name__ == "__main__":
  lightly_train.train(
    out="out/my_experiment",
    data="my_data_dir",
    model="ultralytics/yolov8s",
    method="distillation",
    method_args={
      "teacher": "dinov3/vitb16",
    },
  )
```

See the [documentation](https://docs.lightly.ai/train/stable/methods/distillation.html)
on how to get started!

</details>

<details>
<summary><strong>Pretraining (DINOv2 Foundation Models)</strong></summary>

With LightlyTrain you can train your very own foundation model like DINOv2 on your data.

#### ImageNet-1K Results

| Implementation | Model | Val ImageNet k-NN |
|:--------------:|:-----:|:------------------:|
| LightlyTrain | dinov2/vitl16 | **81.9%** |
| DINOv2 | dinov2/vitl16 | 81.6% |

Models are pretrained on ImageNet-1k for 100 epochs and evaluated with a k-NN classifier
on the ImageNet validation set.

#### Usage

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

See the [documentation](https://docs.lightly.ai/train/stable/methods/dinov2.html)
for more details.

</details>

<details>
<summary><strong>Autolabeling</strong></summary>

LightlyTrain provides simple commands to autolabel your unlabeled data using DINOv2 or
DINOv3 pretrained models. This allows you to efficiently boost performance of your
smaller models by leveraging all your unlabeled images.

#### ADE20K Results

| Implementation | Model | Autolabel | Val mIoU | Params (M) | Input Size |
|:--------------:|:----------------------------:|:---------:|:---------:|:-----------:|:----------:|
| LightlyTrain | dinov3/vits16-eomt | ❌ | 0.466 | 21.6 | 518×518 |
| LightlyTrain | dinov3/vits16-eomt-ade20k | ✅ | **0.533** | 21.6 | 518×518 |
| LightlyTrain | dinov3/vitb16-eomt | ❌ | 0.544 | 85.7 | 518×518 |
| LightlyTrain | dinov3/vitb16-eomt-ade20k | ✅ | **0.573** | 85.7 | 518×518 |

The better results with auto-labeling were achieved by fine-tuning a ViT-H+ on the
ADE20K dataset, which reaches 0.595 validation mIoU. This model was then used to autolabel
100k images from the SUN397 dataset. Using these labels, we subsequently fine-tuned the
smaller models, and then used the ADE20k dataset for validation.

#### Usage

```python
import lightly_train

if __name__ == "__main__":
  lightly_train.predict_semantic_segmentation(
    out="out/my_autolabeled_data",
    data="my_data_dir",
    model="dinov3/vitb16-eomt-ade20k",
  )
```

See the [documentation](https://docs.lightly.ai/train/stable/predict_autolabel.html) for
more details.

</details>

## Features

- Python, Command Line, and [Docker](https://docs.lightly.ai/train/stable/docker.html) support
- Built for [high performance](https://docs.lightly.ai/train/stable/performance/index.html) including [multi-GPU](https://docs.lightly.ai/train/stable/performance/multi_gpu.html) and [multi-node](https://docs.lightly.ai/train/stable/performance/multi_node.html) support
- [Monitor training progress](https://docs.lightly.ai/train/stable/train.html#loggers) with MLflow, TensorBoard, Weights & Biases, and more
- Runs fully on-premises with no API authentication
- Export models in their native format for fine-tuning or inference
- Export models in ONNX or TensorRT format for edge deployment

## Models

LightlyTrain supports the following model and workflow combinations.

Scroll right to see all columns!

| Framework | Model | Object Detection<br><sub>(*Labeled Images*)</sub> | Instance Segmentation<br><sub>(*Labeled Images*)</sub> | Semantic Segmentation<br><sub>(*Labeled Images*)</sub> | Distill From<br> DINOv2/v3<br><sub>(*Unlabeled Images*)</sub> | Pretrain<br><sub>(*Unlabeled Images*)</sub> | Autolabel |
|:--------------:|:------------:|:----------------------:|:----------------------:|:----------------------:|:-------:|:------------------------------------------------------------------------:|:---------:|
| LightlyTrain | DINOv3 | ✅ [🔗](https://docs.lightly.ai/train/stable/object_detection.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/instance_segmentation.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/semantic_segmentation.html#use-eomt-with-dinov3) | ✅ [🔗](https://docs.lightly.ai/train/stable/methods/distillation.html#distill-from-dinov3) | | ✅ [🔗](https://docs.lightly.ai/train/stable/predict_autolabel.html) |
| | DINOv2 | ✅ [🔗](https://docs.lightly.ai/train/stable/object_detection.html) | | ✅ [🔗](https://docs.lightly.ai/train/stable/semantic_segmentation.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/methods/distillation.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/methods/dinov2.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/predict_autolabel.html) |
| Torchvision | ResNet | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/torchvision.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/torchvision.html) | |
| | ConvNext | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/torchvision.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/torchvision.html) | |
| | ShuffleNetV2 | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/torchvision.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/torchvision.html) | |
| TIMM | All models | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/timm.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/timm.html) | |
| Ultralytics | YOLOv5 | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | |
| | YOLOv6 | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | |
| | YOLOv8 | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | |
| | YOLO11 | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | |
| | YOLO12 | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/ultralytics.html) | |
| RT-DETR | RT-DETR | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/rtdetr.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/rtdetr.html) | |
| | RT-DETRv2 | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/rtdetr.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/rtdetr.html) | |
| RF-DETR | RF-DETR | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/rfdetr.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/rfdetr.html) | |
| YOLOv12 | YOLOv12 | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/yolov12.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/yolov12.html) | |
| SuperGradients | PP-LiteSeg | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/supergradients.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/supergradients.html) | |
| | SSD | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/supergradients.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/supergradients.html) | |
| | YOLO-NAS | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/supergradients.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/supergradients.html) | |
| Custom Models | Any PyTorch model | | | | ✅ [🔗](https://docs.lightly.ai/train/stable/models/custom_models.html) | ✅ [🔗](https://docs.lightly.ai/train/stable/models/custom_models.html) | |

[Contact us](https://www.lightly.ai/contact) if you need support for additional models or libraries.

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
