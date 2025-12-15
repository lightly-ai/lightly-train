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
| LightlyTrain | dinov3/vitt16-ltdetr-coco | 49.8 | 5.4 | 10.1 | 640×640 |
| LightlyTrain | dinov3/vitt16plus-ltdetr-coco | 52.5 | 7.0 | 18.1 | 640×640 |
| LightlyTrain | dinov3/vits16-ltdetr-coco | 55.4 | 10.5 | 36.4 | 640×640 |
| LightlyTrain | dinov2/vits14-noreg-ltdetr-coco | 55.7 | 16.9 | 55.3 | 644×644 |
| LightlyTrain | dinov3/convnext-tiny-ltdetr-coco | 54.4 | 13.3 | 61.1 | 640×640 |
| LightlyTrain | dinov3/convnext-small-ltdetr-coco | 56.9 | 17.7 | 82.7 | 640×640 |
| LightlyTrain | dinov3/convnext-base-ltdetr-coco | 58.6 | 24.7 | 121.0 | 640×640 |
| LightlyTrain | dinov3/convnext-large-ltdetr-coco | 60.0 | 42.3 | 230.0 | 640×640 |

Models are trained on the COCO 2017 dataset and evaluated on the validation
set with single-scale testing. Latency is measured with TensorRT on a NVIDIA T4 GPU with
batch size 1. All models are optimized using `tensorrt==10.13.3.9`.

#### Usage

[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://docs.lightly.ai/train/stable/object_detection.html)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb)

```python
import lightly_train

if __name__ == "__main__":
    # Train an object detection model with a DINOv3 backbone
    lightly_train.train_object_detection(
        out="out/my_experiment",
        model="dinov3/vitt16-ltdetr-coco",
        data={
            "path": "my_data_dir",
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "person",
                1: "bicycle",
                2: "car",
                # ...
            },
        },
    )

    # Load model and run inference
    model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
    # Or use one of the models provided by LightlyTrain
    # model = lightly_train.load_model("dinov3/vitt16-ltdetr-coco")
    results = model.predict("image.jpg")
    results["labels"]   # Class labels, tensor of shape (num_boxes,)
    results["bboxes"]   # Bounding boxes in (xmin, ymin, xmax, ymax) absolute pixel
                        # coordinates of the original image. Tensor of shape (num_boxes, 4).
    results["scores"]   # Confidence scores, tensor of shape (num_boxes,)
```

</details>

<details>
<summary><strong>Panoptic Segmentation</strong></summary>

Train state-of-the-art panoptic segmentation models with DINOv3 backbones using the
EoMT method from CVPR 2025.

#### COCO Results

| Implementation | Model | Val PQ | Avg. Latency (ms) | Params (M) | Input Size |
|----------------|----------------|-------------|----------|-----------|------------|
| LightlyTrain | dinov3/vits16-eomt-panoptic-coco | 46.8 | 21.2 | 23.4 | 640×640 |
| LightlyTrain | dinov3/vitb16-eomt-panoptic-coco | 53.2 | 39.4 | 92.5 | 640×640 |
| LightlyTrain | dinov3/vitl16-eomt-panoptic-coco | 57.0 | 80.1 | 315.1 | 640×640 |
| LightlyTrain | dinov3/vitl16-eomt-panoptic-coco-1280 | **59.0** | 500.1 | 315.1 | 1280×1280 |
| EoMT (CVPR 2025 paper, current SOTA) | dinov3/vitl16-eomt-panoptic-coco-1280 | 58.9 | - | 315.1 | 1280×1280 |

Small and base models are trained for 24 epochs and large models for 12 epochs on the
COCO 2017 dataset and evaluated on the validation set with single-scale testing.
Avg. Latency is measured on a single NVIDIA T4 GPU with batch size 1. All models are
optimized using `torch.compile`.

#### Usage

[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://docs.lightly.ai/train/stable/panoptic_segmentation.html)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_panoptic_segmentation.ipynb)

```python
import lightly_train

if __name__ == "__main__":
    # Train an panoptic segmentation model with a DINOv3 backbone
    lightly_train.train_panoptic_segmentation(
        out="out/my_experiment",
        model="dinov3/vitb16-eomt-panoptic-coco",
        data={
            "train": {
                "images": "images/train",
                "masks": "annotations/train",
                "annotations": "annotations/train.json",
            },
            "val": {
                "images": "images/val",
                "masks": "annotations/val",
                "annotations": "annotations/val.json",
            },
        },
    )

    model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
    results = model.predict("image.jpg")
    results["masks"]    # Masks with (class_label, segment_id) for each pixel, tensor of
                        # shape (height, width, 2). Height and width correspond to the
                        # original image size.
    results["segment_ids"]    # Segment ids, tensor of shape (num_segments,).
    results["scores"]   # Confidence scores, tensor of shape (num_segments,)
```

</details>

<details>
<summary><strong>Instance Segmentation</strong></summary>

Train state-of-the-art instance segmentation models with DINOv3 backbones using the
EoMT method from CVPR 2025.

#### COCO Results

| Implementation | Model | Val mAP mask | Avg. Latency (ms) | Params (M) | Input Size |
|----------------|----------------|-------------|----------|-----------|------------|
| LightlyTrain | dinov3/vits16-eomt-inst-coco | 32.6 | 19.4 | 21.6 | 640×640 |
| LightlyTrain | dinov3/vitb16-eomt-inst-coco | 40.3 | 39.7 | 85.7 | 640×640 |
| LightlyTrain | dinov3/vitl16-eomt-inst-coco | **46.2** | 80.0 | 303.2 | 640×640 |
| EoMT (CVPR 2025 paper, current SOTA) | dinov3/vitl16-eomt-inst-coco | 45.9 | - | 303.2 | 640×640 |

Models are trained for 12 epochs on the COCO 2017 dataset and evaluated on the validation
set with single-scale testing. Average latency is measured on a single NVIDIA T4 GPU with
batch size 1. All models are optimized using `torch.compile`.

#### Usage

[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://docs.lightly.ai/train/stable/instance_segmentation.html)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_instance_segmentation.ipynb)

```python
import lightly_train

if __name__ == "__main__":
    # Train an instance segmentation model with a DINOv3 backbone
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

    model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
    results = model.predict("image.jpg")
    results["labels"]   # Class labels, tensor of shape (num_instances,)
    results["masks"]    # Binary masks, tensor of shape (num_instances, height, width).
                        # Height and width correspond to the original image size.
    results["scores"]   # Confidence scores, tensor of shape (num_instances,)
```

</details>

<details>
<summary><strong>Semantic Segmentation</strong></summary>

Train state-of-the-art semantic segmentation models with DINOv2 or DINOv3 backbones using
the EoMT method from CVPR 2025.

#### COCO-Stuff Results

| Implementation | Model | Val mIoU | Avg. Latency (ms) | Params (M) | Input Size |
|----------------|----------------------|----------|----------|-----------|------------|
| LightlyTrain | dinov3/vits16-eomt-coco | 46.5 | 11.3 | 21.6 | 512×512 |
| LightlyTrain | dinov3/vitb16-eomt-coco | 52.0 | 23.1 | 85.7 | 512×512 |
| LightlyTrain | dinov3/vitl16-eomt-coco | **54.4** | 49.0 | 303.2 | 512×512 |

Models are trained for 12 epochs with `num_queries=200` on the COCO-Stuff dataset and
evaluated on the validation set with single-scale testing. Average latency is measured
on a single NVIDIA T4 GPU with batch size 1. All models optimized using `torch.compile`.

#### Cityscapes Results

| Implementation | Model | Val mIoU | Avg. Latency (ms) | Params (M) | Input Size |
|:------------------------------------:|:------------------------------:|:---------:|:--------:|:-----------:|:----------:|
| LightlyTrain | dinov3/vits16-eomt-cityscapes | 78.6 | 53.8 | 21.6 | 1024×1024 |
| LightlyTrain | dinov3/vitb16-eomt-cityscapes | 81.0 | 114.9 | 85.7 | 1024×1024 |
| LightlyTrain | dinov3/vitl16-eomt-cityscapes | **84.4** | 256.4 | 303.2 | 1024×1024 |
| EoMT (CVPR 2025 paper, current SOTA) | dinov2/vitl16-eomt | 84.2 | - | 319 | 1024×1024 |

Average latency is measured on a single NVIDIA T4 GPU with batch size 1. All models are
optimized using `torch.compile`.

#### Usage

[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://docs.lightly.ai/train/stable/semantic_segmentation.html)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_semantic_segmentation.ipynb)

```python
import lightly_train

if __name__ == "__main__":
    # Train a semantic segmentation model with a DINOv3 backbone
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

    # Load model and run inference
    model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
    # Or use one of the models provided by LightlyTrain
    # model = lightly_train.load_model("dinov3/vits16-eomt")
    masks = model.predict("image.jpg")
    # Masks is a tensor of shape (height, width) with class labels as values.
    # It has the same height and width as the input image.
```

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

[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://docs.lightly.ai/train/stable/pretrain_distill/methods/distillation.html)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/distillation.ipynb)

```python
import lightly_train

if __name__ == "__main__":
    # Distill the knowledge from a DINOv3 teacher into a YOLOv8 model
    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        model="ultralytics/yolov8s",
        method="distillation",
        method_args={
            "teacher": "dinov3/vitb16",
        },
    )

    # Load model for fine-tuning
    model = YOLO("out/my_experiment/exported_models/exported_last.pt")
    model.train(data="coco8.yaml")
```

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

[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://docs.lightly.ai/train/stable/pretrain_distill/methods/dinov2.html)

```python
import lightly_train

if __name__ == "__main__":
    # Pretrain a DINOv2 vision foundation model
    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        model="dinov2/vitb14",
        method="dinov2",
    )
```

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

[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://docs.lightly.ai/train/stable/predict_autolabel.html)

```python
import lightly_train

if __name__ == "__main__":
    # Autolabel your data with a DINOv3 semantic segmentation model
    lightly_train.predict_semantic_segmentation(
        out="out/my_autolabeled_data",
        data="my_data_dir",
        model="dinov3/vitb16-eomt-coco",
        # Or use one of your own model checkpoints
        # model="out/my_experiment/exported_models/exported_best.pt",
    )

    # The autolabeled masks will be saved in this format:
    # out/my_autolabeled_data
    # ├── <image name>.png
    # ├── <image name>.png
    # └── …
```

</details>

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
