(depth-estimation)=

# Monocular Depth Estimation with fastai U-Net (Advanced)

```{important}
This tutorial requires substantial computational resources. We recommend at least 4 x RTX-4090 GPUs (or comparable) and approximately 3-4 days of training time.
```

This advanced tutorial demonstrates how to pretrain and fine-tune a U-Net from [fast.ai](https://github.com/fastai/fastai) for monocular depth estimation while exploring the customization capabilities of Lightly**Train**. We will pretrain two ResNet-50 encoders with different augmentation settings to analyze their impact on model performance.

To begin, install the required dependencies:

```bash
pip install lightly-train fastai
```

The tutorial consists of three main steps:

1. Dataset acquisition and preprocessing for pretraining and fine-tuning
1. Pretraining of two U-Net encoders using Lightly**Train** with distinct augmentation configurations
1. Fine-tuning and performance comparison of both networks

## Data Downloading and Processing

For this implementation, we utilize two complementary datasets: [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/) for pretraining and [DIODE](https://diode-dataset.org/) for fine-tuning. MegaDepth provides a comprehensive collection of outdoor scenes with synthetic depth maps derived from structure-from-motion reconstruction. While the synthetic depth maps aren't used during pretraining, the dataset's extensive outdoor scene distribution aligns well with our target domain. DIODE complements this with high-precision LiDAR-scanned ground-truth depth maps, ensuring accurate supervision during fine-tuning.

To obtain the MegaDepth dataset run the following command (approximately 200GB):

```{note}
Due to the lengthy download process we recommend using a terminal multiplexer such as `tmux`.
```

```bash
wget https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz
```

For the DIODE dataset, download both training and validation splits (approximately 110GB combined):

```bash
wget http://diode-dataset.s3.amazonaws.com/train.tar.gz
wget http://diode-dataset.s3.amazonaws.com/val.tar.gz
```

To inspect the characteristics of both datasets, we can visualize representative samples:

```python
import glob
from pathlib import Path
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

root = "/datasets/MegaDepthv1" # replace with your dataset root
root = Path(root)

imgs = glob.glob(f"{str(root)}/**/*.jpg", recursive=True)
shuffle(imgs)

print(f"Total images: {len(imgs)}") # MegaDepth contains 128228 images

imgs = [np.array(Image.open(img)) for img in imgs[:10]]

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i, img in enumerate(imgs):
    ax = axs[i // 5, i % 5]
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"img {i}")
plt.tight_layout()
plt.show()
```

![MegaDepth Sample](megadepth_samples.png)

```python
import glob
from pathlib import Path
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

root = "/datasets/DIODE/train/outdoor" # replace this with your dataset root
root = Path(root)

imgs = glob.glob(f"{str(root)}/**/*.png", recursive=True)
shuffle(imgs)

print(f"Total Outdoors Train Images: {len(imgs)}") # DIODE has 16884 outdoor images

imgs = imgs[:5]
corr_depths = [np.load(elem.replace("image", "depth").replace(".png", "_depth.npy")) for elem in imgs]
imgs = [np.array(Image.open(img)) for img in imgs]
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i, img in enumerate(imgs):
    ax = axs[0, i]
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"img {i}")
for i, depth in enumerate(corr_depths):
    ax = axs[1, i]
    ax.imshow(depth, cmap="viridis")
    ax.axis("off")
    ax.set_title(f"depth {i}")
plt.tight_layout()
plt.show()
```

![DIODE Samples](diode_outdoor_samples.png)

## Pretraining

The key to effective pretraining lies in the augmentation strategy. Looking at the MegaDepth dataset, we observe a consistent spatial hierarchy - objects at the top of images are typically further away than those at the bottom (consider the sky-to-ground relationship). This spatial consistency means we should avoid training a rotation-invariant model, unlike in scenarios with satellite or aerial imagery where rotation-invariance would be desirable.

To empirically demonstrate the impact of the augmentation choices, we'll train two encoders:

1. One with aggressive rotations (90°) and vertical flips.
1. One with conservative tilts (15°) and no vertical flips.

Those parameters can be adjusted with `lightly_train.train`'s `transform_args` argument, which expects a dictionary of augmentation parameters.

```python
# pretrain.py
import lightly_train

# Change this to turn on the aggressive rotations.
ROTATION_OFF = True

def get_transform_args(rotation_off: bool) -> dict:
    if ROTATION_OFF:
        transform_args = {
            "random_flip": {
                "vertical_prob": 0.0,
                "horizontal_prob": 0.5,
            },
            "random_rotation": {
                "prob": 1.0,
                "degrees": 15,
            }
        }
    else:
        transform_args = {
            "random_flip": {
                "vertical_prob": 0.5,
                "horizontal_prob": 0.5,
            },
            "random_rotation": {
                "prob": 1.0,
                "degrees": 90,
            }
        }
    return transform_args

if __name__ == "__main__":
           
    lightly_train.train(
        out=f"pretrain_logs/megadepth_rotationOff{ROTATION_OFF}",
        data="/datasets/MegaDepthv1",
        model="torchvision/resnet50",
        epochs=500,
        transform_args=get_transform_args(),
    )
```

## Fine-tuning

For fine-tuning, we implement a custom depth estimation pipeline using PyTorch Lightning. While fast.ai provides excellent high-level abstractions for a lot of downstream tasks, depth estimation is not available out-of-the-box. Let's start by implementing our model, which inherits from `LightningModule`.

```python
# model.py
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.nn import Module
from torch.optim import Adam


class DepthUnet(LightningModule):
    def __init__(self, unet: Module):
        super().__init__()
        self.unet = unet
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        out = self(batch["image"])
        loss = F.mse_loss(out, batch["depth"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch["image"])
        loss = F.mse_loss(out, batch["depth"])
        self.log("val_loss", loss)
        return loss

    def forward(self, x):
        x = self.unet(x)
        return x

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optim
```

Our model implementation uses MSE loss, which while simple, is effective for depth estimation when combined with proper normalization. As you can see, the batch is supposed to arrive in the model as a dictionary, for which we will implement a custom dataset in the next step.

```python
# datasets.py
import glob
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DIODEDepthDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        split: str, 
        transform: Callable = None, 
        outdoor_only: bool = False, 
        indoor_only: bool = False
    ):
        if outdoor_only and indoor_only:
            raise ValueError("Cannot specify both outdoor_only and indoor_only")
        if split not in ["train", "val"]:
            raise ValueError("split must be 'train' or 'val'")
        if outdoor_only:
            self.imgs = sorted(glob.glob(f"{data_dir}/{split}/outdoor/**/*.png", recursive=True))
            self.depths = sorted(glob.glob(f"{data_dir}/{split}/outdoor/**/*_depth.npy", recursive=True))
        elif indoor_only:
            self.imgs = sorted(glob.glob(f"{data_dir}/{split}/indoors/**/*.png", recursive=True))
            self.depths = sorted(glob.glob(f"{data_dir}/{split}/indoors/**/*_depth.npy", recursive=True))
        else:
            self.imgs = sorted(glob.glob(f"{data_dir}/{split}/**/*.png", recursive=True))
            self.depths = sorted(glob.glob(f"{data_dir}/{split}/**/*_depth.npy", recursive=True))

        self.transform = transform

        assert len(self.imgs) == len(self.depths), "Mismatch in number of images and depth maps"

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        depth = self.depths[idx]

        img = np.array(Image.open(img).convert("RGB"))
        depth = np.load(depth)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth).permute(2, 0, 1).float()

        if self.transform:
            img = self.transform(torch.cat([img, depth], dim=0))
            _img = img[:3]
            depth = img[3:4]
            img = _img

        return {"image": img, "depth": depth}
```

We focus on outdoor scenes to maintain domain alignment with our MegaDepth pretraining. For the augmentation pipeline we will stay conservative, only allowing slight rotational corrections (±15°) to account for camera tilt while preserving the crucial vertical spatial relationships in depth estimation. This will also make any performance differences between the encoders attributable to the different pretraining strategies.

With this we can finalize our fine-tuning script (make sure to have `CKPT_PATH` point to one your pretrained checkpoints, or set it to `None` for fine-tuning from scratch):

```python
# finetune.py
import torch
import torchvision.transforms as T
from datasets import DIODEDepthDataset
from fastai.vision.models.unet import DynamicUnet
from model import DepthUnet
from pytorch_lightning import Trainer
from torch.nn import Module, Sequential
from torch.utils.data import DataLoader
from torchvision import models

# Change this to point to your LightlyTrain pretrained model.
CKPT_PATH = "<path-to-pretrained-model>"


def get_train_transform():
    return T.Compose([
        T.RandomRotation(degrees=15),
        T.RandomResizedCrop(size=(768, 768), scale=(0.2, 0.9)),
        T.RandomHorizontalFlip(),
    ])

def get_val_transform():
    return T.Compose([
        T.Resize(size=(768, 768)),
    ])

def init_model(ckpt_path: str | None, scratch: bool):
    encoder = models.resnet50()
    if not scratch:
        state_dict = torch.load(ckpt_path)
        # make sure that some keys match
        assert any(k in state_dict.keys() for k in encoder.state_dict().keys()), "No matching keys found in the checkpoint"
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
    unet = DynamicUnet(Sequential(*list(encoder.children())[:-2]), n_out=1, img_size=(768, 768))
    unet.train()
    return unet

def finetune_unet(
    unet: Module,
    data_dir: str,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    max_epochs: int = 10,
    num_workers: int = 4,
):
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset = DIODEDepthDataset(data_dir, "train", transform=train_transform, outdoor_only=True)
    val_dataset = DIODEDepthDataset(data_dir, "val", transform=val_transform, outdoor_only=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = DepthUnet(unet)
    model.hparams.learning_rate = learning_rate

    trainer = Trainer(
        max_epochs=max_epochs,
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":

    unet = init_model(CKPT_PATH, False)
    finetune_unet(
        unet = unet,
        data_dir = "/datasets/DIODE",
        batch_size = 32,
        learning_rate = 1e-4,
        max_epochs = 50,
        num_workers = 4,
    )
    print("Training completed! 🥳")
```

In order to compare the performance of the two pretrained backbones, you can launch tensorboard and inspect the finetuning runs in your browser.

```bash
tensorboard --logdir=lightning_logs
```
