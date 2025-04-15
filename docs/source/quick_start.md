(quick-start)=

# Quick Start

```{image} https://colab.research.google.com/assets/colab-badge.svg
:target: https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/quick_start.ipynb
```

## Installation

```bash
pip install lightly-train
```

```{important}
Check the [Installation](installation.md#installation) page for supported platforms.
```

## Prepare Data

You can use any image dataset for training. No labels are required, and the dataset can
be structured in any way, including subdirectories. If you don't have a dataset at hand,
you can download one like this:

```bash
git clone https://github.com/lightly-ai/dataset_clothing_images.git my_data_dir
rm -rf my_data_dir/.git
```

See the [data guide](#train-data) for more information on supported data formats.

## Train

Once the data is ready, you can train the model like this:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",            # Output directory
        data="my_data_dir",                 # Directory with images
        model="torchvision/resnet18",       # Model to train
        epochs=10,                          # Number of epochs to train
        batch_size=32,                      # Batch size
    )
````

````{tab} Command Line
```bash
lightly-train train out="out/my_experiment" data="my_data_dir" model="torchvision/resnet18" epochs=10 batch_size=32
````

```{important}
This is a minimal example for illustration purposes. In practice you would want to use
a larger dataset (>=10'000 images), more epochs (>=100), and a larger batch size (>=128).

**Best Choice**: 
The default pretraining method `distillation` is recommended, as it consistently outperforms others in extensive experiments. Batch sizes between `128` and `1536` strike a good balance between speed and performance. Moreover, long training runs, such as 2,000 epochs on COCO, significantly improve results. Check the [Methods](#methods-comparison) page for more details why `distillation` is the best choice.
```

```{tip}
Lightly**Train** supports many [popular models](#models) out of the box.
```

This pretrains a Torchvision ResNet-18 model using images from `my_data_dir`.
All training logs, model exports, and checkpoints are saved to the output directory
at `out/my_experiment`.

Once the training is complete, the `out/my_experiment` directory contains the
following files:

```text
out/my_experiment
├── checkpoints
│   ├── epoch=99-step=123.ckpt          # Intermediate checkpoint
│   └── last.ckpt                       # Last checkpoint
├── events.out.tfevents.123.0           # Tensorboard logs
├── exported_models
|   └── exported_last.pt                # Final model exported
├── metrics.jsonl                       # Training metrics
└── train.log                           # Training logs
```

The final model is exported to `out/my_experiment/exported_models/exported_last.pt` in
the default format of the used library. It can directly be used for
fine-tuning. See [export format](export.md#format) for more information on how to export
models to other formats or on how to export intermediate checkpoints.

While the trained model has already learned good representations of the images, it
cannot yet make any predictions for tasks such as classification, detection, or
segmentation. To solve these tasks, the model needs to be fine-tuned on a labeled
dataset.

## Fine-Tune

Now the model is ready for fine-tuning! You can use your favorite library for this step.
Below is a simple example using PyTorch:

```python
import torch
import torchvision.transforms.v2 as v2
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
dataset = datasets.ImageFolder(root="my_data_dir", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

# Load the exported model
model = models.resnet18()
model.load_state_dict(torch.load("out/my_experiment/exported_models/exported_last.pt", weights_only=True))

# Update the classification head with the correct number of classes
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Starting fine-tuning...")
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for inputs, labels in progress_bar:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```

The output shows the loss decreasing over time:

```text
Starting fine-tuning...
Epoch [1/10], Loss: 2.1686
Epoch [2/10], Loss: 2.1290
Epoch [3/10], Loss: 2.1854
Epoch [4/10], Loss: 2.2936
Epoch [5/10], Loss: 1.9303
Epoch [6/10], Loss: 1.9949
Epoch [7/10], Loss: 1.8429
Epoch [8/10], Loss: 1.9873
Epoch [9/10], Loss: 1.8179
Epoch [10/10], Loss: 1.5360
```

Congratulations! You just trained and fine-tuned a model using Lightly**Train**!

## Embed

Instead of fine-tuning the model, you can also use it to generate image embeddings. This
is useful for clustering, retrieval, or visualization tasks. The `embed` command
generates embeddings for all images in a directory:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.embed(
        out="my_embeddings.pth",                                # Exported embeddings
        checkpoint="out/my_experiment/checkpoints/last.ckpt",   # LightlyTrain checkpoint
        data="my_data_dir",                                     # Directory with images
    )
````

````{tab} Command Line
```bash
lightly-train embed out="my_embeddings.pth" checkpoint="out/my_experiment/checkpoints/last.ckpt" data="my_data_dir"
````

The embeddings are saved to `my_embeddings.pth` and are loaded like this:

```python
import torch

embeddings = torch.load('my_embeddings.pth')
print(embeddings['filenames'][:5])       # Print first five filenames
print(embeddings['embeddings'].shape)    # Tensor with embeddings with shape (num_images, embedding_dim)
```
