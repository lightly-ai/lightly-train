(pretrain-distill)=

(train)=

# Pretrain & Distill

This part of the documentation focuses on how to improve your models with **unlabeled
data**. LightlyTrain offers three main functionalities for this:

1. **[Pretraining](#methods-dinov2)**

   Pretraining lets you train a model on unlabeled data with self-supervised learning
   (SSL) methods. This is ideal if you want to train your own vision foundation models
   using a method like {ref}`DINOv2 <methods-dinov2>`.

1. **[Distillation](#methods-distillation)**

   Distillation is a special form of pretraining where a large, pretrained teacher
   model, like {ref}`DINOv2 <models-dinov2>` or {ref}`DINOv3 <models-dinov3>`, is used
   to guide the training of a smaller student model. This is the ideal starting point if
   you want to improve performance of any model that is not already a large vision
   foundation model, like YOLO, ConvNet, or special transformer architectures.

1. **[Autolabeling](#predict-autolabel)**

   Autolabeling lets you generate pseudo-labels for your unlabeled data using a strong
   fine-tuned model. You can then use the pseudo-labeled data to train your own models
   in a supervised way. This is ideal if you already have enough labeled data to train a
   strong autolabeler. Autolabeling is covered in a
   [separate section](#predict-autolabel) of the documentation.

LightlyTrain has a unified interface for pretraining and distillation through the
`pretrain` command. The remainder of this page will focus on how to use this command. If
you are interested in one of the specific methods, please check out the respective
pages:

- [Distillation](#methods-distillation)
- [DINOv2 Pretraining](#methods-dinov2)
- [Other Pretraining Methods](#methods)

If you need help choosing the right method for your use case, check out the
[Methods Comparison](#methods-comparison) page.

## Pretrain

The pretrain command is a simple interface to pretrain or distill a large number of
models using different SSL methods. An example command looks like this:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        # Examples: "edgecrafter/ecvitt", "timm/vits14", CustomModel(), ...
        model="torchvision/resnet50",
        # Examples: "dinov2", "simclr", "dino", ...
        method="distillation",
        epochs=100,
        batch_size=128,
    )
````

````{tab} Command Line
```bash
lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50" method="distillation" epochs=100 batch_size=128
````

```{important}
The default pretraining method {ref}`distillation <methods-distillation>` is
recommended, as it consistently outperforms others in extensive experiments. Batch
sizes between `128` and `1536` strike
a good balance between speed and performance. Moreover, long training runs, such as
2,000 epochs on COCO, significantly improve results. Check the [Methods](#methods-comparison)
page for more details why {ref}`distillation <methods-distillation>` is the best
choice.
```

This will distill a TorchVision ResNet-50 student model using images from `my_data_dir`
and the default {ref}`distillation <methods-distillation>` setup, which uses a
{ref}`DINOv3 <models-dinov3>` teacher by default. All training logs, model exports, and
checkpoints are saved to the output directory at `out/my_experiment`.

```{tip}
See {meth}`lightly_train.pretrain` for the complete Python API and run
`lightly-train pretrain --help` for all command line arguments.
```

(train-output)=

## Out

The `out` argument specifies the output directory where all training logs, model
exports, and checkpoints are saved. It looks like this after training:

```text
out/my_experiment
├── checkpoints
│   ├── epoch=99-step=123.ckpt                          # Intermediate checkpoint
│   └── last.ckpt                                       # Last checkpoint
├── events.out.tfevents.1721899772.host.1839736.0       # TensorBoard logs
├── exported_models
|   └── exported_last.pt                                # Final model exported
├── metrics.jsonl                                       # Training metrics
└── train.log                                           # Training logs
```

The final model checkpoint is saved to `out/my_experiment/checkpoints/last.ckpt`. The
file `out/my_experiment/exported_models/exported_last.pt` contains the final model,
exported in the default format (`package_default`) of the used library (see
{ref}`export format <export-format>` for more details).

```{tip}
Create a new output directory for each experiment to keep training logs, model exports,
and checkpoints organized.
```

(pretrain-data)=

## Data

Lightly**Train** expects a folder containing images or a list of (possibly mixed)
folders and image files. Any folder will be recursively traversed and finds all image
files within it (even in nested subdirectories).

The following image formats are supported:

- jpg
- jpeg
- png
- ppm
- bmp
- pgm
- tif
- tiff
- webp
- dcm (DICOM)

For more details on LightlyTrain's support for data input, please check the
[Data Input](#data-input) page.

Example of passing a single folder `my_data_dir`:

```bash
my_data_dir
├── dir0
│   ├── image0.jpg
│   └── image1.jpg
└── dir1
    └── image0.jpg
```

````{tab} Python
```python skip_ruff
lightly_train.pretrain(
    out="out/my_experiment",            # Output directory
    data="my_data_dir",                 # Directory with images
    model="torchvision/resnet18",       # Model to train
)
```
````

````{tab} Command Line
```bash
lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="torchvision/resnet18"
```
````

Example of passing a (mixed) list of files and folders:

```bash
├── image2.jpg
├── image3.jpg
└── my_data_dir
    ├── dir0
    │   ├── image0.jpg
    │   └── image1.jpg
    └── dir1
        └── image0.jpg
```

````{tab} Python
```python skip_ruff
lightly_train.pretrain(
    out="out/my_experiment",            # Output directory
    data=["image2.jpg", "image3.jpg", "my_data_dir"],                 # Directory with images
    model="torchvision/resnet18",       # Model to train
)
```
````

````{tab} Command Line
```bash
lightly-train pretrain out="out/my_experiment" data='["image2.jpg", "image3.jpg", "my_data_dir"]' model="torchvision/resnet18"
```
````

## Model

See [supported libraries](#models-supported-libraries) in the Models page for a detailed
list of all supported libraries and their respective docs pages for all supported
models. You can also pass a pre-instantiated wrapper for a
[custom model](#custom-models) when using Python.

## Method

The `method` argument selects the self-supervised learning recipe, not the model
library. For example, `method="dinov2"` trains supported {ref}`DINOv2 <methods-dinov2>`
ViTs with the DINOv2 recipe, while `method="distillation"` trains a student model from a
teacher and is the recommended default. {ref}`DINOv3 <models-dinov3>` models are
currently used through {ref}`distillation <methods-distillation>` or as fine-tuning
backbones, not through a standalone `method="dinov3"` recipe. See [Methods](#methods)
for a list of all supported methods.

(logging)=

## Loggers

Logging is configured with the `loggers` argument passed to `lightly_train.pretrain()`
or the `lightly-train pretrain` command. In Python, pass a nested dictionary such as
`loggers={"jsonl": {"flush_logs_every_n_steps": 10}}`. On the command line, use dot
notation such as `loggers.jsonl.flush_logs_every_n_steps=10`. The following loggers are
supported:

- [`jsonl`](#jsonl): Logs training metrics to a .jsonl file (enabled by default)
- [`mlflow`](#mlflow): Logs training metrics to MLflow (disabled by default, requires
  MLflow to be installed)
- [`tensorboard`](#tensorboard): Logs training metrics to TensorBoard (enabled by
  default, requires TensorBoard to be installed)
- [`wandb`](#wandb): Logs training metrics to Weights & Biases (disabled by default,
  requires Weights & Biases to be installed)

(jsonl)=

### JSONL

The JSONL logger is enabled by default and logs training metrics to a .jsonl file at
`out/my_experiment/metrics.jsonl`. Configure its arguments under the `jsonl` logger key:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        model="torchvision/resnet50",
        loggers={"jsonl": {"flush_logs_every_n_steps": 10}},
    )
````

````{tab} Command Line
```bash
lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50" loggers.jsonl.flush_logs_every_n_steps=10
````

Disable the JSONL logger with:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        model="torchvision/resnet50",
        loggers={"jsonl": None},
    )
````

````{tab} Command Line
```bash
lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50" loggers.jsonl=null
````

(mlflow)=

### MLflow

```{important}
MLflow must be installed with `pip install "lightly-train[mlflow]"`.
```

Configure MLflow under the `mlflow` logger key:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        model="torchvision/resnet50",
        loggers={
            "mlflow": {
                "experiment_name": "my_experiment",
                "run_name": "my_run",
                "tracking_uri": "tracking_uri",
                # "run_id": "my_run_id",  # Use if resuming a training with resume_interrupted=True
                # "log_model": True,      # Currently not supported
            },
        },
    )
````

````{tab} Command Line
```bash
lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50" loggers.mlflow.experiment_name="my_experiment" loggers.mlflow.run_name="my_run" loggers.mlflow.tracking_uri=tracking_uri
````

(tensorboard)=

### TensorBoard

TensorBoard logs are automatically saved to the output directory. Configure TensorBoard
under the `tensorboard` logger key:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        model="torchvision/resnet50",
        loggers={"tensorboard": {"name": "my_experiment"}},
    )
````

````{tab} Command Line
```bash
lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50" loggers.tensorboard.name="my_experiment"
````

Run TensorBoard in a new terminal to visualize the training progress:

```bash
tensorboard --logdir out/my_experiment
```

Disable the TensorBoard logger with:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        model="torchvision/resnet50",
        loggers={"tensorboard": None},
    )
````

````{tab} Command Line
```bash
lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50" loggers.tensorboard=null
````

(wandb)=

### Weights & Biases

```{important}
Weights & Biases must be installed with `pip install "lightly-train[wandb]"`.
```

Configure Weights & Biases under the `wandb` logger key:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        model="torchvision/resnet50",
        loggers={
            "wandb": {
                "project": "my_project",
                "name": "my_experiment",
                "log_model": False,              # Set to True to upload model checkpoints
            },
        },
    )
````

````{tab} Command Line
```bash
lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50" loggers.wandb.project="my_project" loggers.wandb.name="my_experiment" loggers.wandb.log_model=False
````

More configuration options are available through the Weights & Biases environment
variables. See the
[Weights & Biases documentation](https://docs.wandb.ai/guides/track/environment-variables/)
for more information.

Disable the Weights & Biases logger with:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        model="torchvision/resnet50",
        loggers={"wandb": None},
    )
````

````{tab} Command Line
```bash
lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50" loggers.wandb=null
````

## Resume Training

There are two distinct ways to continue training, depending on your intention.

### Resume Interrupted Training

Use `resume_interrupted=True` to **resume a previously interrupted or crashed training
run**. This will pick up exactly where the training left off.

- You **must use the same `out` directory** as the original run.
- You **must not change any training parameters** (e.g., learning rate, batch size,
  data, etc.).
- This is intended for continuing the *same* run without modification.

### Load Weights for a New Run

Use `checkpoint` to further pretrain a model from a previous run. The checkpoint must be
a path to a checkpoint file created by a previous training run, for example
`checkpoint="out/my_experiment/checkpoints/last.ckpt"`. This will only load the model
weights from the previous run. All other training state (e.g. optimizer state, epochs)
from the previous run are not loaded. Instead, a new run is started with the model
weights from the checkpoint.

- You are free to **change training parameters**.
- This is useful for continuing training with a different setup.

### General Notes

```{important}
- `resume_interrupted=True` and `checkpoint=...` are mutually exclusive and cannot be
  used together.
- If `overwrite=True` is set, training will start fresh, overwriting existing outputs or
  checkpoints in the specified output directory.
```

## Advanced Options

### Input Image Resolution

The input image resolution can be set with the transform_args argument. By default a
resolution of 224x224 pixels is used. A custom resolution can be set like this:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",            # Output directory
        data="my_data_dir",                 # Directory with images
        model="torchvision/resnet18",       # Model to train
        transform_args={"image_size": (448, 448)}, # (height, width)
    )
```
````

````{tab} Command Line
```bash
lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="torchvision/resnet18" transform_args.image_size="[448,448]"
```
````

```{warning}
Not all models support all image sizes.
```

### Image Transforms

See {ref}`method-transform-args` on how to configure image transformations.

(method-args)=

### Method Arguments

```{warning}
In 99% of cases, it is not necessary to modify the default method arguments in
LightlyTrain. The default settings are carefully tuned to work well for most use cases.
```

The method arguments can be set with the `method_args` argument:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",            # Output directory
        data="my_data_dir",                 # Directory with images
        model="torchvision/resnet18",       # Model to train
        method="distillation",              # Pretraining method
        method_args={                       # Override the default teacher model (default: dinov3/vitb16)
            "teacher": "dinov3/vitl16",
        },
    )
```
````

````{tab} Command Line
```bash
lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="torchvision/resnet18" method="distillation" method_args.teacher="dinov3/vitl16"
```
````

```{tip}
The default teacher for `distillation` / `distillationv3` is `dinov3/vitb16`. See
{ref}`methods-distillation-default-teacher` for the per-version default table
and the full list of supported teachers. Use a {ref}`DINOv2 <models-dinov2>` teacher
(e.g. `dinov2/vitl14`) for a permissive Apache 2.0 license.
```

Each pretraining method has its own set of arguments that can be configured.
LightlyTrain provides sensible defaults that are adjusted depending on the dataset and
model used. The defaults for each method are listed in the respective {ref}`methods`
documentation pages.

### Performance Optimizations

For performance optimizations, e.g. using accelerators, multi-GPU, multi-node, and half
precision training, see the [performance](#performance) page.

```{toctree}
---
hidden:
maxdepth: 1
---
Overview <self>
Distillation <methods/distillation>
Pretrain DINOv2 <methods/dinov2>
All Methods <methods/index>
models/index
export
```
