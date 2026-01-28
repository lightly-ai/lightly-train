(pretrain-settings)=

# Pretrain/Distill Settings

This page covers the settings available for self-supervised pretraining and distillation
in LightlyTrain. For task-specific fine-tuning options, see the [](train-settings) page.

| Name                                                    | Type                          | Default        | Description                                                                                     |
| ------------------------------------------------------- | ----------------------------- | -------------- | ----------------------------------------------------------------------------------------------- |
| [`out`](#out)                                           | `str`<br>`Path`               | —              | Output directory where checkpoints, logs, and exported models are written.                      |
| [`data`](#data)                                         | `str`<br>`Path`<br>`list`     | —              | Path or list of paths pointing to training data.                                                |
| [`model`](#model)                                       | `str`<br>`Path`<br>`Module`   | —              | Model identifier (e.g. "dinov2/vits14") or a custom PyTorch module to wrap.                     |
| [`method`](#method)                                     | `str`                         | "distillation" | Self-supervised method to run (e.g. "dino", "msn").                                             |
| [`method_args`](#method_args)                           | `dict`                        | `None`         | Method-specific hyperparameters.                                                                |
| [`embed_dim`](#embed_dim)                               | `int`                         | `None`         | Optional embedding dimensionality override.                                                     |
| [`epochs`](#epochs)                                     | `int`                         | "auto"         | Number of training epochs. "auto" derives a value from dataset size and batch size.             |
| [`batch_size`](#batch_size)                             | `int`                         | `128`          | Global batch size across all devices.                                                           |
| [`num_workers`](#num_workers)                           | `int`                         | "auto"         | DataLoader worker processes per device. "auto" chooses a value based on available CPU cores.    |
| [`devices`](#devices)                                   | `int`<br>`str`<br>`list[int]` | "auto"         | Devices to use for training. "auto" selects all available devices for the chosen `accelerator`. |
| [`num_nodes`](#num_nodes)                               | `int`                         | `1`            | Number of nodes for distributed training.                                                       |
| [`resume_interrupted`](#resume_interrupted)             | `bool`                        | `False`        | Resume an interrupted run from the same `out` directory, including optimizer state and epoch.   |
| [`checkpoint`](#checkpoint)                             | `str`<br>`Path`               | `None`         | Path to a checkpoint to initialize weights from before starting a new run.                      |
| [`overwrite`](#overwrite)                               | `bool`                        | `False`        | If `True`, overwrite the `out` directory if it already exists.                                  |
| [`accelerator`](#accelerator)                           | `str`                         | "auto"         | Hardware backend: "cpu", "gpu", "mps", or "auto" to pick the best available.                    |
| [`strategy`](#strategy)                                 | `str`                         | "auto"         | Distributed training strategy (e.g. "ddp"). "auto" selects a suitable default.                  |
| [`precision`](#precision)                               | `str`                         | "auto"         | Numeric precision mode (e.g. "bf16-mixed", "16-mixed").                                         |
| [`float32_matmul_precision`](#float32_matmul_precision) | `str`                         | "auto"         | Precision for float32 matrix multiplication.                                                    |
| [`seed`](#seed)                                         | `int`<br>`None`               | `0`            | Random seed for reproducibility. Set to `None` to disable seeding.                              |
| [`loggers`](#loggers)                                   | `dict`                        | `None`         | Logger configuration dict. `None` uses defaults; keys configure or disable individual loggers.  |
| [`callbacks`](#callbacks)                               | `dict`                        | `None`         | Callback configuration dict. `None` enables the recommended defaults.                           |
| [`optim`](#optim)                                       | `str`                         | "auto"         | Optimizer selection (`"auto"`, "adamw", "sgd").                                                 |
| [`optim_args`](#optim_args)                             | `dict`                        | `None`         | Overrides for optimizer hyperparameters.                                                        |
| [`transform_args`](#transform_args)                     | `dict`                        | `None`         | Data transform configuration (e.g. image size, normalization).                                  |
| [`loader_args`](#loader_args)                           | `dict`                        | `None`         | Advanced DataLoader keyword arguments.                                                          |
| [`trainer_args`](#trainer_args)                         | `dict`                        | `None`         | Additional Lightning Trainer keyword arguments.                                                 |
| [`model_args`](#model_args)                             | `dict`                        | `None`         | Arguments forwarded to model construction.                                                      |
| [`resume`](#resume)                                     | `bool`                        | `None`         | Deprecated. Use [`resume_interrupted`](#resume_interrupted) instead.                            |

```{tip}
LightlyTrain automatically selects suitable default values based on the chosen
model, method, dataset, and hardware. You only need to set parameters that you
want to customize.

Look for the `Resolved configuration` dictionary in the `train.log` file in the output
directory of your run to see the final settings that were applied. This will
include any overrides, automatically resolved values, and method-specific
settings that are not listed on this page.
```

(pretrain-settings-output)=

## Output

### `out`

The output directory where checkpoints and logs are saved. Create a new directory for
each run. LightlyTrain raises an error if the directory already exists unless
[`overwrite`](#overwrite) is `True`.

### `overwrite`

Set to `True` to overwrite the contents of an existing `out` directory. By default,
LightlyTrain raises an error if the specified output directory already exists to prevent
accidental data loss.

(pretrain-settings-data)=

## Data

### `data`

Provide either a path to a single directory or a list of directories or image filenames
containing training images. LightlyTrain indexes the files and builds a dataset with
automatic augmentation defaults for the selected method.

### `batch_size`

Global batch size across all devices. The per-device batch size is computed as
`batch_size / (num_devices * num_nodes)`. Adjust this value to fit your memory budget.
We recommend values between `128` and `2048`. The default is `128`.

### `num_workers`

Number of background worker processes per device used by the PyTorch DataLoader. By
default, this is set to `"auto"`, which selects a value based on the number of available
CPU cores.

### `loader_args`

Advanced keyword arguments passed directly to `torch.utils.data.DataLoader`. Avoid using
this unless you need a PyTorch feature that is not exposed through other settings.

(pretrain-settings-model)=

## Model

### `model`

Model to pretrain. Can be a model identifier string (for example "dinov2/vits14") or a
custom PyTorch module. See [](models) for all supported models and libraries. See
[](custom-models) for instructions on using custom models.

### `model_args`

Dictionary with model-specific arguments. The available keys depend on the selected
model family. Arguments are forwarded to the model constructor. For torchvision models,
the arguments are forwarded to `torchvision.models.get_model`. For Ultralytics models,
the arguments are forwarded to `ultralytics.YOLO` etc. This argument is rarely needed.

### `embed_dim`

Override the default embedding dimensionality of the model. This is useful when you need
a specific embedding size for downstream tasks. This option will add a linear layer on
top of the model. By default, LightlyTrain uses the model's native embedding size
without additional layers.

### `checkpoint`

Path to a checkpoint file (.ckpt or .pt) used to initialize model weights before
starting a new run. The optimizer state is not restored when using this option. See
[`resume_interrupted`](#resume_interrupted) for recovering from crashes.

(pretrain-settings-method)=

## Method

### `method`

Name of the self-supervised method to run. The default "distillation" selects Lightly's
distillation recipe. Other examples include "dino" and "msn".

### `method_args`

Dictionary with method-specific hyperparameters. LightlyTrain validates these arguments
against the chosen method class and fills in sensible defaults.

```python
import lightly_train

lightly_train.pretrain(
	...,
	method="dino",
	method_args={
		"teacher_momentum": 0.996,
		"student_temp": 0.1,
	},
)
```

(pretrain-settings-training-loop)=

## Training Loop

### `epochs`

Total number of training epochs. The default "auto" derives an epoch count from the
dataset size and batch size to reach method-specific target steps.

### `precision`

Training precision setting. LightlyTrain resolves "auto" to a safe default for your
hardware. Supported explicit values include:

- "bf16-mixed": Runs computations in bfloat16 where supported while storing weights in
  float32.
- "16-mixed" and "16-true": Run in float16. Not all accelerators support these modes.
- "32-true": Pure float32 training. Slower but maximally stable.

### `float32_matmul_precision`

Controls PyTorch's float32 matmul precision context. Choose among "auto", "highest",
"high", or "medium". Keep it at "auto" unless you observe numerical instability or want
to trade precision for speed.

### `seed`

Seed used for data order, augmentation randomness, and initialization. Set to `None` for
nondeterministic runs.

(pretrain-settings-hardware)=

## Hardware

### `devices`

Number of devices (CPUs/GPUs) to use. Accepts an integer, a comma-separated string such
as "0,1", or an explicit list `[0, 1]`.

### `accelerator`

Hardware backend to target. Valid strings: "cpu", "gpu", "mps", and "auto". The default
selects the best available backend for your machine.

### `num_nodes`

Number of nodes for distributed training. Keep this at `1` unless you orchestrate
multi-node training.

### `strategy`

Distributed training strategy. The default "auto" picks a strategy aligned with your
accelerator and device choice (e.g. DDP for multi-GPU runs).

(pretrain-settings-resume)=

## Resume Training

There are two ways to continue from previous work:

1. [Resume an interrupted run](#resume_interrupted) to continue with identical
   parameters.
1. [Load a checkpoint for a new run](#checkpoint) to start fresh with different
   settings.

### `resume_interrupted`

Use when a run stops unexpectedly and you want to continue from
`out/checkpoints/last.ckpt`. Do not change any parameters, including batch size, method,
or transforms. LightlyTrain restores the optimizer, scheduler, and epoch state to match
the interrupted run.

```python
import lightly_train

lightly_train.pretrain(
	out="out/my_experiment",
	data="/data/images",
	model="dinov2/vits14",
	resume_interrupted=True,
)
```

### `checkpoint`

Load model weights from a previous run while starting a fresh training session. Specify
a new `out` directory and adjust parameters as needed. Only the model weights are
restored; optimizer state and schedulers restart from scratch.

```python
import lightly_train

lightly_train.pretrain(
	out="out/longer_run",
	data="/data/images",
	model="dinov2/vits14",
	checkpoint="out/my_experiment/exported_models/exported_last.pt",
	epochs=400,
)
```

### `resume`

Deprecated alias for [`resume_interrupted`](#resume_interrupted). It remains for
backward compatibility but emits a warning.

(pretrain-settings-optimizer)=

## Optimizer

### `optim`

Optimizer choice. "auto" selects a default that matches the chosen method. Set to
"adamw" or "sgd" to force a specific optimizer.

### `optim_args`

Dictionary overriding optimizer hyperparameters. Supported keys depend on the selected
optimizer. For example:

```python
import lightly_train

lightly_train.pretrain(
	...,
	optim="adamw",
	optim_args={
		"lr": 3e-4,
		"betas": (0.9, 0.95),
		"weight_decay": 0.04,
	},
)
```

(pretrain-settings-logging)=

## Logging

### `loggers`

Dictionary configuring training loggers. By default, TensorBoard logging is enabled.
Supported keys include `"tensorboard"`, `"wandb"`, and `"mlflow"`. Set a key to `None`
to disable the corresponding logger.

```python
import lightly_train

lightly_train.pretrain(
	...,
	loggers={
		"tensorboard": None,
		"wandb": {
			"project": "ssl-research",
			"name": "dino-run",
			"log_model": True,
		},
	},
)
```

Refer to the Lightning logger documentation for the full list of supported options.

(pretrain-settings-callbacks)=

## Callbacks

### `callbacks`

Dictionary configuring Lightning callbacks. LightlyTrain enables a sensible set of
defaults (for example checkpointing and learning rate logging). Add entries to customize
or disable callbacks.

```python
import lightly_train

lightly_train.pretrain(
	...,
	callbacks={
		"model_checkpoint": {
			"every_n_epochs": 5,
		},
	},
)
```

(pretrain-settings-trainer)=

## Trainer

### `trainer_args`

Advanced keyword arguments passed directly to `lightning.pytorch.Trainer`. Use this for
features not exposed through dedicated LightlyTrain settings, such as gradient clipping
or custom limits.

(pretrain-settings-transforms)=

## Transforms

### `transform_args`

Dictionary to customize data transforms. LightlyTrain fills in method-specific defaults.
Common options include `image_size`, `normalize`, and augmentation probabilities.
Provide nested dictionaries matching the method's schema.

```python
import lightly_train

lightly_train.pretrain(
	...,
	transform_args={
		"image_size": (224, 224),
		"normalize": {
			"mean": [0.485, 0.456, 0.406],
			"std": [0.229, 0.224, 0.225],
		},
	},
)
```

```{toctree}
---
hidden:
maxdepth: 1
---
self
train_settings
```
