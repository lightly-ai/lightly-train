(pretrain-settings)=

# Pretrain & Distill Settings

This page covers the settings available for self-supervised pretraining and distillation
in LightlyTrain. For task-specific fine-tuning options, see the [](train-settings) page.

| Name                                                    | Type                          | Default          | Description                                                                                       |
| ------------------------------------------------------- | ----------------------------- | ---------------- | ------------------------------------------------------------------------------------------------- |
| [`out`](#out)                                           | `str`<br>`Path`               | —                | Output directory where checkpoints, logs, and exported models are written.                        |
| [`data`](#data)                                         | `str`<br>`Path`<br>`list`     | —                | Path or list of paths to directory with training images.                                          |
| [`model`](#model)                                       | `str`<br>`Path`<br>`Module`   | —                | Model identifier (e.g. "dinov2/vits14") or a custom PyTorch module to wrap.                       |
| [`method`](#method)                                     | `str`                         | `"distillation"` | Self-supervised method to run (e.g. "dino", "msn").                                               |
| [`method_args`](#method_args)                           | `dict`                        | `None`           | Method-specific hyperparameters.                                                                  |
| [`embed_dim`](#embed_dim)                               | `int`                         | `None`           | Optional embedding dimensionality override.                                                       |
| [`epochs`](#epochs)                                     | `int`                         | `auto`           | Number of training epochs. `auto` derives a value from dataset size and batch size.               |
| [`batch_size`](#batch_size)                             | `int`                         | `128`            | Global batch size across all devices.                                                             |
| [`num_workers`](#num_workers)                           | `int`                         | `"auto"`         | DataLoader worker processes per device. `"auto"` chooses a value based on available CPU cores.    |
| [`devices`](#devices)                                   | `int`<br>`str`<br>`list[int]` | `"auto"`         | Devices to use for training. `"auto"` selects all available devices for the chosen `accelerator`. |
| [`num_nodes`](#num_nodes)                               | `int`                         | `1`              | Number of nodes for distributed training.                                                         |
| [`resume_interrupted`](#resume_interrupted)             | `bool`                        | `False`          | Resume an interrupted run from the same `out` directory, including optimizer state and epoch.     |
| [`checkpoint`](#checkpoint)                             | `str`<br>`Path`               | `None`           | Path to a checkpoint to initialize weights from before starting a new run.                        |
| [`overwrite`](#overwrite)                               | `bool`                        | `False`          | If `True`, overwrite the `out` directory if it already exists.                                    |
| [`accelerator`](#accelerator)                           | `str`                         | `"auto"`         | Hardware backend: `"cpu"`, `"gpu"`, `"mps"`, or `"auto"` to pick the best available.              |
| [`strategy`](#strategy)                                 | `str`                         | `"auto"`         | Distributed training strategy (e.g. `ddp`). `"auto"` selects a suitable default.                  |
| [`precision`](#precision)                               | `str`                         | `"auto"`         | Numeric precision mode (e.g. `bf16-mixed`, `16-mixed`).                                           |
| [`float32_matmul_precision`](#float32_matmul_precision) | `str`                         | `"auto"`         | Precision for float32 matrix multiplication.                                                      |
| [`seed`](#seed)                                         | `int`<br>`None`               | `0`              | Random seed for reproducibility. Set to `None` to disable seeding.                                |
| [`loggers`](#loggers)                                   | `dict`                        | `None`           | Logger configuration dict. `None` uses defaults; keys configure or disable individual loggers.    |
| [`callbacks`](#callbacks)                               | `dict`                        | `None`           | Callback configuration dict. `None` enables the recommended defaults.                             |
| [`optim`](#optim)                                       | `str`                         | `"auto"`         | Optimizer selection (`"auto"`, `"adamw"`, `"lars"`, `"sgd"`).                                     |
| [`optim_args`](#optim_args)                             | `dict`                        | `None`           | Overrides for optimizer hyperparameters.                                                          |
| [`transform_args`](#transform_args)                     | `dict`                        | `None`           | Data transform configuration (e.g. image size, normalization).                                    |
| [`loader_args`](#loader_args)                           | `dict`                        | `None`           | Advanced DataLoader keyword arguments.                                                            |
| [`trainer_args`](#trainer_args)                         | `dict`                        | `None`           | Additional Lightning Trainer keyword arguments.                                                   |
| [`model_args`](#model_args)                             | `dict`                        | `None`           | Arguments forwarded to model construction.                                                        |

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
automatic augmentation defaults for the selected method. See the [](data-input) page for
details on supported image formats.

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

Model to pretrain. Can be a model identifier string (for example `"dinov2/vits14"`) or a
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

(pretrain-settings-method)=

## Method

### `method`

Name of the self-supervised method to run. The default `"distillation"` selects
Lightly's distillation recipe. See [](methods) for the full list of supported methods.

### `method_args`

Dictionary with method-specific hyperparameters. LightlyTrain chooses sensible defaults
for each method, so you only need to set this if you want to customize specific
settings.

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

Total number of training epochs. We recommend the following epoch settings:

- Small datasets (\<100k images): 200-2000 epochs
- Medium datasets (100k-1M images): 100-500 epochs
- Large datasets (>10M images): 5-100 epochs

For DINOv2, keep the default value of `"auto"`, which selects a suitable number of
epochs based on the dataset size.

### `precision`

Training precision setting. Must be one of the following strings:

- `"bf16-mixed"`: Default. Operations run in bfloat16 where supported, weights are saved
  in float32. Not supported on all hardware.
- `"16-true"`: All operations and weights in float16. Fastest but may be unstable
  depending on model, hardware, and dataset.
- `"16-mixed"`: Most operations run in float16 precision. Not supported on all hardware.
- `"32-true"`: All operations and weights are in float32. Slower but more stable.
  Supported on all hardware.

### `float32_matmul_precision`

Controls PyTorch's float32 matmul precision context. Choose among `"auto"`, `"highest"`,
`"high"`, or `"medium"`. Keep it at `"auto"` unless you observe numerical instability or
want to trade precision for speed.

### `seed`

Controls reproducibility for data order, augmentation randomness, and initialization.
Set to `None` to use a random seed on each run. Default is `0`.

(pretrain-settings-hardware)=

## Hardware

### `devices`

Number of devices (CPUs/GPUs) to use for training. Accepts an integer (number of
devices), an explicit list of device indices, or a string with device ids such as
`"1,2,3"`.

### `accelerator`

Type of hardware accelerator to use. Valid options are `"cpu"`, `"gpu"`, `"mps"`, or
`"auto"`. `"auto"` selects the best available accelerator on the system.

### `num_nodes`

Number of nodes for distributed training. By default a single node is used. We recommend
to keep this at `1`.

### `strategy`

Distributed training strategy, for example `"ddp"` or `"fsdp"`. By default, this is set
to `"auto"`, which selects a suitable strategy based on the chosen accelerator and
number of devices. We recommend to keep this at `"auto"` unless you have specific
requirements.

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

(pretrain-settings-optimizer)=

## Optimizer

### `optim`

Optimizer choice. `"auto"` selects a default that matches the chosen method. Set to
`"adamw"`, `"lars"`, or `"sgd"` to force a specific optimizer. We recommend to keep this
at `"auto"` unless you have specific requirements.

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

Dictionary to configure logging behavior. By default, LightlyTrain uses the built-in
TensorBoard logger. You can customize logging frequency and enable/disable additional
loggers like MLflow and Weights & Biases.

| Key                           | Type             | Description                                                 |
| ----------------------------- | ---------------- | ----------------------------------------------------------- |
| [`mlflow`](#mlflow)           | `dict`<br>`None` | MLflow logger configuration. Disabled by default.           |
| [`wandb`](#wandb)             | `dict`<br>`None` | Weights & Biases logger configuration. Disabled by default. |
| [`tensorboard`](#tensorboard) | `dict`<br>`None` | TensorBoard logger configuration. Set to `None` to disable. |

#### `mlflow`

MLFlow logger configuration. It is disabled by default. Requires MLFlow to be installed
with:

```bash
pip install "lightly-train[mlflow]"
```

```python
import lightly_train

lightly_train.pretrain(
	...,
	loggers={
		"mlflow": {
			# Optional experiment name.
			"experiment_name": "my_experiment",
			# Optional custom run name.
			"run_name": "my_run",
			# Optional tags dictionary.
			"tags": {"team": "research"},
			# Optional address of local or remote tracking server, e.g. "http://localhost:5000"
			"tracking_uri": "tracking_uri",
			# Enable checkpoint uploading to MLflow. (default: False)
			"log_model": True,
			# Optional string to put at the beginning of metric keys.
			"prefix": "",
			# Optional location where artifacts are stored.
			"artifact_location": "./mlruns",
		},
	},
)
```

See the
[PyTorch Lightning MLflow Logger documentation](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.MLFlowLogger.html#mlflowlogger)
for details on the available configuration options.

#### `wandb`

Weights & Biases logger configuration. It is disabled by default. Requires Weights &
Biases to be installed with:

```bash
pip install "lightly-train[wandb]"
```

```python
import lightly_train

lightly_train.pretrain(
	...,
	loggers={
		"wandb": {
			# Optional display name for the run.
			"name": "my_run",
			# Optional project name.
			"project": "my_project",
			# Optional version, mainly used to resume a previous run.
			"version": "my_version",
			# Optional, upload model checkpoints as artifacts. (default: False)
			"log_model": False,
			# Optional name for uploaded checkpoints. (default: None)
			"checkpoint_name": "checkpoint.ckpt",
			# Optional, run offline without syncing to the W&B server. (default: False)
			"offline": False,
			# Optional, configure anonymous logging. (default: False)
			"anonymous": False,
			# Optional string to put at the beginning of metric keys.
			"prefix": "",

		},
	},
)
```

See the
[PyTorch Lightning Weights & Biases Logger documentation](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html#wandblogger)
for details on the available configuration options.

#### `tensorboard`

Configuration for the built-in TensorBoard logger. TensorBoard logs are by default
enabled and automatically saved to the output directory. Run TensorBoard in a new
terminal to visualize the training progress:

```bash
tensorboard --logdir out/my_experiment
```

Disable TensorBoard logging by setting this argument to `None`:

```python
import lightly_train

lightly_train.train_object_detection(
	...,
	loggers={
		"tensorboard": None,  # Disable TensorBoard logging.
	},
)
```

(pretrain-settings-checkpoint-saving)=

## Checkpoint Saving

LightlyTrain saves two types of checkpoints during training:

1. `out/my_experiment/checkpoints`: Full checkpoints including optimizer, scheduler, and
   training state. Used to resume training with
   [`resume_interrupted`](#resume_interrupted).
   - `last.ckpt`: Latest checkpoint saved at regular intervals.
   - `best.ckpt`: Best-performing checkpoint based on a validation metric.
1. `out/my_experiment/exported_models`: Lightweight exported models containing only
   model weights. Used for further fine-tuning.
   - `exported_last.pt`: Model weights from the latest checkpoint.

Use the exported models in `out/my_experiment/exported_models/` for any downstream tasks
whenever training completed successfully. Use `out/my_experiment/checkpoints/` only to
resume training with [`resume_interrupted`](#resume_interrupted) after an unexpected
interruption.

Checkpoint saving is configured through the [`callbacks`](#callbacks) setting.

### `callbacks`

Dictionary configuring Lightning callbacks. LightlyTrain enables a sensible set of
defaults (for example checkpointing and learning rate logging). Add entries to customize
or disable callbacks.

| Key                                     | Type             | Description                  |
| --------------------------------------- | ---------------- | ---------------------------- |
| [`model_export`](#model_export)         | `dict`<br>`None` | Configure model exporting.   |
| [`model_checkpoint`](#model_checkpoint) | `dict`<br>`None` | Configure checkpoint saving. |

#### `model_export`

Exports the wrapped model package to `out/exported_models/exported_last.pt` on a fixed
epoch cadence. Enabled by default with `every_n_epochs=1`. Set this entry to `None` to
disable automatic exports. Reducing export frequency can speed up training.

```python
import lightly_train

lightly_train.pretrain(
	...,
	callbacks={
		"model_export": {
			# Export after every third epoch instead of every epoch.
			"every_n_epochs": 3,
		},
	},
)
```

#### `model_checkpoint`

Persists training checkpoints, including optimizer, scheduler, and training state for
resuming runs and exporting. Enabled by default with `every_n_epochs=1`. Set this entry
to `None` to disable automatic checkpointing. Reducing checkpoint frequency can speed up
training.

```python
from datetime import timedelta

import lightly_train

lightly_train.pretrain(
	...,
	callbacks={
		"model_checkpoint": {
			# Save every N epochs when not using `every_n_train_steps`.
			"every_n_epochs": 1,
			# Save every fixed number of training steps instead of epochs.
			"every_n_train_steps": 1000,
			# Combine with steps/epochs to enforce a minimum wall-clock interval.
			"train_time_interval": timedelta(minutes=30),
			# Choose whether to checkpoint at the end of the training epoch.
			"save_on_train_epoch_end": True,
			# Optional custom directory; defaults to `out/checkpoints` when omitted.
			"dirpath": "out/custom_checkpoints",
			# Optional filename pattern using Lightning tokens such as "{epoch}".
			"filename": "epoch{epoch}-step{step}",
			# Track a metric and keep the best checkpoints.
			"monitor": "val_loss",
			# Log Lightning's checkpointing messages.
			"verbose": True,
			# Keep a copy of the final checkpoint alongside the best ones.
			"save_last": True,
			# Number of top-performing checkpoints to retain.
			"save_top_k": 3,
			# Store full checkpoints including optimizer and scheduler state.
			"save_weights_only": False,
			# Direction for comparing monitored metrics.
			"mode": "min",
			# Include the monitored metric name in checkpoint filenames.
			"auto_insert_metric_name": True,
		},
	},
)
```

(pretrain-settings-trainer)=

## Trainer

### `trainer_args`

Advanced keyword arguments passed directly to
[`lightning.pytorch.Trainer`](https://lightning.ai/docs/pytorch/stable/common/trainer.html).
Use this for features not exposed through dedicated LightlyTrain settings.

(pretrain-settings-transforms)=

(method-transform-args)=

## Transforms

LightlyTrain automatically applies suitable data augmentations and preprocessing steps
for each model and task. The default transforms are designed to work well in most
scenarios. You can customize transform parameters via the
[`transform_args`](#transform_args) setting.

```{note}
In most cases, it is not necessary to modify the default image transforms in
LightlyTrain. The default settings are carefully tuned to work well for most use
cases. However, for specific downstream tasks or unique image domains, you might want
to override these defaults as shown below.
```

### `transform_args`

Dictionary to configure data transforms applied during pretraining. The most commonly
customized parameters are listed in the table below:

| Key                                       | Type              | Description                                             |
| ----------------------------------------- | ----------------- | ------------------------------------------------------- |
| [`image_size`](#image_size)               | `tuple[int, int]` | Image height and width after cropping and resize.       |
| [`normalize`](#normalize)                 | `dict`            | Mean and standard deviation used for normalization.     |
| [`channel_drop`](#channel_drop)           | `dict`            | Randomly drops channels from the image.                 |
| [`random_resize`](#random_resize)         | `dict`            | Random cropping and resizing configuration.             |
| [`random_flip`](#random_flip)             | `dict`            | Horizontal or vertical flip probabilities.              |
| [`random_rotation`](#random_rotation)     | `dict`            | Rotation probability and angle configuration.           |
| [`color_jitter`](#color_jitter)           | `dict`            | Strength of color jitter augmentation.                  |
| [`random_gray_scale`](#random_gray_scale) | `float`           | Probability of converting images to grayscale.          |
| [`gaussian_blur`](#gaussian_blur)         | `dict`            | Gaussian blur probability and kernel configuration.     |
| [`solarize`](#solarize)                   | `dict`            | Solarization probability and threshold.                 |
| [`global_view_1`](#global_view_1)         | `dict`            | DINO-specific modifications for the second global view. |
| [`local_view`](#local_view)               | `dict`            | DINO-specific configuration for additional local views. |

Check the method pages to see the default transforms applied for each method:

- [](methods-distillation)
- [](methods-dinov2)
- [](methods-dino)
- [](methods-simclr)

#### `image_size`

Tuple specifying the height and width of input images after cropping and resizing. The
default size is 224x224.

```python
import lightly_train

lightly_train.pretrain(
	...,
	transform_args={
		"image_size": (128, 128),  # Resize images to 128x128
	},
)
```

#### `normalize`

Dictionary specifying the mean and standard deviation used for input normalization.
ImageNet statistics are used by default. Change these values when working with datasets
that have different color distributions.

```python
import lightly_train

lightly_train.pretrain(
    ...,
    transform_args={
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
)
```

#### `random_resize`

Random cropping and resizing. Can be disabled by setting to `None`.

```python
import lightly_train

lightly_train.pretrain(
	...,
	transform_args={
		"random_resize": {
			"min_scale": 0.2,
			"max_scale": 1.0,
		},
	},
)
```

#### `random_flip`

Dictionary to configure random flipping augmentation. By default, horizontal flipping is
applied with a probability of `0.5` and vertical flipping is disabled. Adjust the
probabilities as needed.

```python
import lightly_train

lightly_train.pretrain(
    ...,
    transform_args={
        "random_flip": {
            "horizontal_prob": 0.7,  # 70% chance to flip horizontally
            "vertical_prob": 0.2,    # 20% chance to flip vertically
        },
    },
)
```

#### `random_rotation`

Dictionary to configure random rotation augmentation. By default, rotation is disabled.
Specify the maximum rotation angle and probability to enable.

```python
import lightly_train

lightly_train.pretrain(
    ...,
    transform_args={
        "random_rotation": {
            "prob": 0.5,     # 50% chance to apply rotation
            "degrees": 30,  # Maximum rotation angle of 30 degrees
        },
    },
)
```

#### `color_jitter`

Dictionary to configure color jitter augmentation. By default, color jitter is disabled.
Not all models support color jitter augmentation.

```python
import lightly_train

lightly_train.pretrain(
    ...,
    transform_args={
        "color_jitter": {
            "prob": 0.8,  # 80% chance to apply color jitter
            "strength": 2.0,  # Strength of color jitter. Multiplied with the individual parameters below.
            "brightness": 0.4,  
            "contrast": 0.4,    
            "saturation": 0.4,  
            "hue": 0.1,
        },
    },
)
```

#### `random_gray_scale`

Random grayscaling. Can be disabled by setting to `None`.

```python
import lightly_train

lightly_train.pretrain(
	...,
	transform_args={
		"random_gray_scale": 0.2,
	},
)
```

#### `gaussian_blur`

Gaussian blurring. Can be disabled by setting to `None`.

```python
import lightly_train

lightly_train.pretrain(
	...,
	transform_args={
		"gaussian_blur": {
			"prob": 0.5,
			"sigmas": (0.1, 2.0),
			"blur_limit": (3, 7),
		},
	},
)
```

#### `solarize`

Random solarization. Can be disabled by setting to `None`.

```python
import lightly_train

lightly_train.pretrain(
	...,
	transform_args={
		"solarize": {
			"prob": 0.2,
			"threshold": 0.5,
		},
	},
)
```

Methods Distillation and SimCLR have no transform configuration options beyond the
globally available ones listed above.

(method-transform-args-channel-drop)=

#### `channel_drop`

Dictionary to configure channel dropping augmentation for multi-channel datasets. It
randomly drops channels until only a specified number of channels remain. Useful for
training models on datasets with varying channel availability. Requires
`LIGHTLY_TRAIN_IMAGE_MODE="UNCHANGED"` to be set in the environment. See
[](multi-channel) for details.

```python
import lightly_train

lightly_train.pretrain(
    ...,
    transform_args={
        "channel_drop": {
            "num_channels_keep": 3,  # Number of channels to keep
            "weight_drop": [1.0, 1.0, 0.0, 0.0],  # Drop channels 1 and 2 with equal probability. Don't drop channels 3 and 4.
        },
    },
)
```

#### `global_view_1`

DINO-specific modifications for the second global view. Cannot be disabled; use nested
keys to tune the transforms.

```python
import lightly_train

lightly_train.pretrain(
	...,
	transform_args={
		"global_view_1": {
			"gaussian_blur": {
				"prob": 0.1,
				"sigmas": (0.1, 2.0),
				"blur_limit": 0,
			},
			"solarize": {
				"prob": 0.2,
				"threshold": 0.5,
			},
		},
	},
)
```

#### `local_view`

DINO-specific configuration for local views. Can be disabled by setting to `None`.

```python
import lightly_train

lightly_train.pretrain(
	...,
	transform_args={
		"local_view": {
			"num_views": 6,
			"view_size": (96, 96),
			"random_resize": {
				"min_scale": 0.05,
				"max_scale": 0.14,
			},
			"gaussian_blur": {
				"prob": 0.5,
				"sigmas": (0.1, 2.0),
				"blur_limit": 0,
			},
		},
	},
)
```
