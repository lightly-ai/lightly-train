(train-settings)=

# Train Settings

This page covers the settings available for training tasks like object detection and
segmentation in LightlyTrain. For settings related to pretraining and distillation,
please refer to the [](pretrain-settings) page.

| Name                                            | Type                          | Default        | Description                                                                                                                                                         |
| ----------------------------------------------- | ----------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`out`](#out)                                   | `str`<br>`Path`               | —              | Output directory where checkpoints, logs, and exported models are written.                                                                                          |
| [`data`](#data)                                 | `dict`<br>`str`               | —              | Dataset configuration dict, or path to a YAML file containing the dataset configuration.                                                                            |
| [`model`](#model)                               | `str`<br>`Path`               | —              | Model identifier (e.g. `"dinov3/vitt16-ltdetr-coco"`) or path to a local checkpoint to fine-tune from.                                                              |
| [`model_args`](#model_args)                     | `dict`                        | `None`         | Task/model-specific training hyperparameters.                                                                                                                       |
| [`steps`](#steps)                               | `int`                         | `"auto"`       | Number of training steps. `"auto"` selects a model-dependent default.                                                                                               |
| [`precision`](#precision)                       | `str`                         | `"bf16-mixed"` | Numeric precision mode (e.g. `"16-true"`, `"32-true"`, `"bf16-mixed"`).                                                                                             |
| [`batch_size`](#batch_size)                     | `int`                         | `"auto"`       | Global batch size across all devices.                                                                                                                               |
| [`num_workers`](#num_workers)                   | `int`                         | `"auto"`       | DataLoader worker processes per device. `"auto"` chooses a value based on available CPU cores.                                                                      |
| [`devices`](#devices)                           | `int`<br>`str`<br>`list[int]` | `"auto"`       | Devices to use for training. `"auto"` selects all available devices for the chosen `accelerator`.                                                                   |
| [`num_nodes`](#num_nodes)                       | `int`                         | `1`            | Number of nodes for distributed training.                                                                                                                           |
| [`resume_interrupted`](#resume_interrupted)     | `bool`                        | `False`        | Resume an interrupted/crashed run from the same `out` directory, including optimizer state and current step. Do not change any training parameters when using this. |
| [`overwrite`](#overwrite)                       | `bool`                        | `False`        | If `True`, overwrite the `out` directory if it already exists.                                                                                                      |
| [`accelerator`](#accelerator)                   | `str`                         | `"auto"`       | Hardware backend: `"cpu"`, `"gpu"`, `"mps"`, or `"auto"` to pick the best available.                                                                                |
| [`strategy`](#strategy)                         | `str`                         | `"auto"`       | Distributed training strategy (e.g. `"ddp"`). `"auto"` selects a suitable default.                                                                                  |
| [`seed`](#seed)                                 | `int`<br>`None`               | `0`            | Random seed for reproducibility. Set to `None` to disable seeding.                                                                                                  |
| [`logger_args`](#logger_args)                   | `dict`                        | `None`         | Logger configuration dict. `None` uses defaults; keys configure or disable individual loggers.                                                                      |
| [`transform_args`](#transform_args)             | `dict`                        | `None`         | Data transform configuration (e.g. image size, normalization).                                                                                                      |
| [`save_checkpoint_args`](#save_checkpoint_args) | `dict`                        | `None`         | Checkpoint saving configuration (e.g. save frequency).                                                                                                              |

```{tip}
LightlyTrain automatically selects suitable default values based on the chosen model,
dataset, and hardware. You only need to set parameters that you want to customize.

Look for the `Resolved Args` dictionary in the `train.log` file in the output directory
of your run to see the final settings that were applied. This will include any
overrides, automatically resolved values, and model-specific settings that are not
listed on this page.
```

(train-settings-output)=

## Output

### `out`

The output directory where the model checkpoints and logs are saved. Create a new
directory for each run! LightlyTrain will raise an error if the output directory already
exists unless the [`overwrite`](#overwrite) flag is set to `True`.

### `overwrite`

Set to `True` to overwrite the contents of an existing `out` directory. By default,
LightlyTrain raises an error if the specified output directory already exists to prevent
accidental data loss.

(train-settings-data)=

## Data

### `data`

Dataset configuration. You can either provide a dictionary with dataset parameters or a
path to a YAML file containing the dataset configuration. Each task (detection,
segmentation, etc.) has different dataset requirements. Refer to the task documentation
for details on the expected dataset structure and configuration options.

- [Object Detection](object-detection-data)
- [Instance Segmentation](instance-segmentation-data)
- [Panoptic Segmentation](panoptic-segmentation-data)
- [Semantic Segmentation](semantic-segmentation-data)

### `batch_size`

Global batch size across all devices. The batch size per device/GPU is computed as
`batch_size / (num_devices * num_nodes)`. By default, `batch_size` is set to `"auto"`,
which selects a model-dependent default value.

### `num_workers`

Number of background worker processes per device used by the PyTorch DataLoader. By
default, this is set to `"auto"`, which selects a value based on the number of available
CPU cores.

(train-settings-model)=

## Model

### `model`

Model identifier (for example `"dinov3/vitt16-ltdetr-coco"`) or the path to a checkpoint
or exported model file. LightlyTrain automatically downloads weights if needed.

To resume from a crashed or interrupted run, use the
[`resume_interrupted`](#resume_interrupted) setting instead of pointing `model` to a
previous checkpoint. This ensures that optimizer state and training progress are
restored correctly.

### `model_args`

Dictionary with model-specific training parameters. The available keys vary by
architecture. The table lists the most commonly tuned options:

| Key                                             | Type                      | Description                        |
| ----------------------------------------------- | ------------------------- | ---------------------------------- |
| [`lr`](#lr)                                     | `float`                   | Base learning rate.                |
| [`backbone_weights`](#backbone_weights)         | `Path`<br>`str`<br>`None` | Path to backbone weights to load.  |
| [`metric_log_classwise`](#metric_log_classwise) | `bool`                    | Whether to log class-wise metrics. |

#### `lr`

Base learning rate for the optimizer. All models come with a good default learning rate.
Only set this value if you want to override the default for your specific use case. We
recommend to perform learning rate sweeps to find the optimal value for your dataset.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    model_args={
        "lr": 0.0001,
    },
)
```

#### `backbone_weights`

Path to a checkpoint or exported model containing backbone weights to load before
training. This enables loading custom pretrained weights. See [](pretrain-distill) for
more details on pretraining on unlabeled data.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    model="dinov3/vitt16-ltdetr",  # Model without fine-tuned weights.
    model_args={
        "backbone_weights": "/path/to/backbone_weights.ckpt",
    },
)
```

The backbone weights argument is ignored when loading an existing checkpoint via the
[`model`](#model) argument:

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    model="out/my_experiment/checkpoints/last.ckpt",  # Loads full checkpoint including backbone weights.
    model_args={
        "backbone_weights": "/path/to/backbone_weights.ckpt",  # Ignored when loading from checkpoint.
    },
)
```

Similarly, the backbone weights argument is also ignored when loading one of the
built-in fine-tuned models:

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    model="dinov3/vitt16-ltdetr-coco",  # Loads built-in fine-tuned model.
    model_args={
        "backbone_weights": "/path/to/backbone_weights.ckpt",  # Ignored when loading built-in model.
    },
)
```

The backbone weights are only loaded when training starts from scratch using a model
identifier without a dataset suffix (e.g. `-coco`, `-cityscapes`, etc.).

#### `metric_log_classwise`

If set to `True`, class-wise metrics (for example AP per class) are logged during
validation. Default is `False` to reduce logging overhead. Not all models support this
feature.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    model_args={
        "metric_log_classwise": True,
    },
)
```

(train-settings-training-loop)=

## Training Loop

### `steps`

Total number of training steps. The default is `"auto"`, which selects a model-dependent
value. Reduce for shorter training or increase to train longer.

Epoch based training is currently not supported.

### `precision`

Training precision setting. Must be one of the following strings:

- `"bf16-mixed"`: Default. Operations run in bfloat16 where supported, weights are saved
  in float32. Not supported on all hardware.
- `"16-true"`: All operations and weights in float16. Fastest but may be unstable
  depending on model, hardware, and dataset.
- `"16-mixed"`: Most operations run in float16 precision. Not supported on all hardware.
- `"32-true"`: All operations and weights are in float32. Slower but more stable.
  Supported on all hardware.

### `seed`

Controls reproducibility for data order, augmentation randomness, and initialization.
Set to `None` to use a random seed on each run. Default is `0`.

(train-settings-hardware)=

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

(train-settings-resume-training)=

## Resume Training

There are two ways to continue training from a previous run:

1. [Resume an interrupted/crashed run](#resume_interrupted) and finish training with the
   same parameters.
   - You **CANNOT** change any training parameters (including steps)!
   - You **CANNOT** change the `out` directory.
   - YOU **CANNOT** change the dataset.
   - This restores the exact training state, including optimizer parameters and current
     step.
1. [Load a checkpoint from a previous run](#load-checkpoint-for-a-new-run) and fine-tune
   with different parameters.
   - You **CAN** change training parameters.
   - You **MUST** specify a new `out` directory.
   - You **CAN** change the dataset.
   - This initializes model weights from the checkpoint but starts a fresh training
     state.

### `resume_interrupted`

Use when a run terminates unexpectedly and you want to continue from the latest
checkpoint stored in `out/checkpoints/last.ckpt`. Do not modify any other training
arguments! This will restore the exact training state, including optimizer parameters,
current step, and any learning rate or other schedules from the previous run. The flag
is intended for crash recovery only. See [](#load-checkpoint-for-a-new-run) for
continuing training with different parameters, for example to train for more steps.

```python
import lightly_train

lightly_train.train_object_detection(
    out="out/my_experiment",  # Same output directory as the interrupted run.
    resume_interrupted=True,  # Resume from last.ckpt in out directory.
)
```

### Load Checkpoint for a New Run

To continue training from a previous run but change training parameters (for example to
train for more steps), set the `model` argument to the path of an exported model from a
previous run and specify a new `out` directory. This way, training starts fresh but
initializes weights from the provided checkpoint.

We recommend using the exported best model weights from
`out/my_experiment/exported_models/exported_best.pt` for this purpose.

See [`resume_interrupted`](#resume_interrupted) if you want to recover from a crashed
run instead.

```python
import lightly_train

lightly_train.train_object_detection(
    out="out/my_new_experiment",  # New output directory for new run.
    model="out/my_experiment/exported_models/exported_best.pt",  # Load model from previous run.
    steps=2000,  # Change training parameters as needed.
)
```

(train-settings-checkpoint-saving)=

## Checkpoint Saving

LightlyTrain saves two types of checkpoints during training:

1. `out/my_experiment/checkpoints`: Full checkpoints including optimizer, scheduler, and
   training state. Used to resume training with
   [`resume_interrupted`](#resume_interrupted).
   - `last.ckpt`: Latest checkpoint saved at regular intervals.
   - `best.ckpt`: Best-performing checkpoint based on a validation metric.
1. `out/my_experiment/exported_models`: Lightweight exported models containing only
   model weights. Used for inference and any further fine-tuning.
   - `exported_last.pt`: Model weights from the latest checkpoint.
   - `exported_best.pt`: Model weights from the best-performing checkpoint.

Use the exported models in `out/my_experiment/exported_models/` for any downstream tasks
whenever training completed successfully. Use `out/my_experiment/checkpoints/` only to
resume training with [`resume_interrupted`](#resume_interrupted) after an unexpected
interruption.

### `save_checkpoint_args`

Settings to configure checkpoint saving behavior. By default, LightlyTrain saves
`last.ckpt` and `best.ckpt` while tracking a validation metric defined by the selected
model.

| Key                                             | Type               | Description                                                                                           |
| ----------------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------------------- |
| [`save_every_num_steps`](#save_every_num_steps) | `int`              | Training step interval for saving checkpoints.                                                        |
| [`save_last`](#save_last)                       | `bool`             | Persist `last.ckpt` after each save cycle. Disable only when storage is constrained.                  |
| [`save_best`](#save_best)                       | `bool`             | Track the best-performing checkpoint according to [`watch_metric`](#watch_metric).                    |
| [`watch_metric`](#watch_metric)                 | `str`              | Validation metric name (for example `"val_metric/map"`) monitored when selecting the best checkpoint. |
| [`mode`](#mode)                                 | `"min"`<br>`"max"` | Operation used when selecting the best checkpoint based on [`watch_metric`](#watch_metric).           |

#### `save_every_num_steps`

Number of training steps between each checkpoint save. Default is `1000`. Decrease to
save more frequently. Too frequent saving may slow down training.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    save_checkpoint_args={
        "save_every_num_steps": 500,  # Save checkpoint every 500 steps.
    },
)
```

#### `save_last`

If set to `True`, the latest checkpoint and exported model (`last.ckpt` and
`exported_last.pt`) are saved at each save interval. Default is `True`. Disable only
when storage space is limited.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    save_checkpoint_args={
        "save_last": False,  # Disable saving last.ckpt
    },
)
```

#### `save_best`

If set to `True`, the best-performing checkpoint and exported model (`best.ckpt` and
`exported_best.pt`) are tracked and saved based on the validation metric defined by
[`watch_metric`](#watch_metric). Default is `True`.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    save_checkpoint_args={
        "save_best": False,  # Disable saving best.ckpt
    },
)
```

#### `watch_metric`

Validation metric used to determine the best checkpoint when [`save_best`](#save_best)
is `True`. The default metric depends on the selected model.

Default metrics:

- Object Detection: `"val_metric/map"` (Mean Average Precision)
- Instance Segmentation: `"val_metric/map"` (Mean Average Precision)
- Panoptic Segmentation: `"val_metric/pq"` (Panoptic Quality)
- Semantic Segmentation: `"val_metric/miou"` (Mean Intersection over Union)

Check the logs for all available validation metrics for your task and model. See also
[`metric_log_classwise`](#metric_log_classwise) to enable class-wise metric logging.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    save_checkpoint_args={
        "watch_metric": "val_metric/map",  # Use mAP as the best-checkpoint metric.
        "mode": "max", # Higher is better for mAP. Set to "min" for metrics where lower is better.
    },
)
```

#### `mode`

Operation used when selecting the best checkpoint based on
[`watch_metric`](#watch_metric). Must be either `"min"` (lower is better) or `"max"`
(higher is better). Default depends on the selected [`watch_metric`](#watch_metric).

(train-settings-logging)=

## Logging

### `logger_args`

Dictionary to configure logging behavior. By default, LightlyTrain uses the built-in
TensorBoard logger. You can customize logging frequency and enable/disable additional
loggers like MLflow and Weights & Biases.

| Key                                                   | Type              | Description                                                 |
| ----------------------------------------------------- | ----------------- | ----------------------------------------------------------- |
| [`mlflow`](#mlflow)                                   | `dict`<br>`None`  | MLflow logger configuration. Disabled by default.           |
| [`wandb`](#wandb)                                     | `dict`<br>`None`  | Weights & Biases logger configuration. Disabled by default. |
| [`tensorboard`](#tensorboard)                         | `dict`<br>`None`  | TensorBoard logger configuration. Set to `None` to disable. |
| [`log_every_num_steps`](#log_every_num_steps)         | `int`<br>`"auto"` | Training step interval for logging training metrics.        |
| [`val_every_num_steps`](#val_every_num_steps)         | `int`<br>`"auto"` | Training step interval that triggers a validation run.      |
| [`val_log_every_num_steps`](#val_log_every_num_steps) | `int`<br>`"auto"` | Validation step interval for logging validation metrics.    |

#### `mlflow`

MLFlow logger configuration. It is disabled by default. Requires MLFlow to be installed
with:

```bash
pip install "lightly-train[mlflow]"
```

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    logger_args={
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

lightly_train.train_object_detection(
    ...,
    logger_args={
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
    logger_args={
        "tensorboard": None,  # Disable TensorBoard logging.
    },
)
```

#### `log_every_num_steps`

Controls how frequently training metrics are written. Is set to `"auto"` by default,
which selects a value based on the dataset size. Decrease the value for more frequent
updates.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    logger_args={
        "log_every_num_steps": 50,  # Log every 50 training steps.
    },
)
```

#### `val_every_num_steps`

Controls how frequently validation is performed during training. Is set to `"auto"` by
default, which selects a value based on the dataset size. `"auto"` validates at least
once every 1000 steps. Decrease the value to validate more often.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    logger_args={
        "val_every_num_steps": 500,  # Validate every 500 training steps.
    },
)
```

#### `val_log_every_num_steps`

Controls how frequently progress is logged during validation runs. Is set to `"auto"` by
default, which selects a value based on the dataset size.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    logger_args={
        "val_log_every_num_steps": 20,  # Log every 20 validation steps.
    },
)
```

(train-settings-transforms)=

## Transforms

LightlyTrain automatically applies suitable data augmentations and preprocessing steps
for each model and task. The default transforms are designed to work well in most
scenarios. You can customize transform parameters via the
[`transform_args`](#transform_args) setting.

### `transform_args`

Dictionary to configure data transforms applied during training. The most commonly
customized parameters are listed in the table below:

| Key                                     | Type              | Description                                                |
| --------------------------------------- | ----------------- | ---------------------------------------------------------- |
| [`image_size`](#image_size)             | `tuple[int, int]` | Image height and width after random cropping and resize.   |
| [`normalize`](#normalize)               | `dict`            | Mean and standard deviation used for input normalization.  |
| [`random_flip`](#random_flip)           | `dict`            | Horizontal or vertical flip probabilities.                 |
| [`random_rotate`](#random_rotate)       | `dict`            | Rotation angle range and probability.                      |
| [`random_rotate_90`](#random_rotate_90) | `dict`            | 90-degree rotation probability.                            |
| [`color_jitter`](#color_jitter)         | `dict`            | Strength of color jitter augmentation.                     |
| [`channel_drop`](#channel_drop)         | `dict`            | Channel dropping configuration for multi-channel datasets. |
| [`val`](#val)                           | `dict`            | Validation transform configuration.                        |

Check the respective task pages for the default transforms applied:

- [Object Detection](object-detection-transform-args)
- [Instance Segmentation](instance-segmentation-transform-args)
- [Panoptic Segmentation](panoptic-segmentation-transform-args)
- [Semantic Segmentation](semantic-segmentation-transform-args)

#### `image_size`

Tuple specifying the height and width of input images after cropping and resizing. The
default size depends on the selected model. Increase for higher-resolution inputs or
decrease to speed up training. Not all image sizes are supported by all models.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    transform_args={
        "image_size": (512, 512),  # Random crop and resize images to (height, width)
    },
)
```

#### `normalize`

Dictionary specifying the mean and standard deviation used for input normalization.
ImageNet statistics are used by default. Change these values when working with datasets
that have different color distributions.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    transform_args={
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
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

lightly_train.train_object_detection(
    ...,
    transform_args={
        "random_flip": {
            "horizontal_prob": 0.7,  # 70% chance to flip horizontally
            "vertical_prob": 0.2,    # 20% chance to flip vertically
        },
    },
)
```

#### `random_rotate`

Dictionary to configure random rotation augmentation. By default, rotation is disabled.
Specify the maximum rotation angle and probability to enable.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    transform_args={
        "random_rotate": {
            "prob": 0.5,     # 50% chance to apply rotation
            "degrees": (-30, 30),  # Rotate between -30 and +30 degrees
        },
    },
)
```

#### `random_rotate_90`

Dictionary to configure random rotation by a multiple of 90 degrees. By default, this is
disabled. Specify the probability to enable.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    transform_args={
        "random_rotate_90": {
            "prob": 0.3,  # 30% chance to rotate by 90/180/270 degrees
        },
    },
)
```

#### `color_jitter`

Dictionary to configure color jitter augmentation. By default, color jitter is disabled.
Not all models support color jitter augmentation.

```python
import lightly_train

lightly_train.train_object_detection(
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

#### `channel_drop`

Dictionary to configure channel dropping augmentation for multi-channel datasets. It
randomly drops channels until only a specified number of channels remain. Useful for
training models on datasets with varying channel availability.

```python
import lightly_train

lightly_train.train_object_detection(
    ...,
    transform_args={
        "channel_drop": {
            "num_channels_keep": 3,  # Number of channels to keep
            "weight_drop": [1.0, 1.0, 0.0, 0.0],  # Drop channels 1 and 2 with equal probability. Don't drop channels 3 and 4.
        },
    },
)
```

#### `val`

Dictionary to configure validation transforms. Can be used to override validation
transforms separately from training transforms. By default, validation transforms use
the same image size and normalization as training transforms, but disable other
augmentations.

```python

import lightly_train

lightly_train.train_object_detection(
    ...,
    transform_args={
        "image_size": (518, 518), # Resize training images to (height, width)
        "val": {
            "image_size": (512, 512),  # Resize validation images to (height, width)
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
pretrain_settings
```
