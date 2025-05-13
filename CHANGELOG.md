# Changelog

All notable changes to Lightly**Train** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- New, improved distillationv2 method that achieves higher accuracy and trains up to 3x faster than distillationv1.
  The new method is selected as default by LightlyTrain with `method="distillation"`. The old distillation method
  can still be used with `method="distillationv1"` for backwards compatibility.
- Support lists of files and folders as input to the `embed` and `train` commands.
- Faster dataset initialization with SLURM and Windows.
- Model is exported after every n-th epoch instead of only at the end. Default n=1
- A tutorial on how to [train embedding models with LightlyTrain](https://docs.lightly.ai/train/stable/tutorials/embedding/index.html).

### Changed

- Distillation uses the official DINOv2 implementation for the teacher model.
- The RT-DETR example uses RT-DETRv2, which is backward compatible while imposing fewer constraints on package versions.
- The original distillation method is now selected with `method="distillationv1"`.

### Deprecated

### Removed

- Dependency on the transformers library.

### Fixed

- `num_workers="auto"` now limits the number of workers to a maximum of 8 workers/GPU
  to avoid overloading systems with many CPU cores.

### Security

## [0.6.3] - 2025-04-23

### Added

- Transforms and methods are now documented on dedicated pages.
- Add [version compatibility table](https://docs.lightly.ai/train/stable/installation.html#version-compatibility) to the documentation.

### Fixed

- Fix image size mismatch issue when using TIMM models and DINO.

## [0.6.2] - 2025-04-09

### Added

- Document [RF-DETR models](https://docs.lightly.ai/train/stable/models/rtdetr.html).
- Add [frequently asked questions](https://docs.lightly.ai/train/stable/faq.html) page.
- Add [Torchvision classification tutorial](https://docs.lightly.ai/train/stable/tutorials/resnet/index.html).
- Add [depth estimation tutorial](https://docs.lightly.ai/train/stable/tutorials/depth_estimation/index.html).

### Changed

- Increase minimum Wandb version to `0.17.2` which contains fixes for `numpy>=2.0` support.
- Limit PyTorch version to `<2.6`. We'll remove this limitation once PyTorch Lightning 2.6 is released.
- Limit Python version to `<3.13`. We'll remove this limitation once PyTorch supports Python 3.13.

### Removed

- Remove Albumentations versions `1.4.18-1.4.22` support.

## [0.6.1] - 2025-03-31

### Added

- Add support for RFDETR models.
- Document platform compatibility.

### Changed

- TensorBoard is now automatically installed and no longer an optional dependency.
- Update the [Models documentation](https://docs.lightly.ai/train/stable/models/index.html).
- Update the [YOLO tutorial](https://docs.lightly.ai/train/stable/tutorials/yolo/index.html)

### Removed

- Remove DenseCL from the documentation.

## [0.6.0] - 2025-03-24

### Added

- Add support for DINOv2 distillation pretraining with the `"distillation"` method.
- Add support for [YOLO11 and YOLO12 models](https://docs.lightly.ai/train/stable/models/ultralytics.html).
- Add support for [RT-DETR models](https://docs.lightly.ai/train/stable/models/rtdetr.html).
- Add support for [YOLOv12 models](https://docs.lightly.ai/train/stable/models/yolov12.html) by the original authors.
- The Git info (branch name, commit, uncommited changes) for the LightlyTrain package
  and the directory from where the code runs are now logged in the `train.log` file.

### Changed

- The default pretraining method is now `"distillation"`.
- The default embedding format is now `"torch"`.
- The log messages in the `train.log` file are now more concise.

### Fixed

- Ensures proper usage of the `blur_limit` parameter in the `GaussianBlur` transforms.

## [0.5.0] - 2025-03-04

### Added

- Add tutorial on how to use [LightlyTrain with YOLO](https://docs.lightly.ai/train/stable/tutorials/yolo/index.html).
- Show the [`data_wait` percentage](https://docs.lightly.ai/train/stable/performance/index.html#finding-the-performance-bottleneck) in the progress bar to better monitor performance bottlenecks.
- Add [auto format](https://docs.lightly.ai/train/stable/export.html#format) export with example logging, which automatically determines the best export option for your model based on the [used model library](https://docs.lightly.ai/train/stable/models/index.html#supported-libraries).
- Add support for configuring the random rotation transform via `transform_args.random_rotation`.
- Add support for configuring the color jitter transform via `transform_args.color_jitter`.
- When using the DINO method and configuring the transforms: Removes `local_view_size`, `local_view_resize` and `n_local_views` from `DINOTransformArgs` in favor of `local_view.view_size`, `local_view.random_resize` and `local_view.num_views`. When using the CLI, replace `transform_args.local_view_size` with `transform_args.local_view.view_size`, ... respectively.
- Allow specifying the precision when using the `embed` command. The loaded checkpoint will be casted to that precision if necessary.

### Changed

- Increase default DenseCL SGD learning rate to 0.1.
- Dataset initialization is now faster when using multiple GPUs.
- Models are now automatically exported at the end of a training.
- Update the docker image to PyTorch 2.5.1, CUDA 11.8, and cuDNN 9.
- Switched from using PIL+torchvision to albumentations for the image transformations. This gives a performance boost and allows for more advanced augmentations.
- The metrics `batch_time` and `data_time` are grouped under `profiling` in the logs.

### Fixed

- Fix Ultralytics model export for Ultralytics v8.1 and v8.2
- Fix that the export command may fail when called in the same script as a train command using DDP.
- Fix the logging of the `train_loss` to report the batch_size correctly.

## [0.4.0] - 2024-12-05

### Added

- Log system information during training
- Add [Performance Tuning guide](https://docs.lightly.ai/train/stable/performance/index.html)
  with documentation for [multi-GPU](https://docs.lightly.ai/train/stable/performance/multi_gpu.html)
  and [multi-node](https://docs.lightly.ai/train/stable/performance/multi_node.html) training
- Add [Pillow-SIMD support](https://docs.lightly.ai/train/stable/performance/index.html#dataloader-bottleneck-cpu-bound)
  for faster data processing
  - The docker image now has Pillow-SIMD installed by default
- Add [`ultralytics`](https://docs.lightly.ai/train/stable/export.html#format) export format
- Add support for DINO weight decay schedule
- Add support for SGD optimizer with `optim="sgd"`
- Report final `accelerator`, `num_devices`, and `strategy` in the resolved config
- Add [Changelog](https://docs.lightly.ai/train/stable/changelog.html) to the documentation

### Changed

- Various improvements for the DenseCL method
  - Increase default memory bank size
  - Update local loss calculation
- Custom models have a [new interface](https://docs.lightly.ai/train/stable/models/custom_models.html#custom-models)
- The number of warmup epochs is now set to 10% of the training epochs for runs with less than 100 epochs
- Update default optimizer settings
  - SGD is now the default optimizer
  - Improve default learning rate and weight decay values
- Improve automatic `num_workers` calculation
- The SPPF layer of Ultralytics YOLO models is no longer trained

### Removed

- Remove DenseCLDINO method
- Remove DINO `teacher_freeze_last_layer_epochs` argument

## [0.3.2] - 2024-11-06

### Added

- Log data loading and forward/backward pass time as `data_time` and `batch_time`
- Batch size is now more uniformly handled

### Changed

- The custom model `feature_dim` property is now a method
- Replace FeatureExtractor base class by the set of Protocols

### Fixed

- Datasets support symlinks again

## [0.3.1] - 2024-10-29

### Added

- The documentation is now available at https://docs.lightly.ai/train
- Support loading checkpoint weights with the `checkpoint` argument
- Log resolved training config to tensorboard and WandB

### Fixed

- Support single-channel images by converting them to RGB
- Log config instead of locals
- Skip pooling in DenseCLDino

## [0.3.0] - 2024-10-22

### Added

- Add Ultralytics model support
- Add SuperGradients PP-LiteSeg model support
- Save normalization transform arguments in checkpoints and automatically use them
  in the embed command
- Better argument validation
- Automatically configure `num_workers` based on available CPU cores
- Add faster and more memory efficient image dataset
- Log more image augmentations
- Log resolved config for CallbackArgs, LoggerArgs, MethodArgs, MethodTransformArgs, and OptimizerArgs
