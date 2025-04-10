(installation)=

# Installation

## Platform Compatibility

LightlyTrain is officially supported on

- Linux: CPU or CUDA
- MacOS: CPU only

We are planning to support MPS for MacOS, and to support Windows in the future.

## Installation from PyPI

Lightly**Train** is available on [PyPI](https://pypi.org/project/lightly-train/) and can be installed via pip or other package managers.

```{warning}
To successfully install Lightly**Train** the Python version has to be >=3.8 and <=3.12 .
```

```bash
pip install lightly-train
```

To update to the latest version, run:

```bash
pip install --upgrade lightly-train
```

See {ref}`docker` for Docker installation instructions.

(optional-dependencies)=

## Optional Dependencies

Lightly**Train** has optional dependencies that are not installed by default. The following dependencies are available:

### Logging

- `wandb`: For logging to [Weights & Biases](#wandb)

### Model Support

- `rfdetr`: For [RF-DETR](#models-rfdetr) models
- `super-gradients`: For [SuperGradients](#models-supergradients) models
- `timm`: For [TIMM](#models-timm) models
- `ultralytics`: For [Ultralytics](#models-ultralytics) models

To install optional dependencies, run:

```bash
pip install "lightly-train[wandb]"
```

Or for multiple optional dependencies:

```bash
pip install "lightly-train[wandb,timm]"
```

## Hardware Recommendations

An example hardware setup and its performance when using Lightly**Train** is provided in {ref}`hardware-recommendations`.
