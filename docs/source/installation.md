(installation)=

# Installation

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

## Platform Compatibility

| Platform | Supported Compute |
|----------|----------------------------|
| Linux | CPU or CUDA |
| MacOS | CPU (MPS is planned) |
| Windows | CPU or CUDA (experimental) |

## Version Compatibility

| `lightly-train` | `torch` | `torchvision` | `pytorch-lightning` | Python |
|:---------------:|:-------:|:-------------:|:-------------------:|:------:|
| `0.6` | `>=2.1`, `<2.6` | `>=0.16` | `>=2.1`, `<2.6` | `>=3.8`, `<3.13` |

```{warning}
We recommend installing versions of the `torch`, `torchvision`, and `pytorch-lightning` packages that
are compatible with each other. The latest compatible versions are `torch==2.5 torchvision==0.21 pytorch-lightning==2.5`.

See the [Torchvision](https://github.com/pytorch/vision?tab=readme-ov-file#installation)
and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/versioning.html#compatibility-matrix)
documentation for more information on version compatibility between different PyTorch packages.
```

(optional-dependencies)=

## Optional Dependencies

Lightly**Train** has optional dependencies that are not installed by default. The following dependencies are available:

### Logging

- `mlflow`: For logging to [MLflow](#mlflow)
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
