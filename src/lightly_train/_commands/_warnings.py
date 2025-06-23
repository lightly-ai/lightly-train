#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import warnings

# Ignore warning raised by torchvision/convnext models.


def filter_train_warnings() -> None:
    filter_warnings()
    # PytorchLightning warnings.
    warnings.filterwarnings(
        "ignore",
        message=(
            "Consider setting `persistent_workers=True` in 'train_dataloader' to speed "
            "up the dataloader worker initialization."
        ),
    )
    warnings.filterwarnings(
        "ignore",
        message="The verbose parameter is deprecated. Please use get_last_lr()",
    )
    # Ignore warning as we handle it with overwrite flag.
    warnings.filterwarnings(
        "ignore",
        message="Checkpoint directory .* exists and is not empty.",
    )
    # Ignore warning as we handle it with overwrite flag.
    warnings.filterwarnings(
        "ignore",
        message=(
            "Experiment logs directory .* exists and is not empty. Previous log files "
            "in this directory can be modified when the new ones are saved!"
        ),
    )
    # Ignore mixed precision CUDA warnings as the information that CUDA is not available
    # can be found elsewhere. The same warnings don't pop up for full precision training.
    warnings.filterwarnings(
        "ignore",
        message="User provided device_type of 'cuda', but CUDA is not available.",
        module="torch.amp",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available.",
        module="torch.amp",
        category=UserWarning,
    )
    # Ignore `lr_scheduler.step()` before `optimizer.step()` warning as it's a PyTorch Lightning issue.
    # See https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    # TODO(Philipp, 09/24): Remove this once the issue is resolved.
    warnings.filterwarnings(
        "ignore",
        message="Detected call of \`lr_scheduler.step\(\)\` before \`optimizer.step\(\)\`",
        category=UserWarning,
        module="torch.optim.lr_scheduler",
    )


def filter_embed_warnings() -> None:
    filter_warnings()
    warnings.filterwarnings(
        "ignore", message="Consider setting `persistent_workers=True`"
    )


def filter_export_warnings() -> None:
    filter_warnings()


def filter_warnings() -> None:
    # PyTorch Lighting warnings
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
    warnings.filterwarnings("ignore", message="Deprecated call to `pkg_resources")
    warnings.filterwarnings(
        "ignore",
        message=(
            "torch.nn.utils.weight_norm is deprecated in favor of "
            "torch.nn.utils.parametrizations.weight_norm."
        ),
    )
    warnings.filterwarnings(
        "ignore", message="The `srun` command is available on your system"
    )

    # Torch ConvNext warning
    warnings.filterwarnings(
        "ignore", message="Grad strides do not match bucket view strides"
    )
