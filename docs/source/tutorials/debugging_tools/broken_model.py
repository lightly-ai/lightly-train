"""
Broken Model Construction for the Debugging Tutorial

The LightlyTrain fine-tuning API (``train_image_classification``, etc.) accepts
the model as a string of the form ``"<package>/<model_name>"`` and creates the
underlying ``torch.nn.Module`` itself. There is no public hook to inject a
custom layer into that model.

To work around this without modifying LightlyTrain, we **monkey-patch** the
relevant ``torchvision`` factory function so that any ``resnet18`` created by
LightlyTrain comes out with one of its ReLU activations replaced by
:class:`unstable_layers.UnstableReLU`. We expose this as a context manager so
the patch is automatically reverted after training.

Why a context manager?
    The patch mutates global state in ``torchvision``. A context manager keeps
    the change scoped to the training call and leaves the rest of the
    user's program (and any subsequent imports) untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models._api import BUILTIN_MODELS

from unstable_layers import UnstableReLU

# Fully qualified layer name that DebugUnderflowOverflow will report.
# This is the path LightlyTrain prints when the monitor fires on the broken model.
BROKEN_LAYER_NAME = "model.layer2[1].relu"


@contextmanager
def patched_resnet18(scale: float = 1.0) -> Iterator[None]:
    """Patch ``torchvision.models.resnet18`` so the model has ``UnstableReLU``.

    While the context manager is active, every call to
    ``torchvision.models.resnet18(...)`` (or
    ``torchvision.models.get_model("resnet18", ...)``) returns a ResNet18
    where ``layer2[1].relu`` is replaced by :class:`UnstableReLU`. The patch is
    reverted on exit, even if an exception is raised inside the ``with`` block.

    LightlyTrain's fine-tuning commands build the model internally via
    ``torchvision.models.get_model("resnet18", ...)``, so the patched
    function is what gets invoked when ``model="torchvision/resnet18"`` is
    passed to ``train_image_classification``.

    Args:
        scale:
            Passed through to :class:`UnstableReLU`. ``1.0`` (the default)
            overflows quickly and is best paired with the
            ``DebugUnderflowOverflow`` monitor. Smaller values (e.g. ``0.3``)
            produce a milder instability that grows over several training
            steps — useful when you want to visualise gradient-norm
            explosion without an immediate abort.

    Yields:
        None. The patch is active for the duration of the ``with`` block.

    Example:
        >>> with patched_resnet18():
        ...     lightly_train.train_image_classification(
        ...         model="torchvision/resnet18",
        ...         ...
        ...     )
    """
    # Save the originals so we can restore them on exit.
    import torchvision.models
    import torchvision.models.resnet

    original_resnet18 = torchvision.models.resnet.resnet18
    original_in_builtin = BUILTIN_MODELS["resnet18"]

    def broken_resnet18(*args: object, **kwargs: object) -> nn.Module:
        model = original_resnet18(*args, **kwargs)
        # Replace the activation inside the second block of layer2.
        # This is where the overflow will be triggered.
        model.layer2[1].relu = UnstableReLU(scale=scale)
        return model

    # Patch all three references to be safe:
    #   - torchvision.models.resnet18 (re-export)
    #   - torchvision.models.resnet.resnet18 (definition site)
    #   - torchvision.models._api.BUILTIN_MODELS["resnet18"] (the registry
    #     used by torchvision.models.get_model)
    torchvision.models.resnet18 = broken_resnet18
    torchvision.models.resnet.resnet18 = broken_resnet18
    BUILTIN_MODELS["resnet18"] = broken_resnet18

    try:
        yield
    finally:
        torchvision.models.resnet18 = original_resnet18
        torchvision.models.resnet.resnet18 = original_resnet18
        BUILTIN_MODELS["resnet18"] = original_in_builtin


def make_broken_model(
    weights: ResNet18_Weights | None = None, scale: float = 1.0
) -> nn.Module:
    """Build a torchvision ResNet18 with the unstable ReLU baked in.

    Use this only for forward-pass sanity checks (e.g. confirming that the
    patched model actually produces ``inf``/``nan``). The fine-tuning scripts
    use :func:`patched_resnet18` instead so that LightlyTrain's internally
    constructed model is the one that gets the unstable layer.

    Args:
        weights:
            Pretrained weights to load, or ``None`` for random initialization.
            Defaults to ``None``.
        scale:
            Passed through to :class:`UnstableReLU`. See :func:`patched_resnet18`
            for guidance on choosing a value.

    Returns:
        A ResNet18 with ``UnstableReLU`` replacing the activation at
        ``layer2[1].relu``.
    """
    model = resnet18(weights=weights)
    model.layer2[1].relu = UnstableReLU(scale=scale)
    return model


def make_fixed_model(weights: ResNet18_Weights | None = None) -> nn.Module:
    """Build a standard torchvision ResNet18 — used as the "fixed" baseline.

    Args:
        weights:
            Pretrained weights to load, or ``None`` for random initialization.
            Defaults to ``None``.

    Returns:
        A standard ResNet18 with no modifications.
    """
    return resnet18(weights=weights)


def smoke_test() -> None:
    """Confirm the broken and fixed models behave as expected on CPU.

    This is a small unit test that the tutorial author can run after
    editing :mod:`unstable_layers` to make sure the unstable layer still
    triggers overflow.
    """
    print("Smoke test: confirming broken model produces inf/nan.\n")

    for label, model in (
        ("BROKEN (UnstableReLU at layer2[1].relu)", make_broken_model()),
        ("BROKEN (mild UnstableReLU, scale=0.3)   ", make_broken_model(scale=0.3)),
        ("FIXED  (standard ResNet18)               ", make_fixed_model()),
    ):
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 3, 224, 224) * 1.0
            y = model(x)
        bad = torch.isinf(y).any().item() or torch.isnan(y).any().item()
        flag = "OVERFLOW" if bad else "OK"
        print(
            f"  {label}  -> output range "
            f"[{y.min().item():+.3e}, {y.max().item():+.3e}]  [{flag}]"
        )
    print(
        "\nExpected: BROKEN should report OVERFLOW (the unstable layer overflows), "
        "FIXED should be OK."
    )


if __name__ == "__main__":
    smoke_test()