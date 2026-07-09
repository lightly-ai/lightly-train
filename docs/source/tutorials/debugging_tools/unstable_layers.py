"""
Unstable Layer Variants for Debugging Tutorial

This module contains intentionally unstable layer implementations that demonstrate
common numerical stability issues in deep learning. The layers are used for
tutorial purposes to show how LightlyTrain's debugging tools can identify specific
problematic layers in a model.

The tutorial focuses on :class:`UnstableReLU`, the simplest and most reliable
example of a numerically unstable layer that we can inject into a real
``torchvision`` ResNet without touching the LightlyTrain source code.
"""

from __future__ import annotations

import torch
from torch import nn


class UnstableReLU(nn.Module):
    """An unstable ReLU variant that causes overflow in mixed precision training.

    A standard ReLU returns ``max(0, x)``. This variant instead returns
    ``exp(scale * x)`` for positive inputs, computed in **float16** so the
    result overflows the float16 range (~65,504 max) when activations exceed
    ``11 / scale``.

    Args:
        scale:
            Multiplier applied to the input before ``exp``. Default ``1.0``
            overflows quickly for any activation above ~11. Use a smaller
            value (e.g. ``0.3``) to produce a milder instability that grows
            over several training steps before overflowing — useful for the
            gradient-norm diagnostic where you want to see the explosion
            pattern rather than an instant ``nan``.

    Why it is unstable:
        - ``exp(x)`` overflows quickly. For ``x ≈ 11``, ``exp(x) ≈ 60,000``;
          for ``x ≈ 12``, ``exp(x) > 65,000`` and the result saturates to
          ``inf`` in float16.
        - The explicit ``.half()`` cast is what makes this layer unreliable
          in practice. PyTorch's autocast keeps ``torch.exp`` in float32 by
          default, so without the manual downcast the layer would *not*
          overflow in mixed precision training — a real-world gotcha when
          porting models across frameworks.
        - Once one activation overflows, the overflow propagates through
          every subsequent layer, producing ``inf``/``nan`` in the loss.

    How to fix it:
        - Use the standard ``nn.ReLU()`` (or ``torch.clamp(x, min=0)``).
        - If you genuinely need exponential scaling, clamp the input first:
          ``torch.exp(x.clamp(max=math.log(max_value)))``.
        - Consider gradient clipping to prevent explosive growth.

    What ``DebugUnderflowOverflow`` reports when this layer is used:
        - The fully qualified module name (e.g. ``model.layer2[1].relu``).
        - The frame dump shows the input range is normal but the output
          contains ``inf`` values, making the bad layer easy to spot.
    """

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Force the computation into float16. Without this cast PyTorch's
        # autocast would keep `torch.exp` in float32 and the layer would
        # NOT overflow, regardless of the global precision setting.
        x_fp16 = x.to(dtype=torch.float16)
        # `torch.exp(scale * x)` instead of the identity is what makes this
        # layer unstable. The negative branch is still safe (zeros).
        out_fp16 = torch.where(
            x_fp16 > 0,
            torch.exp(x_fp16 * self.scale),
            torch.zeros_like(x_fp16),
        )
        # Cast back to the input dtype so the rest of the network keeps its
        # dtype unchanged; the inf values survive the cast and propagate.
        return out_fp16.to(dtype=x.dtype)


def demonstrate_unstable_relu() -> None:
    """Demonstrate how :class:`UnstableReLU` produces overflow on simple inputs.

    Useful for the tutorial introduction: a single forward call on a small
    tensor is enough to show the overflow behavior.
    """
    print("UnstableReLU demonstration:\n")

    unstable_relu = UnstableReLU()

    print("1. Small positive input (still in float16 range):")
    x_small = torch.tensor([[2.0, 4.0, 6.0]])
    out_small = unstable_relu(x_small)
    print(f"   Input:  {x_small}")
    print(f"   Output: {out_small}")
    print(f"   Has inf/nan: {torch.isinf(out_small).any().item() or torch.isnan(out_small).any().item()}\n")

    print("2. Moderate positive input (overflows float16):")
    x_mod = torch.tensor([[10.0, 12.0, 14.0]])
    out_mod = unstable_relu(x_mod)
    print(f"   Input:  {x_mod}")
    print(f"   Output: {out_mod}")
    print(f"   Has inf/nan: {torch.isinf(out_mod).any().item() or torch.isnan(out_mod).any().item()}\n")

    print("3. Large positive input (way past float16 max, definitely inf):")
    x_big = torch.tensor([[20.0, 30.0, 50.0]])
    out_big = unstable_relu(x_big)
    print(f"   Input:  {x_big}")
    print(f"   Output: {out_big}")
    print(f"   Has inf/nan: {torch.isinf(out_big).any().item() or torch.isnan(out_big).any().item()}\n")

    print("4. Negative input is clamped to zero (safe branch):")
    x_neg = torch.tensor([[-2.0, -5.0, -10.0]])
    out_neg = unstable_relu(x_neg)
    print(f"   Input:  {x_neg}")
    print(f"   Output: {out_neg}")
    print(f"   Has inf/nan: {torch.isinf(out_neg).any().item() or torch.isnan(out_neg).any().item()}\n")


if __name__ == "__main__":
    demonstrate_unstable_relu()