#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.#
from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    """Make a value divisible by a given divisor.

    Args:
        v: The value to make divisible.
        divisor: The divisor.
        min_value: Minimum result value.

    Returns:
        The divisible value.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """Channel shuffle operation for ShuffleNet.

    Args:
        x: Input tensor of shape (B, C, H, W).
        groups: Number of groups for shuffling.

    Returns:
        Tensor with shuffled channels.
    """
    batch_size, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups

    # Reshape: (B, C, H, W) -> (B, G, C//G, H, W)
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # Transpose: (B, G, C//G, H, W) -> (B, C//G, G, H, W)
    x = x.transpose(1, 2).contiguous()

    # Flatten: (B, C//G, G, H, W) -> (B, C, H, W)
    x = x.view(batch_size, num_channels, height, width)

    return x


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        bias: bool = False,
        act: Literal["hardswish", "relu", "none"] = "hardswish",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)

        if act == "hardswish":
            self.act: nn.Module = nn.Hardswish(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.act(self.bn(self.conv(x)))
        return out


class SEModule(nn.Module):
    """Squeeze-and-Excitation module.

    Args:
        channels: Number of input/output channels.
        reduction: Reduction ratio for the bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced_channels = channels // reduction
        self.fc1 = nn.Conv2d(channels, reduced_channels, 1)
        self.fc2 = nn.Conv2d(reduced_channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = F.hardsigmoid(self.fc2(scale), inplace=True)
        return x * scale


class EnhancedInvertedResidual(nn.Module):
    """Enhanced Inverted Residual block for ESNet backbone (stride=1).

    This block performs channel split, processes one half with convolutions
    and SE, then concatenates and shuffles.

    Args:
        in_channels: Number of input channels.
        mid_channels: Number of middle channels for ghost module.
        out_channels: Number of output channels.
        se_channels: Number of channels for SE module.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        se_channels: int,
    ) -> None:
        super().__init__()

        self.conv_pw = ConvBNAct(
            in_channels // 2, mid_channels // 2, kernel_size=1, act="hardswish"
        )
        self.conv_dw = ConvBNAct(
            mid_channels // 2,
            mid_channels // 2,
            kernel_size=3,
            groups=mid_channels // 2,
            act="none",
        )
        self.se = SEModule(se_channels, reduction=4)
        self.conv_linear = ConvBNAct(
            mid_channels, out_channels // 2, kernel_size=1, act="hardswish"
        )

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)

        x2 = self.conv_pw(x2)
        x3 = self.conv_dw(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x3 = self.se(x3)
        x3 = self.conv_linear(x3)

        out = torch.cat([x1, x3], dim=1)
        out = _channel_shuffle(out, 2)
        return out


class EnhancedInvertedResidualDS(nn.Module):
    """Enhanced Inverted Residual block with downsampling for ESNet (stride=2).

    This block has two branches that both downsample, then concatenates and
    applies additional convolutions.

    Args:
        in_channels: Number of input channels.
        mid_channels: Number of middle channels.
        out_channels: Number of output channels.
        se_channels: Number of channels for SE module.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        se_channels: int,
    ) -> None:
        super().__init__()

        self.conv_dw_1 = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            groups=in_channels,
            act="none",
        )
        self.conv_linear_1 = ConvBNAct(
            in_channels, out_channels // 2, kernel_size=1, act="hardswish"
        )

        self.conv_pw_2 = ConvBNAct(
            in_channels, mid_channels // 2, kernel_size=1, act="hardswish"
        )
        self.conv_dw_2 = ConvBNAct(
            mid_channels // 2,
            mid_channels // 2,
            kernel_size=3,
            stride=2,
            groups=mid_channels // 2,
            act="none",
        )
        self.se = SEModule(se_channels, reduction=4)
        self.conv_linear_2 = ConvBNAct(
            mid_channels // 2, out_channels // 2, kernel_size=1, act="hardswish"
        )

        self.conv_dw_mv1 = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=3,
            groups=out_channels,
            act="hardswish",
        )
        self.conv_pw_mv1 = ConvBNAct(
            out_channels, out_channels, kernel_size=1, act="hardswish"
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv_dw_1(x)
        x1 = self.conv_linear_1(x1)

        x2 = self.conv_pw_2(x)
        x2 = self.conv_dw_2(x2)
        x2 = self.se(x2)
        x2 = self.conv_linear_2(x2)

        out: Tensor = torch.cat([x1, x2], dim=1)
        out = self.conv_dw_mv1(out)
        out = self.conv_pw_mv1(out)
        return out


class ESNet(nn.Module):
    """Enhanced ShuffleNet backbone for PicoDet.

    This is a lightweight backbone based on ShuffleNetV2 with enhancements
    including SE modules and Ghost-like operations.

    Architecture:
        - Stem: 3×3 Conv (3 → 24, stride 2) + 3×3 MaxPool (stride 2)
        - Stage 1: 3× ESBlock (24 → 96)
        - Stage 2: 7× ESBlock (96 → 192)
        - Stage 3: 3× ESBlock (192 → 384)

    Args:
        model_size: Size variant ('s', 'm', or 'l').
        out_indices: Indices of blocks to return as outputs.
            Default (2, 9, 12) returns C3, C4, C5 feature maps.
        in_channels: Number of input image channels.
    """

    ARCH_SETTINGS: dict[str, dict[str, float | list[float]]] = {
        "s": {
            "scale": 0.75,
            "channel_ratios": [
                0.875,
                0.5,
                0.5,
                0.5,
                0.625,
                0.5,
                0.625,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
        },
        "m": {
            "scale": 1.0,
            "channel_ratios": [
                0.875,
                0.5,
                1.0,
                0.625,
                0.5,
                0.75,
                0.625,
                0.625,
                0.5,
                0.625,
                1.0,
                0.625,
                0.75,
            ],
        },
        "l": {
            "scale": 1.25,
            "channel_ratios": [
                0.875,
                0.5,
                1.0,
                0.625,
                0.5,
                0.75,
                0.625,
                0.625,
                0.5,
                0.625,
                1.0,
                0.625,
                0.75,
            ],
        },
    }
    STAGE_REPEATS = [3, 7, 3]  # Number of blocks in each stage

    def __init__(
        self,
        model_size: Literal["s", "m", "l"] = "s",
        out_indices: Sequence[int] = (2, 9, 12),
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        if model_size not in self.ARCH_SETTINGS:
            raise ValueError(
                f"Invalid model_size '{model_size}'. "
                f"Must be one of {list(self.ARCH_SETTINGS.keys())}."
            )

        self.model_size = model_size
        self.out_indices = out_indices

        settings = self.ARCH_SETTINGS[model_size]
        scale_val = settings["scale"]
        ratios_val = settings["channel_ratios"]
        assert isinstance(scale_val, float), (
            f"scale must be float, got {type(scale_val)}"
        )
        assert isinstance(ratios_val, list), (
            f"channel_ratios must be list, got {type(ratios_val)}"
        )
        scale = scale_val
        channel_ratios = ratios_val

        # Stage output channels: [stem, stage1, stage2, stage3]
        # Base channels: [24, 128, 256, 512] (scaled except for stem)
        stage_out_channels = [
            24,
            _make_divisible(128 * scale, divisor=16),
            _make_divisible(256 * scale, divisor=16),
            _make_divisible(512 * scale, divisor=16),
        ]
        self._out_channels = [
            stage_out_channels[1],  # Stage 1 output (C3)
            stage_out_channels[2],  # Stage 2 output (C4)
            stage_out_channels[3],  # Stage 3 output (C5)
        ]

        self.conv1 = ConvBNAct(
            in_channels, stage_out_channels[0], kernel_size=3, stride=2, act="hardswish"
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList()
        arch_idx = 0

        for stage_id, num_repeat in enumerate(self.STAGE_REPEATS):
            for i in range(num_repeat):
                in_ch = (
                    stage_out_channels[stage_id]
                    if i == 0
                    else stage_out_channels[stage_id + 1]
                )
                out_ch = stage_out_channels[stage_id + 1]
                mid_ch = _make_divisible(
                    int(out_ch * channel_ratios[arch_idx]), divisor=8
                )

                block: EnhancedInvertedResidualDS | EnhancedInvertedResidual
                if i == 0:
                    se_channels = mid_ch // 2
                    block = EnhancedInvertedResidualDS(
                        in_channels=in_ch,
                        mid_channels=mid_ch,
                        out_channels=out_ch,
                        se_channels=se_channels,
                    )
                else:
                    se_channels = mid_ch
                    block = EnhancedInvertedResidual(
                        in_channels=in_ch,
                        mid_channels=mid_ch,
                        out_channels=out_ch,
                        se_channels=se_channels,
                    )

                self.blocks.append(block)
                arch_idx += 1

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def out_channels(self) -> list[int]:
        """Return the number of output channels for each output level."""
        return self._out_channels

    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            List of feature tensors at the specified output indices.
        """
        out = self.conv1(x)
        out = self.max_pool(out)

        outs = []
        for i, block in enumerate(self.blocks):
            out = block(out)
            if i in self.out_indices:
                outs.append(out)

        return outs
