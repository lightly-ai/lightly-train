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

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for depthwise conv.
        stride: Stride for depthwise conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.act(self.bn(self.conv(x)))
        return out


class DarknetBottleneck(nn.Module):
    """Basic bottleneck block used in CSP layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for the second convolution.
        expansion: Expansion ratio for hidden channels.
        add_identity: Whether to add residual connection.
        use_depthwise: Whether to use depthwise separable conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = True,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        self.conv2: DepthwiseSeparableConv | ConvBNAct
        if use_depthwise:
            self.conv2 = DepthwiseSeparableConv(
                hidden_channels, out_channels, kernel_size=kernel_size
            )
        else:
            self.conv2 = ConvBNAct(
                hidden_channels, out_channels, kernel_size=kernel_size
            )

        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out: Tensor = self.conv1(x)
        out = self.conv2(out)
        if self.add_identity:
            out = out + identity
        return out


class CSPLayer(nn.Module):
    """Cross Stage Partial layer.

    This layer splits the input, processes one path through bottleneck blocks,
    and concatenates with the other path before a final convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for bottleneck convolutions.
        expansion: Expansion ratio for hidden channels.
        num_blocks: Number of bottleneck blocks.
        add_identity: Whether to add residual connections in blocks.
        use_depthwise: Whether to use depthwise separable convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        expansion: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = False,
        use_depthwise: bool = True,
    ) -> None:
        super().__init__()
        mid_channels = int(out_channels * expansion)

        self.short_conv = ConvBNAct(in_channels, mid_channels, kernel_size=1)

        self.main_conv = ConvBNAct(in_channels, mid_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                DarknetBottleneck(
                    mid_channels,
                    mid_channels,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    add_identity=add_identity,
                    use_depthwise=use_depthwise,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_conv = ConvBNAct(2 * mid_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x_short: Tensor = self.short_conv(x)
        x_main: Tensor = self.main_conv(x)
        x_main = self.blocks(x_main)
        x_final = torch.cat([x_main, x_short], dim=1)
        out: Tensor = self.final_conv(x_final)
        return out


class CSPPAN(nn.Module):
    """Cross Stage Partial - Path Aggregation Network.

    This neck takes multi-level features from the backbone (C3, C4, C5) and
    produces unified-channel feature maps (P3, P4, P5, P6) through top-down
    and bottom-up pathways with CSP blocks.

    Args:
        in_channels: Number of input channels for each level from backbone.
        out_channels: Number of output channels (unified across all levels).
        kernel_size: Kernel size for depthwise convolutions.
        num_features: Number of output feature levels (3 or 4).
        expansion: Expansion ratio for CSP layers.
        num_csp_blocks: Number of bottleneck blocks in each CSP layer.
        use_depthwise: Whether to use depthwise separable convolutions.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 96,
        kernel_size: int = 5,
        num_features: int = 4,
        expansion: float = 0.5,
        num_csp_blocks: int = 1,
        use_depthwise: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features

        # Transform input channels to unified out_channels
        self.transforms = nn.ModuleList()
        for in_ch in in_channels:
            self.transforms.append(ConvBNAct(in_ch, out_channels, kernel_size=1))

        # Extra convs for P6 if num_features == 4
        if num_features == 4:
            self.first_top_conv = DepthwiseSeparableConv(
                out_channels, out_channels, kernel_size=kernel_size, stride=2
            )
            self.second_top_conv = DepthwiseSeparableConv(
                out_channels, out_channels, kernel_size=kernel_size, stride=2
            )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.top_down_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.top_down_blocks.append(
                CSPLayer(
                    out_channels * 2,
                    out_channels,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                )
            )

        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsamples.append(
                DepthwiseSeparableConv(
                    out_channels, out_channels, kernel_size=kernel_size, stride=2
                )
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    out_channels * 2,
                    out_channels,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                )
            )

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

    def forward(self, inputs: Sequence[Tensor]) -> tuple[Tensor, ...]:
        """Forward pass.

        Args:
            inputs: List of feature tensors [C3, C4, C5] from backbone.

        Returns:
            Tuple of feature tensors [P3, P4, P5] or [P3, P4, P5, P6].
        """
        assert len(inputs) == len(self.in_channels)

        # Transform to unified channels
        inputs = [transform(x) for transform, x in zip(self.transforms, inputs)]

        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            upsample_feat = self.upsample(feat_high)

            # Handle size mismatch due to odd spatial dimensions
            if upsample_feat.shape[-2:] != feat_low.shape[-2:]:
                upsample_feat = F.interpolate(
                    upsample_feat, size=feat_low.shape[-2:], mode="nearest"
                )

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_high], dim=1)
            )
            outs.append(out)

        # Add P6 if needed
        if self.num_features == 4:
            top_features = self.first_top_conv(inputs[-1])
            top_features = top_features + self.second_top_conv(outs[-1])
            outs.append(top_features)

        return tuple(outs)
