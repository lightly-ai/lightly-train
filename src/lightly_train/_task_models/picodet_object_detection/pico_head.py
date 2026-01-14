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

from lightly_train._task_models import task_model_helpers


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for the detection head.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for depthwise conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
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


class Integral(nn.Module):
    """Integral layer for Distribution Focal Loss (DFL) decoding.

    This layer converts the discrete distribution outputs to continuous
    regression values by computing the expectation.

    Args:
        reg_max: Maximum value of the integral set {0, 1, ..., reg_max}.
    """

    def __init__(self, reg_max: int = 7) -> None:
        super().__init__()
        self.reg_max = reg_max
        # Register project as buffer (values 0 to reg_max)
        self.register_buffer(
            "project", torch.linspace(0, reg_max, reg_max + 1), persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """Convert distribution to single value via integration.

        Args:
            x: Distribution logits of shape (..., 4*(reg_max+1)) or
               (..., reg_max+1).

        Returns:
            Integrated values of shape (..., 4) or (...,).
        """
        # Reshape to (..., reg_max+1) for softmax
        original_shape = x.shape
        x = x.reshape(-1, self.reg_max + 1)

        # Softmax to get distribution
        x = F.softmax(x, dim=-1)

        # Compute expectation
        project: Tensor = self.project  # type: ignore[assignment]
        x = F.linear(x, project.view(1, -1)).squeeze(-1)

        # Reshape back to (..., 4) if input was 4*(reg_max+1)
        if original_shape[-1] == 4 * (self.reg_max + 1):
            return x.reshape(*original_shape[:-1], 4)
        return x.reshape(*original_shape[:-1])


def distance2bbox(
    points: Tensor, distances: Tensor, max_shape: tuple[int, int] | None = None
) -> Tensor:
    """Convert distances from points to bounding boxes.

    Args:
        points: Center points of shape (N, 2) as [x, y].
        distances: Distances of shape (N, 4) as [left, top, right, bottom].
        max_shape: Maximum shape (H, W) for clamping.

    Returns:
        Bounding boxes of shape (N, 4) as [x1, y1, x2, y2].
    """
    x1 = points[..., 0] - distances[..., 0]
    y1 = points[..., 1] - distances[..., 1]
    x2 = points[..., 0] + distances[..., 2]
    y2 = points[..., 1] + distances[..., 3]

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])

    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox2distance(
    points: Tensor, bboxes: Tensor, reg_max: float | None = None
) -> Tensor:
    """Convert bounding boxes to distances from points.

    Args:
        points: Center points of shape (N, 2) as [x, y].
        bboxes: Bounding boxes of shape (N, 4) as [x1, y1, x2, y2].
        reg_max: Maximum distance value for clamping.

    Returns:
        Distances of shape (N, 4) as [left, top, right, bottom].
    """
    left = points[..., 0] - bboxes[..., 0]
    top = points[..., 1] - bboxes[..., 1]
    right = bboxes[..., 2] - points[..., 0]
    bottom = bboxes[..., 3] - points[..., 1]

    if reg_max is not None:
        left = left.clamp(min=0, max=reg_max - 0.01)
        top = top.clamp(min=0, max=reg_max - 0.01)
        right = right.clamp(min=0, max=reg_max - 0.01)
        bottom = bottom.clamp(min=0, max=reg_max - 0.01)

    return torch.stack([left, top, right, bottom], dim=-1)


def generate_grid_points(
    height: int, width: int, stride: int, device: torch.device, offset: float = 0.5
) -> Tensor:
    """Generate grid center points for a feature map.

    Args:
        height: Feature map height.
        width: Feature map width.
        stride: Stride (downsampling factor) of the feature map.
        device: Device to create tensors on.
        offset: Offset for center points (0.5 means center of cell).

    Returns:
        Grid points of shape (H*W, 2) as [x, y] in pixel coordinates.
    """
    y = (torch.arange(height, device=device, dtype=torch.float32) + offset) * stride
    x = (torch.arange(width, device=device, dtype=torch.float32) + offset) * stride
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], dim=-1)


class PicoHead(nn.Module):
    """Anchor-free detection head for PicoDet.

    This head uses GFL-style outputs with shared cls/reg branches,
    Distribution Focal Loss for regression, and Varifocal Loss for
    classification.

    Args:
        in_channels: Number of input channels (from neck).
        num_classes: Number of object classes.
        feat_channels: Number of feature channels in the head.
        stacked_convs: Number of stacked convolution layers.
        kernel_size: Kernel size for depthwise convolutions.
        reg_max: Maximum value for DFL distribution.
        strides: Stride for each feature level.
        share_cls_reg: Whether to share convs between cls and reg branches.
        use_depthwise: Whether to use depthwise separable convolutions.
    """

    def __init__(
        self,
        in_channels: int = 96,
        num_classes: int = 80,
        feat_channels: int = 96,
        stacked_convs: int = 2,
        kernel_size: int = 5,
        reg_max: int = 7,
        strides: Sequence[int] = (8, 16, 32, 64),
        share_cls_reg: bool = True,
        use_depthwise: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.reg_max = reg_max
        self.strides = strides
        self.share_cls_reg = share_cls_reg

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.gfl_cls = nn.ModuleList()
        self.gfl_reg = nn.ModuleList()

        for _ in strides:
            cls_convs = nn.ModuleList()
            for i in range(stacked_convs):
                chn = in_channels if i == 0 else feat_channels
                if use_depthwise:
                    cls_convs.append(
                        DepthwiseSeparableConv(chn, feat_channels, kernel_size)
                    )
                else:
                    cls_convs.append(
                        nn.Sequential(
                            nn.Conv2d(
                                chn,
                                feat_channels,
                                kernel_size,
                                padding=kernel_size // 2,
                                bias=False,
                            ),
                            nn.BatchNorm2d(feat_channels),
                            nn.Hardswish(inplace=True),
                        )
                    )
            self.cls_convs.append(cls_convs)

            if not share_cls_reg:
                reg_convs = nn.ModuleList()
                for i in range(stacked_convs):
                    chn = in_channels if i == 0 else feat_channels
                    if use_depthwise:
                        reg_convs.append(
                            DepthwiseSeparableConv(chn, feat_channels, kernel_size)
                        )
                    else:
                        reg_convs.append(
                            nn.Sequential(
                                nn.Conv2d(
                                    chn,
                                    feat_channels,
                                    kernel_size,
                                    padding=kernel_size // 2,
                                    bias=False,
                                ),
                                nn.BatchNorm2d(feat_channels),
                                nn.Hardswish(inplace=True),
                            )
                        )
                self.reg_convs.append(reg_convs)
            else:
                self.reg_convs.append(nn.ModuleList())

            if share_cls_reg:
                self.gfl_cls.append(
                    nn.Conv2d(
                        feat_channels,
                        num_classes + 4 * (reg_max + 1),
                        1,
                        padding=0,
                    )
                )
                self.gfl_reg.append(None)  # type: ignore[arg-type]
            else:
                self.gfl_cls.append(nn.Conv2d(feat_channels, num_classes, 1, padding=0))
                self.gfl_reg.append(
                    nn.Conv2d(feat_channels, 4 * (reg_max + 1), 1, padding=0)
                )

        self.integral = Integral(reg_max)

        if hasattr(self, "register_load_state_dict_pre_hook"):
            self.register_load_state_dict_pre_hook(  # type: ignore[no-untyped-call]
                task_model_helpers.picodet_gfl_cls_reuse_or_reinit_hook
            )
        else:
            # Backwards compatibility for PyTorch <= 2.4
            self._register_load_state_dict_pre_hook(  # type: ignore[no-untyped-call]
                task_model_helpers.picodet_gfl_cls_reuse_or_reinit_hook,
                with_module=True,
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        bias_init = -4.6
        for gfl_cls in self.gfl_cls:
            if (
                gfl_cls is not None
                and isinstance(gfl_cls, nn.Conv2d)
                and gfl_cls.bias is not None
            ):
                nn.init.constant_(gfl_cls.bias[: self.num_classes], bias_init)

    def forward(self, feats: Sequence[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Forward pass.

        Args:
            feats: List of feature tensors from the neck, one per stride level.

        Returns:
            Tuple of (cls_scores, bbox_preds) where each is a list of tensors
            per level:
            - cls_scores: (B, num_classes, H, W)
            - bbox_preds: (B, 4*(reg_max+1), H, W)
        """
        cls_scores = []
        bbox_preds = []

        for level_idx, x in enumerate(feats):
            cls_convs_level = self.cls_convs[level_idx]
            gfl_cls = self.gfl_cls[level_idx]

            cls_feat = x
            if isinstance(cls_convs_level, nn.ModuleList):
                for conv in cls_convs_level:
                    cls_feat = conv(cls_feat)

            if self.share_cls_reg:
                out = gfl_cls(cls_feat)
                cls_score, bbox_pred = torch.split(
                    out, [self.num_classes, 4 * (self.reg_max + 1)], dim=1
                )
            else:
                reg_convs_level = self.reg_convs[level_idx]
                gfl_reg = self.gfl_reg[level_idx]

                reg_feat = x
                if isinstance(reg_convs_level, nn.ModuleList):
                    for conv in reg_convs_level:
                        reg_feat = conv(reg_feat)

                cls_score = gfl_cls(cls_feat)
                bbox_pred = gfl_reg(reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        return cls_scores, bbox_preds

    def decode_predictions(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode predictions to boxes in pixel coordinates.

        Args:
            cls_scores: List of classification scores per level.
            bbox_preds: List of bbox distribution predictions per level.

        Returns:
            Tuple of:
            - all_points: (sum(H*W), 4) [cx, cy, stride_w, stride_h]
            - all_cls_scores: (B, sum(H*W), num_classes)
            - all_decoded_bboxes: (B, sum(H*W), 4) in xyxy pixel coords
        """
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]

        points_list: list[Tensor] = []
        cls_scores_list: list[Tensor] = []
        decoded_bboxes_list: list[Tensor] = []

        for level_idx, (cls_score, bbox_pred) in enumerate(zip(cls_scores, bbox_preds)):
            stride = self.strides[level_idx]
            _, _, h, w = cls_score.shape

            points = generate_grid_points(h, w, stride, device)
            num_points = h * w

            points_with_stride = torch.cat(
                [points, torch.full((num_points, 2), stride, device=device)], dim=-1
            )
            points_list.append(points_with_stride)

            cls_score_reshaped = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, num_points, self.num_classes
            )
            cls_scores_list.append(cls_score_reshaped)

            bbox_pred_reshaped = bbox_pred.permute(0, 2, 3, 1).reshape(
                batch_size, num_points, 4 * (self.reg_max + 1)
            )
            distances = self.integral(bbox_pred_reshaped)
            distances = distances * stride

            points_expanded = points.unsqueeze(0).expand(batch_size, -1, -1)
            decoded_bboxes = distance2bbox(points_expanded, distances)
            decoded_bboxes_list.append(decoded_bboxes)

        all_points = torch.cat(points_list, dim=0)
        all_cls_scores = torch.cat(cls_scores_list, dim=1)
        all_decoded_bboxes = torch.cat(decoded_bboxes_list, dim=1)

        return all_points, all_cls_scores, all_decoded_bboxes
