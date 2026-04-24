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
"""Copyright (c) 2024 The D-FINE Authors. All Rights Reserved."""

from __future__ import annotations

import torch
from torch import Tensor

from lightly_train._task_models.object_detection_components.box_ops import (
    box_xyxy_to_cxcywh,
)


def weighting_function(
    reg_max: int,
    up: Tensor,
    reg_scale: Tensor,
    deploy: bool = False,
) -> Tensor:
    """Generate the non-uniform weighting function for D-FINE box regression."""
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor(values, dtype=up.dtype, device=up.device)

    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
    values = (
        [-upper_bound2]
        + left_values
        + [torch.zeros_like(up[0][None])]
        + right_values
        + [upper_bound2]
    )
    return torch.cat(values, 0)


def distance2bbox(points: Tensor, distance: Tensor, reg_scale: Tensor) -> Tensor:
    """Decode D-FINE edge distances into ``cxcywh`` boxes."""
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (
        points[..., 2] / reg_scale
    )
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (
        points[..., 3] / reg_scale
    )
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (
        points[..., 2] / reg_scale
    )
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (
        points[..., 3] / reg_scale
    )

    bboxes = torch.stack([x1, y1, x2, y2], -1)
    return box_xyxy_to_cxcywh(bboxes)
