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
"""Geometry helpers for LT-DETR object detection models."""

from __future__ import annotations


def ltdetr_image_size_divisor(patch_size: int) -> int:
    """Return the LT-DETR image-size divisor for a patch size.

    LT-DETR uses a feature map that is 2x smaller than the ViT patch grid, so
    image sizes must be divisible by 2 * patch_size.
    """

    return 2 * patch_size
