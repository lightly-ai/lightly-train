#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torchvision.tv_tensors as tv_tensors
from albumentations import ImageOnlyTransform
from torchvision.transforms import v2

from lightly_train.types import NDArrayImage


class ChannelDrop(ImageOnlyTransform):  # type: ignore[misc]
    def __init__(
        self,
        num_channels_keep: int,
        weight_drop: Sequence[float],
        p: float = 1.0,
    ):
        """
        Randomly drops channels from an image. Different from Albumentations
        ChannelDropout as it does not set channels to zero but removes them completely.

        Args:
            num_channels_keep:
                Number of channels to keep in the image.
            weight_drop:
                Weight for each channel to be dropped. 0 means never dropped,
                higher values mean higher probability of being dropped.
            p:
                Probability of applying the transform.
        """
        super().__init__(p=p)
        self.num_channels_keep = num_channels_keep
        self.weight_drop = list(weight_drop)

        if num_channels_keep < 1:
            raise ValueError(
                f"num_channels_keep must be at least 1, got {num_channels_keep}."
            )
        if any(w < 0 for w in self.weight_drop):
            raise ValueError(
                f"All weights in weight_drop must be non-negative, got {self.weight_drop}."
            )
        if sum(w == 0 for w in self.weight_drop) > self.num_channels_keep:
            raise ValueError(
                "At most num_channels_keep channels can have zero weight "
                f"to guarantee they can be kept, got {self.num_channels_keep} and "
                f"{self.weight_drop}."
            )

        # Normalize weights to probabilities
        weight_array = np.array(self.weight_drop)
        self._prob_drop = weight_array / weight_array.sum()

    def apply(self, img: NDArrayImage, **params: dict[str, Any]) -> NDArrayImage:
        """Apply the channel drop transform to the image.

        Args:
            img: Input image as numpy array with shape (H, W, C).

        Returns:
            Image with selected channels kept, dropped channels removed completely.
        """
        num_channels = img.shape[2]

        if self.num_channels_keep == num_channels:
            return img
        elif self.num_channels_keep > num_channels:
            raise ValueError(
                f"num_channels_keep ({self.num_channels_keep}) cannot be greater "
                f"than the number of channels in the image ({num_channels})."
            )

        if len(self.weight_drop) != num_channels:
            raise RuntimeError(
                f"Length of weight_drop ({len(self.weight_drop)}) must match "
                f"number of image channels ({num_channels})"
            )

        channels_to_drop = np.random.choice(
            num_channels,
            size=num_channels - self.num_channels_keep,
            replace=False,
            p=self._prob_drop,
        )
        channels_to_keep = np.sort(
            np.setdiff1d(np.arange(num_channels), channels_to_drop)
        )
        result = img[:, :, channels_to_keep]
        return result

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return list of arguments used in __init__ for serialization."""
        return ("num_channels_keep", "weight_drop")


class ChannelDropTV(v2.Transform):
    def __init__(
        self,
        num_channels_keep: int,
        weight_drop: Sequence[float],
        num_channels: int = 3,
    ) -> None:
        """
        Randomly drops channels from an image. Different from Albumentations
        ChannelDropout as it does not set channels to zero but removes them completely.

        Args:
            num_channels_keep:
                Number of channels to keep in the image.
            weight_drop:
                Weight for each channel to be dropped. 0 means never dropped,
                higher values mean higher probability of being dropped.
            p:
                Probability of applying the transform.
        """
        super().__init__()
        self.num_channels_keep = num_channels_keep
        self.num_channels = num_channels
        self.weight_drop = list(weight_drop)

        if num_channels_keep < 1:
            raise ValueError(
                f"num_channels_keep must be at least 1, got {num_channels_keep}."
            )
        if any(w < 0 for w in self.weight_drop):
            raise ValueError(
                f"All weights in weight_drop must be non-negative, got {self.weight_drop}."
            )
        if sum(w == 0 for w in self.weight_drop) > self.num_channels_keep:
            raise ValueError(
                "At most num_channels_keep channels can have zero weight "
                f"to guarantee they can be kept, got {self.num_channels_keep} and "
                f"{self.weight_drop}."
            )

        weight_array = torch.tensor(self.weight_drop, dtype=torch.float32)
        self._prob_drop = weight_array / weight_array.sum()

    def make_params(self, flat_inputs: list[Any]) -> dict[str, torch.Tensor]:
        if self.num_channels_keep == self.num_channels:
            return {"channels_to_keep": torch.arange(self.num_channels)}

        channels_to_drop = torch.multinomial(
            self._prob_drop,
            num_samples=self.num_channels - self.num_channels_keep,
            replacement=False,
        )
        all_channels = torch.arange(self.num_channels)
        mask = torch.isin(all_channels, channels_to_drop)
        channels_to_keep = torch.sort(all_channels[~mask])[0]
        return {"channels_to_keep": channels_to_keep}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, tv_tensors.Image):
            if inpt.ndim < 3:
                return inpt
            channels_to_keep = params["channels_to_keep"]
            if channels_to_keep is None:
                return inpt
            if inpt.shape[0] != len(self.weight_drop):
                return inpt
            return tv_tensors.Image(torch.index_select(inpt, 0, channels_to_keep))
        return inpt
