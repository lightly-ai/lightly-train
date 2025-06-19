#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any, Sequence

import numpy as np
from albumentations import ImageOnlyTransform
from numpy.typing import NDArray


class ChannelDrop(ImageOnlyTransform):  # type: ignore[misc]
    def __init__(
        self,
        num_channels_keep: int = 3,
        prob_keep: Sequence[float] = (1.0, 1.0, 1.0),
        p: float = 1.0,
    ):
        """
        Randomly drops channels from an image. Different from Albumentations
        ChannelDropout as it does not set channels to zero but removes them completely.

        Args:
            num_channels_keep:
                Number of channels to keep in the image.
            prob_keep:
                Probability for each channel to be kept.
            p:
                Probability of applying the transform.
        """
        super().__init__(p=p)
        self.num_channels_keep = num_channels_keep
        self.prob_keep = list(prob_keep)

        if num_channels_keep < 1:
            raise ValueError(
                f"num_channels_keep must be at least 1, got {num_channels_keep}."
            )
        if any(p < 0 for p in self.prob_keep):
            raise ValueError(
                f"All probabilities in prob_keep must be non-negative, got {self.prob_keep}."
            )
        if sum(p > 0 for p in self.prob_keep) < self.num_channels_keep:
            raise ValueError(
                "At least num_channels_keep channels must have a non-zero probability "
                f"to be kept, got {self.num_channels_keep} and {self.prob_keep}."
            )

    def apply(
        self, img: NDArray[np.uint8], **params: dict[str, Any]
    ) -> NDArray[np.uint8]:
        """Apply the channel drop transform to the image.

        Args:
            img: Input image as numpy array with shape (H, W, C).

        Returns:
            Image with selected channels kept, dropped channels removed completely.
        """
        num_channels = img.shape[2]

        if len(self.prob_keep) != num_channels:
            raise RuntimeError(
                f"Length of prob_keep ({len(self.prob_keep)}) must match "
                f"number of image channels ({num_channels})"
            )

        if self.num_channels_keep >= num_channels:
            return img

        prob_keep = np.array(self.prob_keep)
        prob_keep = prob_keep / prob_keep.sum()  # Normalize

        # Select channels to keep based on probabilities
        channels_to_keep = np.random.choice(
            num_channels, size=self.num_channels_keep, replace=False, p=prob_keep
        )

        # Sort the channels to maintain order
        channels_to_keep = np.sort(channels_to_keep)

        # Return only the selected channels (remove dropped channels completely)
        result = img[:, :, channels_to_keep]

        return result

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return list of arguments used in __init__ for serialization."""
        return ("num_channels_keep", "prob_keep")
