#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.utils.data import Dataset

from lightly_train.types import MaskSemanticSegmentationDatasetItem


class MaskSemanticSegmentationDataset(Dataset[MaskSemanticSegmentationDatasetItem]):
    def __getitem__(self, index: int) -> MaskSemanticSegmentationDatasetItem:
        return {
            "image_filename": "abc.jpg",
            "image": torch.rand(3, 224, 224, dtype=torch.uint8),
            "mask": torch.rand(5, 224, 224, dtype=torch.uint8),
        }
