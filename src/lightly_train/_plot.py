#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from PIL import ImageDraw, ImageFont
from PIL.Image import Image as PILImage
from torchvision.transforms import functional as torchvision_functional

from lightly_train.types import Batch


def plot_example_augmentations(train_batch: Batch, max_examples: int = 10) -> PILImage:
    views = train_batch["views"]
    n_views = len(views)
    batch1 = views[0]
    n_examples = min(max_examples, len(batch1))

    # Set the grid height and width to be a bit larger than the largest view.
    max_view_height = max([view.shape[2] for view in views])
    max_view_width = max([view.shape[3] for view in views])
    grid_height = max_view_height + 10
    grid_width = max_view_width + 10

    # Additional space for headers
    header_height = 30  # Extra space for column headers
    header_width = 100  # Extra space for row headers

    # Create a tensor for combined image data.
    combined_aug_tensor = torch.ones(
        size=(
            3,
            header_height + n_examples * grid_height,
            header_width + n_views * grid_width,
        ),
        device="cpu",
    )
    # Set the tensor to a nan value to allow setting it to white later.
    # Setting it to white directly would not work, as the tensor will be normalized.
    combined_aug_tensor *= torch.nan

    # Fill the images into the tensor.
    for i_column, batch in enumerate(views):
        for i_row, image_tensor in enumerate(batch):
            if i_row >= n_examples:
                break
            # TODO(Thomas,07/25): Fix swapped x and y.
            x_mid = header_height + i_row * grid_height + grid_height // 2
            x_start = x_mid - image_tensor.shape[1] // 2
            x_end = x_mid + image_tensor.shape[1] // 2
            x_end += image_tensor.shape[1] % 2  # Ensure working with odd sizes
            y_mid = header_width + i_column * grid_width + grid_width // 2
            y_start = y_mid - image_tensor.shape[2] // 2
            y_end = y_mid + image_tensor.shape[2] // 2
            y_end += image_tensor.shape[2] % 2  # Ensure working with odd sizes
            combined_aug_tensor[
                :,
                x_start:x_end,
                y_start:y_end,
            ] = image_tensor.cpu()

    # Note: Getting the normalization specific to the method is not trivial,
    # as it depends on the transform. See
    # https://github.com/lightly-ai/lightly-train/pull/100#discussion_r1750338176
    # Thus normalize using the min/max per channel and ignore the nan placeholders.
    combined_aug_tensor_nan_as_max = combined_aug_tensor.nan_to_num(1e10)
    min_per_channel = combined_aug_tensor_nan_as_max.amin(dim=(1, 2)).view(3, 1, 1)
    combined_aug_tensor_nan_as_min = combined_aug_tensor.nan_to_num(-1e10)
    max_per_channel = combined_aug_tensor_nan_as_min.amax(dim=(1, 2)).view(3, 1, 1)
    combined_aug_tensor -= min_per_channel
    combined_aug_tensor /= max_per_channel - min_per_channel

    # Set the tensor to white where it was set to nan before.
    combined_aug_tensor = combined_aug_tensor.nan_to_num(1.0)

    # Convert to PIL and write the row and column headers.
    pil_image: PILImage = torchvision_functional.to_pil_image(combined_aug_tensor)
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        # Size argument was added in Pillow 10.1.0.
        font = ImageFont.load_default()
    for i_row in range(n_examples):
        row_header_text = f"image_{i_row + 1}"
        text_position = (
            10,
            header_height + i_row * grid_height + grid_height // 2 - 10,
        )
        draw.text(text_position, row_header_text, fill=(0, 0, 0), font=font)
    for i_column in range(n_views):
        column_header_text = f"view_{i_column + 1}"
        text_position = (
            header_width + i_column * grid_width + grid_width // 2 - 20,
            5,
        )
        draw.text(text_position, column_header_text, fill=(0, 0, 0), font=font)

    return pil_image
