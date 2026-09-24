"""Indexes the example images without annotations and opens LightlyStudio.

Started by run_demo.sh, which also runs the LightlyTrain API. Annotating the images and
training on them is the job of the two plugin operators.
"""

from __future__ import annotations

import lightly_studio as ls

DATASET_NAME = "demo"


def main() -> None:
    dataset_path = ls.utils.download_example_dataset(download_dir="dataset_examples")
    dataset = ls.ImageDataset.load_or_create(name=DATASET_NAME)
    dataset.add_images_from_path(path=f"{dataset_path}/coco_subset_128_images/images")
    ls.start_gui(open_browser=True)


if __name__ == "__main__":
    main()
