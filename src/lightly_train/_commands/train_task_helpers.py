#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import json
from contextlib import contextmanager
from json import JSONEncoder
from pathlib import Path
from typing import Any, Generator

from lightning_fabric import Fabric
from lightning_fabric import utilities as fabric_utilities

from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataset,
    MaskSemanticSegmentationDatasetArgs,
)
from lightly_train.types import PathLike


@contextmanager
def rank_zero_unshared_only(
    fabric: Fabric, path: PathLike
) -> Generator[None, None, None]:
    """The code under this context manager is only executed by rank zero.

    If the filesystem at path is shared, the code is executed only on global rank zero.
    If the filesystem at path is not shared, the code is executed on every local rank zero.
    """
    is_shared = fabric_utilities.is_shared_filesystem(
        strategy=fabric.strategy, path=path
    )
    local = not is_shared
    with fabric.rank_zero_first(local=local):
        if fabric.is_global_zero or (local and fabric.local_rank == 0):
            yield
        else:
            return


def get_out_dir(
    fabric: Fabric, out: PathLike, resume_interrupted: bool, overwrite: bool
) -> Path:
    out_dir = Path(out).resolve()
    with rank_zero_unshared_only(fabric=fabric, path=out_dir):
        if out_dir.exists():
            if not out_dir.is_dir():
                raise ValueError(f"Output '{out_dir}' is not a directory!")

            dir_not_empty = any(out_dir.iterdir())

            if dir_not_empty and (not (resume_interrupted or overwrite)):
                raise ValueError(
                    f"Output '{out_dir}' is not empty! Set overwrite=True to overwrite "
                    "the directory or resume_interrupted=True to resume training from "
                    "an interrupted or crashed run. "
                    "See https://docs.lightly.ai/lightly-train/usage/cli.html#resume-training "
                    "for more information on how to resume training."
                )
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


class PrettyFormatArgsJSONEncoder(JSONEncoder):
    """Custom JSON encoder to pretty format the output."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        try:
            return super().default(obj)
        except TypeError:
            # Return class name for objects that cannot be serialized
            return obj.__class__.__name__


def pretty_format_args(args: dict[str, Any], indent: int = 4) -> str:
    return json.dumps(
        args, indent=indent, sort_keys=True, cls=PrettyFormatArgsJSONEncoder
    )


def get_dataset(
    dataset_args: MaskSemanticSegmentationDatasetArgs,
) -> MaskSemanticSegmentationDataset:
    # TODO(Guarin, 07/25): MMAP filenames.
    filenames = list(dataset_args.list_image_filenames())
    dataset_cls = dataset_args.get_dataset_cls()
    return dataset_cls(
        dataset_args=dataset_args,
        image_filenames=filenames,
        # TODO(Guarin, 07/25): Add transforms
        transform=lambda x: x,  # type: ignore[arg-type]
    )
