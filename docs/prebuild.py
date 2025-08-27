# This script creates file in the docs/source before the build process.

import re
import textwrap
from argparse import ArgumentParser
from pathlib import Path
from re import Match

from lightly_train._commands import common_helpers, train_helpers, train_task_helpers
from lightly_train._commands.train_task_helpers import TASK_TRAIN_MODEL_CLASSES
from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
)
from lightly_train._methods import method_helpers
from lightly_train._task_models.dinov2_linear_semantic_segmentation.train_model import (
    DINOv2LinearSemanticSegmentationTrain,
)

THIS_DIR = Path(__file__).parent.resolve()
DOCS_DIR = THIS_DIR / "source"
PROJECT_ROOT = THIS_DIR.parent
SOURCE_AUTO_DIR = DOCS_DIR / "_auto"
METHODS_AUTO_ARGS_DIR = DOCS_DIR / "methods" / "_auto"


# inspired by https://github.com/pydantic/pydantic/blob/6f31f8f68ef011f84357330186f603ff295312fd/docs/plugins/main.py#L102-L103
def build_changelog_html(source_dir: Path) -> None:
    """Creates the changelog.html file from the repos main CHANGELOG.md file"""
    header = textwrap.dedent("""
        (changelog)=
        
    """)

    changelog_content = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    # Remove the "Unreleased" section.
    # Regex matches everything between "## [Unreleased]"" and the next "## [" but does
    # not capture the "## [" part.
    pattern = r"## \\?\[Unreleased\\?\].*?(?=## \\?\[)"
    changelog_content = re.sub(pattern, "", changelog_content, flags=re.DOTALL).strip()

    # Add version targets.
    # Adds a `(changelog-<version>)=` target above each version header. This is needed
    # because sphinx otherwise generates generic `id1`, `id2`, etc. targets for the
    # version headers.
    version_pattern = r"## \\?\[(\d+\.\d+\.\d+)\\?\] - \d+-\d+-\d+\n"

    def add_version_target(match: Match):
        version = match.group(1)
        version_with_target = textwrap.dedent(f"""
            (changelog-{version.replace(".", "-")})=

            {match.group(0)}
        """)
        return version_with_target

    changelog_content = re.sub(
        version_pattern,
        add_version_target,
        changelog_content,
        flags=re.DOTALL,
    )

    # Add header.
    changelog_content = header + changelog_content

    # avoid writing file unless the content has changed to avoid infinite build loop
    new_file = source_dir / "changelog.md"
    if (
        not new_file.is_file()
        or new_file.read_text(encoding="utf-8") != changelog_content
    ):
        new_file.write_text(changelog_content, encoding="utf-8")


def dump_transform_args_for_methods(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for method in method_helpers.list_methods():
        if method in {"distillationv1", "distillationv2"}:
            continue
        transform_args = train_helpers.get_transform_args(
            method=method, transform_args=None
        )
        args = common_helpers.pretty_format_args(transform_args.model_dump())
        # write to file
        with open(dest_dir / f"{method}_transform_args.md", "w") as f:
            f.write("```json\n")
            f.write(args + "\n")
            f.write("```\n")


def dump_transform_args_for_tasks(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for train_model_cls in TASK_TRAIN_MODEL_CLASSES:
        if train_model_cls in {DINOv2LinearSemanticSegmentationTrain}:
            continue
        train_transform_args = train_model_cls.train_transform_cls.transform_args_cls(
            ignore_index=MaskSemanticSegmentationDataArgs.ignore_index
        )
        val_transform_args = train_model_cls.val_transform_cls.transform_args_cls(
            ignore_index=MaskSemanticSegmentationDataArgs.ignore_index
        )
        train_args = train_task_helpers.pretty_format_args(
            train_transform_args.model_dump()
        )
        val_args = train_task_helpers.pretty_format_args(
            val_transform_args.model_dump()
        )
        name = train_model_cls.__name__.lower()
        # write to file
        with open(dest_dir / f"{name}_train_transform_args.md", "w") as f:
            f.write("```json\n")
            f.write(train_args + "\n")
            f.write("```\n")
        with open(dest_dir / f"{name}_val_transform_args.md", "w") as f:
            f.write("```json\n")
            f.write(val_args + "\n")
            f.write("```\n")


def dump_method_args(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    # dump transform args for all methods
    for method in method_helpers.list_methods():
        if method in {"distillationv1", "distillationv2"}:
            continue
        method_args = method_helpers.get_method_cls(method).method_args_cls()()
        args = common_helpers.pretty_format_args(method_args.model_dump())
        # write to file
        with open(dest_dir / f"{method}_method_args.md", "w") as f:
            f.write("```json\n")
            f.write(args + "\n")
            f.write("```\n")


def main(source_dir: Path) -> None:
    build_changelog_html(source_dir=source_dir)
    # Methods
    dump_transform_args_for_methods(dest_dir=METHODS_AUTO_ARGS_DIR)
    dump_method_args(dest_dir=METHODS_AUTO_ARGS_DIR)
    # Tasks
    dump_transform_args_for_tasks(dest_dir=SOURCE_AUTO_DIR)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()

    main(source_dir=args.source_dir)
