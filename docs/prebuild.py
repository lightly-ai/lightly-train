# This script creates file in the docs/source before the build process.

import os
import re
import textwrap
from argparse import ArgumentParser
from pathlib import Path

from lightly_train._commands import common_helpers
from lightly_train._methods.dino.dino_transform import DINOTransformArgs
from lightly_train._methods.distillation.distillation_transform import (
    DistillationTransformArgs,
)
from lightly_train._methods.simclr.simclr_transform import SimCLRTransformArgs

THIS_DIR = Path(__file__).parent.resolve()
DOCS_DIR = THIS_DIR / "source"
PROJECT_ROOT = THIS_DIR.parent
TRANSFORM_ARGS_DIR = DOCS_DIR / "methods" / "_auto"


# inspired by https://github.com/pydantic/pydantic/blob/6f31f8f68ef011f84357330186f603ff295312fd/docs/plugins/main.py#L102-L103
def build_changelog_html(source_dir: Path) -> None:
    """Creates the changelog.html file from the repos main CHANGELOG.md file"""
    header = textwrap.dedent("""
        (changelog)=
        
    """)

    changelog_content = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    # Regex matches everything between "## [Unreleased]"" and the next "## [" but does
    # not capture the "## [" part.
    pattern = r"## \\?\[Unreleased\\?\].*?(?=## \\?\[)"
    changelog_content = re.sub(pattern, "", changelog_content, flags=re.DOTALL).strip()
    changelog_content = header + changelog_content

    # avoid writing file unless the content has changed to avoid infinite build loop
    new_file = source_dir / "changelog.md"
    if (
        not new_file.is_file()
        or new_file.read_text(encoding="utf-8") != changelog_content
    ):
        new_file.write_text(changelog_content, encoding="utf-8")


def dump_transformargs_for_methods(dest_dir: Path) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    # dump transform args for all methods
    for method, transform_args in [
        ("dino", DINOTransformArgs),
        ("simclr", SimCLRTransformArgs),
        ("distillation", DistillationTransformArgs),
    ]:
        # dump transform args
        args = common_helpers.pretty_format_args(transform_args().model_dump())
        # write to file
        with open(dest_dir / f"{method}_transform_args.md", "w") as f:
            f.write("```python\n")
            f.write(args + "\n")
            f.write("```\n")


def main(source_dir: Path) -> None:
    build_changelog_html(source_dir=source_dir)
    dump_transformargs_for_methods(dest_dir=TRANSFORM_ARGS_DIR)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()

    main(source_dir=args.source_dir)
