# This script creates file in the docs/source before the build process.

import re
import textwrap
from argparse import ArgumentParser
from pathlib import Path

from lightly_train._commands import common_helpers, train_helpers
from lightly_train._methods import method_helpers

THIS_DIR = Path(__file__).parent.resolve()
DOCS_DIR = THIS_DIR / "source"
PROJECT_ROOT = THIS_DIR.parent
METHODS_AUTO_ARGS_DIR = DOCS_DIR / "methods" / "_auto"


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
    dump_transform_args_for_methods(dest_dir=METHODS_AUTO_ARGS_DIR)
    dump_method_args(dest_dir=METHODS_AUTO_ARGS_DIR)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()

    main(source_dir=args.source_dir)
