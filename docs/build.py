# This script creates the build/index.html and build/versions.html files.
# build/index.html redirects to build/stable/index.html.
# build/versions.html lists all available documentation versions.

import textwrap
from argparse import ArgumentParser
from pathlib import Path


def build_index_html(build_dir: Path) -> None:
    """Creates the main index.html file that redirects to the stable version."""
    html = textwrap.dedent("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="0;url=stable/index.html">
        <title>Redirecting...</title>
    </head>
    <body>
        <p>If you are not redirected, click <a href="stable/index.html">here</a>.</p>
    </body>
    </html>
    """)

    with open(build_dir / "index.html", "w") as f:
        f.write(html)


def build_versions_html(build_dir: Path) -> None:
    """Creates the versions.html file that lists all available versions."""

    header = textwrap.dedent("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LightlyTrain Documentation</title>
    </head>
    <body>
        <h1>LightlyTrain Documentation</h1>
        <ul>
    """)
    footer = textwrap.dedent("""
        </ul>
    </body>
    </html>
    """)

    html = header
    versions = sorted(
        [path for path in build_dir.iterdir() if path.is_dir()], reverse=True
    )
    for version in versions:
        html += f'        <li><a href="{version.name}">{version.name}</a></li>\n'
    html += footer

    with open(build_dir / "versions.html", "w") as f:
        f.write(html)


def main(build_dir: Path) -> None:
    build_index_html(build_dir=build_dir)
    build_versions_html(build_dir=build_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()

    main(build_dir=args.build_dir)
