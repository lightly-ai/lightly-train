#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_from_url(url: str, destination: Path, timeout: float = 10.0) -> None:
    """
    Downloads a file from `url` to `destination` with timeout, progress bar, and error handling.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            total = response.length
            with tqdm(
                total=total, unit="B", unit_scale=True, desc="Downloading", ncols=80
            ) as t:
                with open(destination, "wb") as out_file:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        t.update(len(chunk))
    except (URLError, HTTPError, TimeoutError, Exception) as e:
        raise RuntimeError(f"Download from '{url}' failed: '{e}'")
