#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("lightly_train_api.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
