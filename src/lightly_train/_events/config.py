#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os

POSTHOG_API_KEY: str = os.getenv("LIGHTLY_TRAIN_POSTHOG_KEY", "phc_placeholder")
EVENTS_DISABLED: bool = os.getenv("LIGHTLY_TRAIN_EVENTS_DISABLED", "0") == "1"
