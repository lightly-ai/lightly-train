#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Single source of truth for the LightlyTrain license notice.

The notice is embedded into training checkpoints and into exported ONNX models so
that both carry the same license terms.
"""

from __future__ import annotations

LICENSE_INFO = (
    "LightlyTrain License Notice\n"
    "\n"
    "Model training and inference in commercial settings require a valid Commercial License.\n"
    "If you are using LightlyTrain for open-source (AGPL-3.0) or under a Free Community License,\n"
    "please ensure your usage complies with the respective terms.\n"
    "See https://docs.lightly.ai/train/stable/index.html#license for more details.\n"
    "Contact us at https://www.lightly.ai/contact to discuss the best licensing option for your use case."
)
