#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__version__ = "0.0.1"

from lightly_train._modules.teachers.dinov2.build_teacher import get_dinov2_teacher

__all__ = ["get_dinov2_teacher"]
