#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformArgs,
    TaskTransformInput,
    TaskTransformOutput,
)


class ObjectDetectionTransformArgs(TaskTransformArgs):
    pass


class ObjectDetectionTransform(TaskTransform):
    transform_args_cls = ObjectDetectionTransformArgs

    def __call__(self, input: TaskTransformInput) -> TaskTransformOutput:  # type: ignore[empty-body]
        pass
