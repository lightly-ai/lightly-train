#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
class LightlyTrainError(Exception):
    pass


class UnknownModelError(LightlyTrainError):
    pass


class ConfigError(LightlyTrainError):
    pass


class ConfigUnknownKeyError(ConfigError):
    pass


class ConfigValidationError(ConfigError):
    pass


class ConfigMissingKeysError(ConfigError):
    pass


class UnresolvedAutoError(LightlyTrainError):
    pass
