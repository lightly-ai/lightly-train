#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch.nn import Module


class DINOv3ConvNextWrapper(Module):
    def __init__(self, model_name="convnext_tiny"):
        super().__init__()
        # Get the model.
        model_getter = MODEL_NAME_TO_GETTER[model_name]
        model_url = MODEL_NAME_TO_URL[model_name]
        self.backbone = model_getter(weights=model_url)

    def forward(self, x):
        feats = self.backbone.get_intermediate_layers(x, n=3, reshape=True)
        return feats
