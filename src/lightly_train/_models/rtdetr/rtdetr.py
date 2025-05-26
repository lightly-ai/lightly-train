from __future__ import annotations

from torch.nn import Module, AdaptiveAvgPool2d
from torch import Tensor

class RTDETRModelWrapper(Module):
    def __init__(self, model: Module):
        super().__init__()
        self._model = model
        self._pool = AdaptiveAvgPool2d((1, 1))

    def get_model(self) -> Module:
        return self._model

    def forward_features(self, x: Tensor) -> dict[str, Tensor]:
        features = self._model.backbone(x)[-1]
        return {"features": features}

    def forward_pool(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        return {"pooled_features": self._pool(x["features"])}

    def feature_dim(self) -> int:
        return self._model.backbone.out_channels[-1]