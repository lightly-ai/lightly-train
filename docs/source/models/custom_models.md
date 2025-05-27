(custom-models)=

# Custom Models

```{note}
Training custom models from the command line or with docker is not yet supported.
```

Lightly**Train** supports training custom models. This requires writing a small wrapper
around your model to implement the necessary methods. The wrapper must be a subclass
of `torch.nn.Module` and implement the following methods:

- `get_model(self) -> Module`

  Returns the unwrapped model. This method is used for exporting the model.

- `forward_features(self, x: Tensor) -> Dict[str, Any]`

  Forward pass of the model that extracts features without pooling them.

- `forward_pool(self, x: Dict[str, Any]) -> Dict[str, Any]`

  Forward pass of the pooling layer that pools the features extracted by
  `forward_features`.

- `feature_dim(self) -> int`

  Dimension of output features (channels) of the features returned by `forward_features`
  and `forward_pool`.

The methods are described in more detail in the template below:

```python
from typing import Any, Dict

from torch import Tensor
from torch.nn import Module

import lightly_train


class MyModelWrapper(Module):
    def __init__(self, model: Module):
        super().__init__()
        self._model = model     # Pass your model here

    def get_model(self) -> Module:
        """Returns the unwrapped model."""
        return self._model

    def forward_features(self, x: Tensor) -> Dict[str, Any]:
        """Forward pass to extract features from images.

        Implement the feature extraction forward pass here. This method takes images
        as input and extracts features from them. In most cases this method should
        call your model's backbone or encoder. The method should not pool the final
        features and should not pass them through any classification, detection, or
        other heads.

        Args:
            x: Batch of images with shape (B, 3, H_in, W_in).
        
        Returns:
            Dict with a "features" entry containing the features tensor with shape
            (B, feature_dim, H_out, W_out). Add any other entries to the dict if they
            are needed in the forward_pool method. For example, for transformer models
            you might want to return the class token as well.
        """
        features = ...
        return {"features": features}

    def forward_pool(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass to pool features extracted by forward_features.

        Implement the pooling layer forward pass here. This method must take the
        output of the forward_features method as input and pool the features.

        Args:
            x: 
                Dict with a "features" entry containing the features tensor with shape
                (B, feature_dim, H_in, W_in).

        Returns:
            Dict with a "pooled_features" entry containing the pooled features tensor
            with shape (B, feature_dim, H_out, W_out). H_out and W_out are usually 1.
        """
        pooled_features = ...
        return {"pooled_features": pooled_features}

    def feature_dim(self) -> int:
        """Return the dimension of output features.

        This method must return the dimension of output features of the forward_features
        and forward_pool methods.
        """
        return 2048

if __name__ == "__main__":
    model = ... # Instatiate the model you want to train
    wrapped_model = MyModelWrapper(model) # Wrap the model

    lightly_train.train(
        out="out/my_experiment",
        data="my_data_dir",
        model=wrapped_model,
    )
```

The wrapped model will be called as follows inside Lightly**Train**:

```python skip_ruff
embedding_layer = EmbeddingLayer(input_dim=wrapped_model.feature_dim())

images = load_batch()
x = transform(images)   # Augment and convert images to tensor
x = wrapped_model.forward_features(x)
x = wrapped_model.forward_pool(x)
x = embedding_layer(x)
embeddings = x.flatten(start_dim=1)
```

Some [SSL methods](#methods) do not call the `forward_pool` method and
only use the unpooled features. In this case, the embedding layer is applied directly to
the output of `forward_features`.

## Example

The following example demonstrates how to write a wrapper for a torchvision ResNet-18
model.

```python
from typing import Any, Dict

from torch import Tensor
from torch.nn import Module
from torchvision.models import resnet18

import lightly_train


class MyModelWrapper(Module):
    def __init__(self, model: Module):
        super().__init__()
        self._model = model     # Pass your model here

    def get_model(self) -> Module:
        return self._model

    def forward_features(self, x: Tensor) -> Dict[str, Any]:
        # Torchvision ResNet has no method for only extracting features. We have to
        # call the intermediate layers of the model individually.
        # Note that we skip the final average pooling and fully connected classification
        # layer.
        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)

        x = self._model.layer1(x)
        x = self._model.layer2(x)
        x = self._model.layer3(x)
        x = self._model.layer4(x)
        return {"features": x}
    
    def forward_pool(self, x: Dict[str, Any]) -> Dict[str, Any]:
        # Here we call the average pooling layer of the model to pool the features.
        x = self._model.avgpool(x["features"])
        return {"pooled_features": x}

    def feature_dim(self) -> int:
        # ResNet-18 has 512 output features after the last convolutional layer.
        return 512


if __name__ == "__main__":
    model = resnet18()
    wrapped_model = MyModelWrapper(model)

    lightly_train.train(
        out="out/my_experiment",
        data="my_data_dir",
        model=wrapped_model,
    )
```

After pretraining completes, you can load the model as follows:

```python
import torch
from torchvision.models import resnet18

model = resnet18()
model.load_state_dict(torch.load("out/my_experiment/exported_models/exported_last.pt"), weights_only=True)
```
