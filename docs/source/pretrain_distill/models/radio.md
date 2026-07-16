(models-radio)=

# NVIDIA RADIO

This page describes how to use NVIDIA RADIO backbones with LightlyTrain. RADIO models
are loaded from [NVlabs/RADIO](https://github.com/NVlabs/RADIO) through Torch Hub; no
NVIDIA source code is vendored in LightlyTrain.

Install the optional runtime dependencies before using a RADIO model:

```bash
pip install 'lightly-train[radio]'
```

## Usage

RADIO expects three-channel image tensors with values in `[0, 1]`; it applies its own
input conditioning. Configure every corresponding LightlyTrain transform with identity
normalization:

```python
transform_args = {
    "normalize": {"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
}
```

Input height and width must each be divisible by the selected model's resolution step.
Choose a compatible `image_size`; LightlyTrain raises an error instead of resizing the
input silently.

The integration pins Torch Hub source to RADIO commit
`c0f37017930e9dda53f93424cf4bf39fc51f287e`. Advanced users may override it with
`model_args={"hub_ref": "<commit-or-tag>"}`. Downloads are managed by Torch Hub.

## Supported Models

- AM-RADIO: `radio/radio_v1`
- RADIOv2.5: `radio/radio_v2.5-b`, `radio/radio_v2.5-l`, `radio/radio_v2.5-h`,
  `radio/radio_v2.5-h-norm`, `radio/radio_v2.5-g`
- C-RADIOv3: `radio/c-radio_v3-b`, `radio/c-radio_v3-l`, `radio/c-radio_v3-h`,
  `radio/c-radio_v3-g`
- C-RADIOv4: `radio/c-radio_v4-so400m`, `radio/c-radio_v4-h`

All RADIO models currently implement LightlyTrain's regular `ModelWrapper` interface.
They expose RADIO's final spatial feature map and use global average pooling. Multiscale
intermediate features, Torch Hub adaptors, text encoding, and custom necks are not yet
supported.

## Licensing

The RADIO source repository and model weights are external to LightlyTrain and retain
their upstream terms. NVIDIA states that AM-RADIO and RADIO models are governed by the
[RADIO source license](https://github.com/NVlabs/RADIO/blob/main/LICENSE), which is
non-commercial. NVIDIA states that C-RADIO models are governed by the
[NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf).
Review and accept the applicable upstream terms before downloading or using a model.
