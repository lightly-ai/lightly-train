(models-radio)=

# NVIDIA RADIO

This page describes how to use NVIDIA C-RADIO backbones with LightlyTrain. The C-RADIO
runtime code is vendored from NVIDIA's official Hugging Face releases, so LightlyTrain
does not download or execute model source code through Torch Hub.

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

LightlyTrain downloads the selected pretrained checkpoint from NVIDIA on first use. The
vendored runtime is pinned to the C-RADIOv3 and C-RADIOv4 Hugging Face releases.

## Supported Models

- C-RADIOv3: `radio/c-radio_v3-b`, `radio/c-radio_v3-l`, `radio/c-radio_v3-h`,
  `radio/c-radio_v3-g`
- C-RADIOv4: `radio/c-radio_v4-so400m`, `radio/c-radio_v4-h`

All C-RADIO models implement LightlyTrain's regular `ModelWrapper` interface. They
expose NVIDIA's summary representation as `cls_token`, use it for pooled embeddings, and
expose the final NCHW patch-token map as `features`. Multiscale intermediate features,
adaptors, text encoding, and custom necks are not supported.

## Licensing

The vendored C-RADIO code and model weights retain their upstream terms. NVIDIA states
that C-RADIO models are governed by the
[NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf).
Review and accept these terms before downloading or using a model.
