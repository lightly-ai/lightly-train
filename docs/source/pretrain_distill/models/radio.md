(models-radio)=

# NVIDIA RADIO

[C-RADIO](https://github.com/NVlabs/RADIO) models are vision foundation models from
NVIDIA. They combine knowledge from multiple teacher models into a single visual
backbone for downstream vision tasks.

```{important}
NVIDIA releases C-RADIO under the
[NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf).
However, C-RADIO is distilled from multiple teacher models, and it is unclear whether
the NVIDIA Open Model License alone covers the resulting distilled models. Users should
assess the applicable terms for their use case.
```

Install the optional runtime dependencies before using a RADIO model:

```bash
pip install "lightly-train[radio]"
```

## Usage

Input height and width must each be divisible by 16. Choose a compatible `image_size`.

## Supported Models

- C-RADIOv3: `radio/c-radio_v3-b`, `radio/c-radio_v3-l`, `radio/c-radio_v3-h`,
  `radio/c-radio_v3-g`
- C-RADIOv4: `radio/c-radio_v4-so400m`, `radio/c-radio_v4-h`
