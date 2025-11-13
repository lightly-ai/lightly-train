(dicom-support)=

# DICOM Images

LightlyTrain supports loading [DICOM](https://www.dicomstandard.org/) images for pretraining, distillation, and fine-tuning.

```{note}
Currently, we do not support loading DICOM images as segmentation masks.
```

## PyDICOM Support

Under the hood, LightlyTrain uses the [`pydicom`](https://pydicom.github.io/pydicom/stable/index.html) library to read and process DICOM images. It is included as an optional dependency.

To install LightlyTrain with PyDICOM support, do:

```bash
pip install lightly-train[dicom]
```

For DICOM images that may require additional processing, LightlyTrain automatically applies the following using `pydicom` functions:

- converting color space from `YBR` to `RGB` via [`convert_color_space`](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.convert_color_space.html)
- decoding palette color images to `RGB` via [`apply_color_lut`](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_color_lut.html)
- rescaling images to HU values via [`apply_modality_lut`](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_modality_lut.html)

Please refer to the respective `pydicom` documentation for more details on these functions.

## Supported Image Types

The following DICOM image types listed in `pydicom.examples` are supported:

| Type | SOP Class | num_channels |
|------|-----------|--------------|
| ct | CT Image | 1 |
| mr | MR Image | 1 |
| overlay | MR Image | 1 |
| rgb_color | US Image | 3 |
| palette_color | US Image | 1 |
| jpeg2k | US Image | 3 |

Currently, LightlyTrain loads one DICOM file as one image. Combining slices from multiple DICOM files into a 3D volume is not supported. As a result, RT Dose (`rt_dose`), ECG Waveform (`waveform`), and US Multi-frame Image (`ybr_color`) are not supported.

## Transforms

When training with DICOM images, you may need to customize the applied transforms for medical domains. We recommend keeping spatial operations such as `RandomResize`, `RandomFlip`, and `RandomRotation`. We strongly suggest disabling the following transforms—even for 3D RGB DICOM files—as they often do not make sense for medical images:

- `Solarize`
- `RandomGrayScale`
- `ColorJitter`
- `ChannelDrop`

Disable them by setting the corresponding transform argument to `None`:

```python
transform_args={
    "solarize": None,
    "random_grayscale": None,
    "color_jitter": None,
    "channel_drop": None,
},
```

Also be aware that some transform options may be inappropriate depending on the acquisition protocol. For example, horizontal flips in `RandomFlip` may not be suitable for certain medical images. To disable horizontal flips:

```python
transform_args={
    "random_flip": {
        "horizontal_prob": 0.0,
    },
},
```
