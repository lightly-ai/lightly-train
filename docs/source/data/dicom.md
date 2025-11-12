(dicom-support)=

# DICOM Images

LightlyTrain supports loading [DICOM](https://www.dicomstandard.org/) images for pretraining, distillation, and fine-tuning.

```{note}
Currently, we do not support loading DICOM images as segmentation masks.
```

## PyDICOM Support

Under the hood, LightlyTrain uses the [`pydicom`](https://pydicom.github.io/pydicom/stable/index.html) library to read and process DICOM images. It is added as an optional dependency to LightlyTrain.

To install LightlyTrain with `pydicom`, you should do:

```bash
pip install lightly-train[dicom]
```

For some types of DICOM images, additional processing might be needed. LightlyTrain does the following for you automatically with the help of `pydicom` functions:

- converting color space from `YBR` to `RGB` via [`convert_color_space`](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.convert_color_space.html)
- decoding palette color images to `RGB` images via [`apply_color_lut`](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_color_lut.html)
- rescaling image to HU values via [`apply_modality_lut`](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_modality_lut.html)

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

For now, LightlyTrain only supports loading one DICOM file as one image. Combining slices from multiple DICOM files within a directory into one 3D image is not supported. Therefore, RT Dose (`rt_dose`), ECG Waveform (`waveform`), and US Multi-frame Image (`ybr_color`) images are not supported.
