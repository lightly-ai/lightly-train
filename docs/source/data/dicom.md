(dicom-support)=

# DICOM Images

LightlyTrain supports loading [DICOM](https://www.dicomstandard.org/) images for pretraining, distillation, and fine-tuning.

## PyDICOM Support

Under the hood, LightlyTrain uses the [`pydicom`](https://pydicom.github.io/pydicom/stable/index.html) library to read and process DICOM images. It is added as an optional dependency to LightlyTrain.

To install LightlyTrain with `pydicom`, you should do:

```bash
pip install lightly-train[dicom]
```

For some types of DICOM images, additional processing might be needed. LightlyTrain does the following for you automatically with the help of `pydicom` functions:

- converting color space from YBR to RGB via [`convert_color_space`](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.convert_color_space.html)
- decoding palette color images to RGB images via [`apply_color_lut`](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_color_lut.html)
- rescaling image to HU values via [`apply_modality_lut`](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_modality_lut.html)

Please refer to the respective `pydicom` documentation for more details on these functions.

## Supported Image Types

| Type | SOP Class | num_channels | num_frames | Supported |
|----------------|-----------------------|---------------|-------------|-------------|
| ct | CT Image | 1 | — | ✅ |
| mr | MR Image | 1 | — | ✅ |
| rt_dose | RT Dose | 1 | 15 | ❌ |
| overlay | MR Image | 1 | — | ✅ |
| waveform | 12 Lead ECG | [12, 12] | — | ❌ |
| rgb_color | US Image | 3 | — | ✅ |
| palette_color | US Image | 1 | — | ✅ |
| ybr_color | US Multi-frame Image | 3 | 30 | ❌ |
| jpeg2k | US Image | 3 | — | ✅ |
