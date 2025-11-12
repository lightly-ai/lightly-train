(dicom-support)=

# DICOM Image Support

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
