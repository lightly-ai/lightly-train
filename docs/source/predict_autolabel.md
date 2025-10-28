(predict-autolabel)=

# Predict & Autolabel

## Benchmark Results

| Implementation | Backbone Model | Val mIoU (direct FT) | Val mIoU (FT + SUN397 masks) | # Params (M) | Input Size | Checkpoint Name |
|:--------------:|:------------------:|:--------------------:|:----------------------------:|:------------:|:----------:|:-------------------------:|
| LightlyTrain | dinov3/vits16-eomt | 0.466 | 0.533 | 21.6 | 518×518 | dinov3/vits16-eomt-ade20k |
| LightlyTrain | dinov3/vitb16-eomt | 0.544 | 0.573 | 85.7 | 518×518 | dinov3/vitb16-eomt-ade20k |

The better results for the respective models were achieved by fine-tuning a ViT-H+ on the ADE20k dataset, which reaches 0.595 validation mIoU. We then used the checkpoint to create pseudo masks for the SUN397 dataset (~100k images). Using these masks, we subsequently fine-tuned the smaller models, and then used the ADE20k dataset for validation.

## Predict Model Checkpoint

### Predict Semantic Segmentation Masks

## Out

## Data

Lightly**Train** supports training semantic segmentation models with images and masks.
Every image must have a corresponding mask whose filename either matches that of the image (under a different directory) or follows a specific template pattern. The masks must be PNG images in either grayscale integer format, where each pixel value corresponds to a class ID, or multi-channel (e.g., RGB) format.

The following image formats are supported:

- jpg
- jpeg
- png
- ppm
- bmp
- pgm
- tif
- tiff
- webp

The following mask formats are supported:

- png
