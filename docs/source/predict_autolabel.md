(predict-autolabel)=

# Predict & Autolabel

We provide a simple interface to perform batch prediction on a full dataset.

## Benchmark Results

The following table compares the performance of the DINOv3 EoMT models on ADE20k validation set with and without using pseudo masks from SUN397 dataset during fine-tuning.

The pseudo masks were generated as follows:

- we first fine-tuned a ViT-H+ model on the ADE20k dataset, which reaches 0.595 validation mIoU;
- we then used the checkpoint to create pseudo masks for the SUN397 dataset (~100k images);
- using these masks, we subsequently fine-tuned the smaller models listed in the table below.

| Implementation | Checkpoint Name | Val mIoU (direct FT) | Val mIoU (FT + SUN397 masks) | # Params (M) | Input Size |
|:--------------:|:------------------:|:--------------------:|:----------------------------:|:------------:|:----------:|
| LightlyTrain | dinov3/vits16-eomt-ade20k | 0.466 | 0.533 | 21.6 | 518×518 |
| LightlyTrain | dinov3/vitb16-eomt-ade20k | 0.544 | 0.573 | 85.7 | 518×518 |

We release the model checkpoints mentioned in the table above for semantic segmentation tasks. You can use these checkpoints to predict semantic segmentation masks on your own images.

## Predict Model Checkpoint

### Predict Semantic Segmentation Masks

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.predict_semantic_segmentation(
        out="out/my_experiment",
        data="my_data_dir",
        model="dinov3/vits16-eomt-ade20k",
    )
```

## Out

The following mask formats are supported:

- png

## Data

Lightly**Train** expects a folder containing images or a list of (possibly mixed) folders and image files.
Any folder will be recursively traversed and finds all image files within it (even in
nested subdirectories).

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

Example of passing a single folder `my_data_dir`:

```bash
my_data_dir
├── dir0
│   ├── image0.jpg
│   └── image1.jpg
└── dir1
    └── image0.jpg
```

## Model

The path to a semantic segmentation checkpoint created by `train_semantic_segmentation`. This can either be a path to the checkpoint or a pretrained model name.
