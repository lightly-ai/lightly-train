(predict-autolabel)=

# Predict & Autolabel

LightlyTrain provides a simple interface to perform batch prediction on a full dataset. You can use this feature to generate predictions for your unlabeled images using a pretrained model checkpoint, which can then be used as e.g. pseudo labels for further training. This allows you to improve model performance by leveraging **all your unlabeled images**.

## Benchmark Results

### Semantic Segmentation with EoMT

The following table compares the performance of the DINOv3 EoMT models on ADE20k validation set with and without using pseudo masks of SUN397 dataset during fine-tuning. You can check the [semantic segmentation docs](#semantic-segmentation) for more details on how to train such models.

The pseudo masks were generated in the following way:

- we first fine-tuned a ViT-H+ model on the ADE20k dataset, which reaches 0.595 validation mIoU;
- we then used the checkpoint to create pseudo masks for the SUN397 dataset (~100k images);
- we subsequently fine-tuned the smaller models using these masks.

The validation results are listed in the table below, where you can notice significant improvements when using the auto-labeled data:

| Implementation | Model | Autolabel | Val mIoU | Params (M) | Input Size |
|:--------------:|:-------------------------------:|:---------:|:---------:|:-----------:|:----------:|
| LightlyTrain | dinov3/vits16-eomt | ❌ | 0.466 | 21.6 | 518×518 |
| LightlyTrain | dinov3/vits16-eomt-ade20k | ✅ | **0.533** | 21.6 | 518×518 |
| LightlyTrain | dinov3/vitb16-eomt | ❌ | 0.544 | 85.7 | 518×518 |
| LightlyTrain | dinov3/vitb16-eomt-ade20k | ✅ | **0.573** | 85.7 | 518×518 |

We also released the model checkpoints fine-tuned with auto-labeled SUN397 dataset in
the table above. You can use these checkpoints by specifying the corresponding model
name in the `model` argument of the `predict_semantic_segmentation` function. See the
[Predict Semantic Segmentation Masks](#predict-semantic-segmentation) section below
for more details.

## Predict Model Checkpoint

(predict-semantic-segmentation)=

### Predict Semantic Segmentation Masks

You can use the `predict_semantic_segmentation` function to generate semantic segmentation masks for a dataset using a pretrained model checkpoint. An example command looks like this:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.predict_semantic_segmentation(
        out="out/my_experiment",
        data="my_data_dir",
        model="dinov3/vits16-eomt-ade20k", # use a pretrained checkpoint name
    )
```

or if you want to use a local checkpoint file:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.predict_semantic_segmentation(
        out="out/my_experiment",
        data="my_data_dir",
        model="path/to/my/checkpoint_file.pt", # use a local checkpoint file
    )
```

This will create predicted semantic segmentation masks for all images in the `my_data_dir` folder and save them to the `out/my_experiment` folder.

## Out

All predicted masks will be saved in the `out` folder. The subdirectory structure will follow the structure of the input `data` folder; if `data` is a list of image files, the images will be saved directly in the `out` folder.

Each predicted mask will have the same filename as the corresponding input image. The following mask formats are supported:

- png

## Data

The `data` parameter expects a folder containing images or a list of (possibly mixed) folders and image files. Any folder will be recursively traversed and finds all image files within it (even in nested subdirectories).

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

## Model

The path to a model checkpoint. This can be:

- a path to an exported checkpoint file (in `.pt`), or
- a checkpoint name that points to a model pretrained by LightlyTrain.

### Supported Checkpoint Names

The following checkpoint names are supported for semantic segmentation:

- `dinov3/vits16-eomt-ade20k`
- `dinov3/vits16-eomt-coco`
- `dinov3/vits16-eomt-cityscapes`
- `dinov3/vitb16-eomt-ade20k`
- `dinov3/vitb16-eomt-coco`
- `dinov3/vitb16-eomt-cityscapes`
- `dinov3/vitl16-eomt-coco`
- `dinov3/vitl16-eomt-cityscapes`
