(method-transform-args)=

# Configuring Image Transforms

Pretraining relies strongly on image transforms (augmentations) such as:

- **Channel Drop**: Randomly drops channels from the image.
- **Random Cropping and Resizing**: Crops random parts of images and resizes them to fixed
  resolutions.
- **Random Horizontal and Vertical Flipping**: Mirrors images across horizontal or vertical
  axes.
- **Random Rotation**: Rotates images by random angles.
- **Color Jittering**: Randomly modifies brightness, contrast, saturation, and hue.
- **Random Grayscaling**: Converts images to grayscale with certain probability.
- **Gaussian Blurring**: Applies Gaussian blur filter of random {math}`\sigma`, smoothing
  the image.
- **Random Solarization**: Inverts pixel values above a random threshold.
- **Normalization**: Scales pixel values using predefined mean and standard deviation.

```{warning}
In 99% of cases, it is not necessary to modify the default image transforms in
LightlyTrain. The default settings are carefully tuned to work well for most use cases.
However, for specific downstream tasks or unique image domains, you might want to
override these defaults as shown below.
```

````{tab} Python
For the Python API, use a dictionary structure to override any transforms settings and 
pass it to the `lightly_train.train` function through the `transform_args` argument. Many 
transforms can also be selectively turned off completely by setting them to `None`, as 
is demonstrated in this example with the `color_jitter` augmentation.
```python
import lightly_train
my_transform_args = {
    "random_resize": {
        "min_scale": 0.1
    },
    "image_size": (128, 128),
    "color_jitter": None,
}
if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",            # Output directory
        data="my_data_dir",                 # Directory with images
        model="torchvision/resnet18",       # Model to train
        transform_args=my_transform_args,   # Overrides of default augmentation parameters
    )
```
````

````{tab} Command Line
There are two options on how you can configure the transforms on the command line:
1. Dotted Notation
2. Pass all arguments as a single JSON structure

```{important}
Make sure that any values that you pass through the command line are JSON-compatible. This 
means:
 - Strings inside JSON structures must have double quotes (wrap the whole structure by 
   single quotes).
 - Tuples do not exist, use bracketed notation (like a Python list).
 - JSON's correspondence to Python's `None` is `null`, which you will have to use in order 
   to turn off an augmentation.
```

An example of how you can use the bracketed notation, would be:
```bash
lightly-train train \
    out="out/my_experiment" \
    overwrite=True \
    data="my_data_dir" \
    model="torchvision/resnet18" \
    transform_args.image_size="[128,128]" \
    transform_args.random_resize.min_scale=0.1 \
    transform_args.color_jitter=null
```

And an example of using a single JSON structure would look as follows:

```bash
lightly-train train \
    out="out/my_experiment" \
    data="my_data_dir" model="torchvision/resnet18" \
    transform_args='{"image_size": [128, 128], "random_resize": {"min_scale": 0.1}, "color_jitter": null}'
```
````

The next sections will cover which arguments are available across all methods, and also the
arguments unique to specific methods.

```{seealso}
Interested in the default augmentation settings for each method? Check the method pages:
 - {ref}`methods-distillation`
 - {ref}`methods-dinov2`
 - {ref}`methods-dino`
 - {ref}`methods-simclr`
```

## Arguments available for all methods

The following arguments are available for all methods.

### Channel Drop

Randomly drops channels from the image. Can be disabled by setting to `None`.
Disabled by default. Only use if you have images with more than 3 channels. Requires
`LIGHTLY_TRAIN_IMAGE_MODE="UNCHANGED"` to be set in the environment.

```python skip_ruff
"channel_drop": {
    "num_channels_keep": int,                 # number of channels to keep in the image
    "weight_drop": tuple[float, ...],         # weight for each channel to be dropped
                                              # 0 means never dropped, higher values mean 
                                              # higher probability of being dropped
}
```

### Random Cropping and Resizing

Can be disabled by setting to `None`.

```python skip_ruff
"random_resize": {
    "min_scale": float,
    "max_scale": float,
}
```

### Image Size

Cannot be disabled, required for all transforms.

```python skip_ruff
"image_size": tuple[int, int]  # height, width
```

### Random Horizontal and Vertical Flipping

Can be disabled by setting to `None`.

```python skip_ruff
"random_flip": {
    "horizontal_prob": float, # probability of applying horizontal flip
    "vertical_prob": float,   # probability of applying vertical flip
}
```

### Random Rotation

Can be disabled by setting to `None`.

```python skip_ruff
"random_rotation": {
    "prob": float,   # probability of applying rotation
    "degrees": int,  # maximum rotation angle in degrees
}
```

### Color Jittering

Can be disabled by setting to `None`.

```python skip_ruff
"color_jitter": {
    "prob": float,       # probability of applying color jitter
    "strength": float,   # multiplier for all parameters below
    "brightness": float, # how much to jitter brightness (non-negative)
    "contrast": float,   # how much to jitter contrast (non-negative)
    "saturation": float, # how much to jitter saturation (non-negative)
    "hue": float,        # how much to jitter hue (non-negative)
}
```

### Random Grayscaling

Can be disabled by setting to `None`.

```python skip_ruff
"random_gray_scale": float  # probability of converting to grayscale
```

### Gaussian Blurring

Can be disabled by setting to `None`.

```python skip_ruff
"gaussian_blur": {
    "prob": float,                     # probability of applying blur
    "sigmas": tuple[float, float],          # range of sigma values
    "blur_limit": int | tuple[int, int],    # range of kernel size, either [0, high] or [low, high]
}
```

### Random Solarization

Can be disabled by setting to `None`.

```python skip_ruff
"solarize": {
    "prob": float,      # probability of applying solarization
    "threshold": float  # threshold value in range [0, 1]
}
```

### Normalization

Cannot be disabled, required for all transforms.

```python skip_ruff
"normalize": {
    "mean": tuple[float, float, float],  # means of the three channels
    "std": tuple[float, float, float]    # standard deviations of the three channels
}
```

## Arguments unique to methods

The methods Distillation and SimCLR have no transform configuration options beyond the
globally available ones, which were listed above.

### DINO

DINO uses a multi-crop strategy with two "global" views (which have slightly
different augmentation parameters) and optional additional smaller resolution "local" views
(default: 6 local views).

Besides the default arguments, the following DINO-specific arguments are available:

```python skip_ruff
"global_view_1": {                     # modifications for second global view (cannot be disabled)
    "gaussian_blur": {                 # can be disabled by setting to None
        "prob": float,                 
        "sigmas": tuple[float, float],
        "blur_limit": int | tuple[int, int]
    },
    "solarize": {                      # can be disabled by setting to None
        "prob": float,
        "threshold": float
    }
},
"local_view": {                        # configuration for local views (can be disabled by setting to None)
    "num_views": int,                  # number of local views to generate
    "view_size": tuple[int, int],      # size of local views
    "random_resize": {                 # can be disabled by setting to None
        "min_scale": float,
        "max_scale": float
    },
    "gaussian_blur": {                 # can be disabled by setting to None
        "prob": float,
        "sigmas": tuple[float, float],
        "blur_limit": int | tuple[int, int]
    }
}
```

Note that `local_view` itself can be disabled by setting it to `None`. Additionally, some
transforms within these structures can be disabled by setting them to `None`
