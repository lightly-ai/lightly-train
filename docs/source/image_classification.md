---
orphan: true
---
(image-classification)=

# Image Classification

(image-classification-data)=

## Data

LightlyTrain supports training image classification models using either a
directory-based dataset structure or CSV annotation files. Both single-label and
multi-label classification are supported.

### Image Formats

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
- dcm (DICOM)

______________________________________________________________________

### Folder-based Datasets (Single-label)

In the simplest setup, images are organized into subdirectories, where each subdirectory
corresponds to one class. The directory name defines the class name, and all images
inside that directory are assigned to that class.

Your dataset directory should be organized like this:

```text
my_data_dir/
├── train
│   ├── class_0
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class_1
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── ...
└── val
    ├── class_0
    │   ├── img1.jpg
    │   └── ...
    ├── class_1
    │   ├── img1.jpg
    │   └── ...
    └── ...
```

To train with this directory structure, set the `data` argument like this:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.image_classification(
        out="out/my_experiment",
        model="dinov3/vits16-classification",
        data={
            "train": "my_data_dir/train/",
            "val": "my_data_dir/val/",
            "classes": {                    
                0: "cat",
                1: "car",
                2: "dog",
                # ...
            },
            # Optional, classes that are in the dataset but should be ignored during
            # training.
            "ignore_classes": [0], 
        },
    )
```

In this setup:

- Each image belongs to exactly one class.
- Class names are taken from the directory names.
- Class IDs are assigned according to the names mapping passed to
  ImageClassificationDataArgs.

### CSV-based Datasets (Single-label and Multi-label)

For more flexibility, LightlyTrain also supports CSV files that explicitly map image
paths to labels. This is required for multi-label classification, and can also be used
for single-label datasets.

Each split (train, val, optionally test) must have its own CSV file.

#### CSV format

A CSV file must contain:

- one column specifying the image path
- one column specifying the label(s)

Example CSV (`train.csv`) with class IDs:

```
image_path,label
/absolute/path/to/image1.jpg,"0,2"
/absolute/path/to/image2.jpg,"1"
```

Example CSV (`train.csv`) with class names:

```
image_path,label
/absolute/path/to/image1.jpg,"cat,dog"
/absolute/path/to/image2.jpg,"car"
```

To train with this CSV-based structure, set the `data` argument like this:

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.image_classification(
        out="out/my_experiment",
        model="dinov3/vits16-classification",
        data={
            "train_csv": "my_data_dir/train.csv",
            "val": "my_data_dir/val.csv",
            "classes": {                   
                0: "cat",
                1: "car",
                2: "dog",
                # ...
            },
            # Optional, classes that are in the dataset but should be ignored during
            # training.
            "ignore_classes": [0], 
        },
    )
```

Notes:

- Image paths must be absolute paths.
- Multiple labels are separated by a delimiter (default: ,).
- When using commas as label delimiters, the label field must be quoted.
- Labels can be specified either as class IDs or class names.

#### Supported CSV Options

The behavior of CSV parsing can be configured via the `data` argument:

- `csv_image_col`: Name of the image path column (default: `"image_path"`).
- `csv_label_col`: Name of the label column (default: `"label"`).
- `csv_label_type`: "name" or `"id"` (default: `"name"`).
- `label_delimiter`: Delimiter for multiple labels (default: `","`).
