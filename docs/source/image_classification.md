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
│   ├── cat
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── car
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── ...
└── val
    ├── cat
    │   ├── img1.jpg
    │   └── ...
    ├── car
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
- Class IDs are assigned according to the passed `classes` dictionary.

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
            # Extra arguments for CSV-based datasets.
            "csv_image_column": "image_path", # Name of the column storing image paths.
            "csv_label_column": "label",      # Name of the column storing labels.
            "csv_label_type": "name",         # Type of labels either "name" or "id".
            "label_delimiter": ",",           # Delimiter used to separate the labels. 
        },
    )
```
