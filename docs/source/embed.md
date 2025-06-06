(embed)=

# Embed

The `embed` command is used to embed images with a [training checkpoint](#train-output).
An example command looks like this:

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",
        data="my_data_dir",
        model="torchvision/resnet50",
    )

    lightly_train.embed(
        out="my_embeddings.pth",                            
        data="my_data_dir",                                 
        checkpoint="out/my_experiment/checkpoints/last.ckpt",
        format="torch",
    )
````

````{tab} Command Line
```bash
lightly-train train out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50"
lightly-train embed out=my_embeddings.pth data=my_data_dir checkpoint=out/my_experiment/checkpoints/last.ckpt format=torch
````

The above code example trains a model and uses the last training checkpoint to generate
image embeddings from the images in `"my_data_dir"`.

```{tip}
See {meth}`lightly_train.embed` for a complete list of arguments.
```

## Out

The `out` argument specifies the output file where the embeddings and corresponding
image filenames are saved. Image filenames are always relative to the `data` directory.
See [format](#embedding-format) for details on how the embeddings are saved.

The embedding dimension is determined by the model used for training. If the `embed_dim`
argument was set during training, the embeddings will have this dimension, otherwise
the default embedding dimension of the model is used.

## Data

The `data` argument specifies the directory with images to embed. The same image formats
as in the [training command](#train-data) are supported.

## Checkpoint

The `checkpoint` argument specifies the Lightly**Train** checkpoint to use for embedding.
This is the checkpoint saved to `out/my_experiment/checkpoints/last.ckpt` after
training.

(embedding-format)=

## Format

The `format` argument specifies the format in which the embeddings are saved. The
following formats are supported:

- `csv`: Embeddings are saved as a CSV file with one row per image. The first column
  contains the filename and the remaining columns contain the embedding values.
  An example CSV file looks like this:
  ```text
    filename,embedding_0,embedding_1,embedding_2
    image1.jpg,0.1,0.2,0.3
    image2.jpg,0.4,0.5,0.6
  ```
- `lightly_csv`: Embeddings are saved as a CSV file with one row per image. The CSV file
  is compatible with the [Lightly Worker](https://docs.lightly.ai/docs/custom-embeddings).
  The labels column is always set to 0. An example CSV file looks like this:
  ```text
    filenames,embedding_0,embedding_1,embedding_2,labels
    image1.jpg,0.1,0.2,0.3,0
    image2.jpg,0.4,0.5,0.6,0
  ```
- `torch`: The embeddings are saved as dictionary in a torch file with the following
  structure:
  ```python
  from torch import Tensor

  {
      "embeddings": Tensor,    # Embeddings with shape (N, D) where N is the number
                                     # of images and D is the embedding dimension.
      "filenames": list[str],        # List of N filenames corresponding to the embeddings.
  }
  ```
  The embeddings can be loaded with:
  ```python
    import torch

    embeddings = torch.load("my_embeddings.pth")
    embeddings["embeddings"]    # Embeddings
    embeddings["filenames"]     # Image filenames
  ```
