# LightlyTrain API plugins for LightlyStudio

Two operators that turn a [LightlyStudio](https://github.com/lightly-ai/lightly-studio)
collection into a trained model and back:

- **LightlyTrain API training** uploads the annotated images of the current view to the
  [LightlyTrain API](../../lightly-train-api/) and retrains that dataset's model.
- **LightlyTrain API inference** predicts the current view with the model the API serves
  and writes the predictions back as annotations.

Both operators are plain HTTP clients. Neither loads a model, so LightlyStudio does not
need torch or `lightly-train` installed.

```
Studio view ──images + annotations──> POST /datasets/{name}/samples ──> training run
Studio view <──────boxes/labels────── POST /datasets/{name}/predict <── served model
```

## Install

```bash
pip install -e plugins/lightly_train_api
```

Start the API service separately, see [lightly-train-api](../../lightly-train-api/).

## Training

Uploads every image of the current view that carries an annotation. Images without one
are skipped.

- Scope: images in the current view
- Task: object detection if any annotation in the view has a bounding box, otherwise
  classification. The API trains one task per dataset.
- Labels: taken from the annotation labels, classification uses the most confident
  annotation per image
- Identity: an image is identified by its absolute path, so re-running only uploads
  images that are new, whose pixels changed, or whose annotations changed
- Output: a training run on the API, waited on by default

| Parameter           | Default                 | Meaning                                            |
| ------------------- | ----------------------- | -------------------------------------------------- |
| `api_url`           | `http://127.0.0.1:8000` | Base URL of the API service                        |
| `user_id`           | `lightly-studio`        | Owner of the dataset on the API                    |
| `dataset`           | collection name         | Dataset to train on the API                        |
| `annotation_source` | all sources             | Only train on annotations from this source         |
| `wait_for_training` | `true`                  | Wait for the run to finish before returning        |
| `timeout_s`         | `900`                   | How long to wait for the run                       |

## Inference

- Scope: images in the current view
- Output: object detection or classification annotations, matching the dataset's task
- Labels: the class names of the API model, created in the dataset if they do not exist
- Requires: a trained model for that dataset on the API

| Parameter           | Default                      | Meaning                               |
| ------------------- | ---------------------------- | ------------------------------------- |
| `api_url`           | `http://127.0.0.1:8000`      | Base URL of the API service           |
| `user_id`           | `lightly-studio`             | Owner of the dataset on the API       |
| `dataset`           | collection name              | Dataset whose model to predict with   |
| `score_threshold`   | `0.5`                        | Minimum score, applied by the API     |
| `annotation_source` | `lightly_train_api__<dataset>` | Source the predictions are written to |

## Develop

```bash
make install
make format
make type-check
```
